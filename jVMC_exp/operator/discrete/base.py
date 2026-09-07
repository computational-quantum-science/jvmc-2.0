from __future__ import annotations
from abc import abstractmethod
import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix

from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import sharded, DEVICE_SPEC
from jVMC_exp.operator.base import AbstractOperator

class Operator(AbstractOperator):
    def __init__(self, ldim):
        self._ldim = ldim
        self._is_compiled = False
        self._scale = 1

    @property
    def ldim(self):
        return self._ldim
    
    def __add__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            # TODO: Since I don't know the total dim of the Hilbert space this is not doable
            raise NotImplementedError 
        elif isinstance(other, Operator):
            return self._create_composite(self, other, 'sum')
        else:
            raise NotImplemented
        
    def __radd__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            # TODO: Same as previous todo
            raise NotImplementedError
        else:
            raise NotImplemented
        
    def __neg__(self) -> Operator:
        return self._create_scaled(self, -1)
    
    def __sub__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        elif isinstance(other, Operator):
            return self._create_composite(self, -other, 'sum')
        else:
            raise NotImplemented
        
    def __rsub__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        else:
            raise NotImplemented
        
    def __mul__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)) or callable(other):
            return self._create_scaled(self, other)
        elif isinstance(other, Operator):
            return self._create_composite(self, other, 'mul')
        else:
            raise NotImplemented
        
    def __rmul__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)) or callable(other):
            return self._create_scaled(self, other)
        else:
            raise NotImplemented
        
    def __truediv__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise ZeroDivisionError("Division of Operator by zero.")
            return self._create_scaled(self, 1 / other)
        else:
            return NotImplemented
        
    def get_O_loc(self, s, psi: NQS, *, logPsiS=None, **kwargs):
        s_p, matEls = self.get_conn_elements(s, psi.batchSize, **kwargs)

        if psi.eval_ratio:
            logPsi_ratio = psi.call_ratio(
                jnp.repeat(s, matEls.shape[1], axis=0), s_p.reshape((-1, *psi.sampleShape))
            ).reshape(matEls.shape)

            return self._get_O_loc_ratio(logPsi_ratio, matEls, batch_size=psi.batchSize) 
        
        logPsiS = psi(s) if logPsiS is None else logPsiS
        logPsiS_p = psi(s_p.reshape((-1, *psi.sampleShape))).reshape(matEls.shape)

        return self._get_O_loc(logPsiS, logPsiS_p, matEls, batch_size=psi.batchSize) 
    
    @sharded(use_vmap=False)
    def _get_O_loc(self, logPsiS, logPsiS_p, matEls, *, batch_size):
        return jnp.sum(jnp.exp(logPsiS_p - logPsiS[:, None]) * matEls, axis=1)

    @sharded(use_vmap=False)
    def _get_O_loc_ratio(self, logPsi_ratio, matEls, *, batch_size):
        return jnp.sum(logPsi_ratio * matEls, axis=1)

    def get_conn_elements(self, s, batch_size, **kwargs):
        if not self._is_compiled:
            self._compile()

        return self._get_conn_elements_sh(s, batch_size=batch_size, **kwargs)

    def to_dense(self, basis, *, zero_tolerance=0.0, **op_kwargs):
        return self.to_sparse(basis, zero_tolerance=0.0, **op_kwargs).todense()

    def to_sparse(self, basis, *, zero_tolerance=0.0, **op_kwargs):
        """
        Return the SciPy sparse matrix representation of this operator
        in the supplied ordered computational basis.

        Parameters
        ----------
        basis : array_like
            Ordered computational basis with shape

                (num_states, *sample_shape).

            The basis must be closed under every nonzero action of the
            operator. Zero-weight padded connections returned by the
            operator are ignored.

        zero_tolerance : float, optional
            Connections with absolute matrix element less than or equal
            to this value are discarded. Default is zero.

        **op_kwargs
            Runtime arguments forwarded to `get_conn_elements`.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix representation of the operator.
        """
        if zero_tolerance < 0:
            raise ValueError(
                "zero_tolerance must be nonnegative. "
                f"Got {zero_tolerance}."
            )

        basis_array = jnp.asarray(basis)
        if basis_array.ndim < 2:
            raise ValueError(
                "basis must have shape (num_states, *sample_shape). "
                f"Got {basis_array.shape}."
            )

        dim = basis_array.shape[0]
        if dim == 0:
            raise ValueError(
                "Cannot construct a matrix from an empty basis. "
                f"Got {basis_array.shape}."
            )

        # SciPy and the Python lookup dictionary operate on CPU arrays
        flat_basis = jax.device_get(basis_array).reshape(dim, -1)
        basis_index = {
            tuple(configuration.tolist()): index
            for index, configuration in enumerate(flat_basis)
        }
        if len(basis_index) != dim:
            raise ValueError(
                "The supplied basis contains duplicate configurations."
            )

        n_devices = jax.device_count()
        s_primes, mat_els = self.get_conn_elements(
            basis_array,
            ((dim + n_devices - 1) // n_devices) * n_devices,
            **op_kwargs,
        )
        s_primes_host = jax.device_get(s_primes)
        mat_els_host = jax.device_get(mat_els)
        flat_s_primes = s_primes_host.reshape(-1, flat_basis.shape[1])
        flat_mat_els = mat_els_host.reshape(-1)

        if flat_s_primes.shape[0] != flat_mat_els.shape[0]:
            raise RuntimeError(
                "Incompatible get_conn_elements output shapes: "
                f"s_primes={s_primes_host.shape}, "
                f"mat_els={mat_els_host.shape}."
            )

        if flat_mat_els.size % dim != 0:
            raise RuntimeError(
                "The number of returned matrix elements is not "
                "divisible by the basis dimension: "
                f"{flat_mat_els.size} versus {dim}."
            )

        n_conn = flat_mat_els.size // dim

        all_cols = jax.device_get(
            jnp.repeat(jnp.arange(dim), n_conn,)
        )

        rows = []
        cols = []
        data = []
        missing_connections = []
        for connected_configuration, matrix_element, col in zip(
            flat_s_primes,
            flat_mat_els,
            all_cols,
        ):
            if abs(matrix_element) <= zero_tolerance:
                continue

            configuration_key = tuple(connected_configuration.tolist())
            row = basis_index.get(configuration_key)

            if row is None:
                missing_connections.append(
                    (
                        int(col),
                        tuple(flat_basis[col].tolist()),
                        configuration_key,
                        matrix_element,
                    )
                )
                continue

            rows.append(row)
            cols.append(int(col))
            data.append(matrix_element)

        if missing_connections:
            raise ValueError(
                "The operator has nonzero connections outside "
                "the supplied basis."
            )

        matrix = coo_matrix(
            (
                jax.device_get(jnp.asarray(data)),
                (
                    jax.device_get(jnp.asarray(rows)),
                    jax.device_get(jnp.asarray(cols))
                ),
            ),
            shape=(dim, dim),
        ).tocsr()

        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        return matrix

    @sharded(out_specs=(DEVICE_SPEC, DEVICE_SPEC))
    def _get_conn_elements_sh(self, s, *, batch_size, **kwargs):
        return  self._get_conn_elements(s, kwargs)
        
    @abstractmethod
    def _compile(self):
        """
        Compile the operator into JAX arrays for efficient computation.
        """
        pass
    
    @abstractmethod
    def _get_conn_elements(self, s, kwargs):
        """
        Compute the connected configurations and corresponding matrix elements
        generated by the action of an operator on a given input configuration.

        This method must return:
        (i) all non-diagonal connected configurations with shape (NumConnectedElements, SampleShape),
        (ii) their associated matrix elements with shape (SampleShape,), and
        (iii) the total diagonal contribution.

        Parameters
        ----------
        s : Single input configuration 

        kwargs : Auxiliary parameters required to compute operator prefactors.

        Returns
        -------
        s_p_non_diag : Array of configurations connected to `s`.

        mat_els_non_diag : One-dimensional array of complex matrix elements corresponding
            to each configuration in `s_p_non_diag`.

        mat_els_diag : Scalar equal to the sum of all diagonal matrix elements.
        """
        pass
    
    @classmethod
    @abstractmethod
    def _create_composite(cls, O_1, O_2, label):
        pass

    @classmethod
    @abstractmethod
    def _create_scaled(cls, O, scalar):
        pass