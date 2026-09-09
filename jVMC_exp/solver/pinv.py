import jax.numpy as jnp
import jax
from typing import Literal

from jVMC_exp.solver.base import AbstractSolver
from jVMC_exp.solver.util import diagonalize, smooth_cutoff_fn
from jVMC_exp.stats import LazySampledObs, SampledObs

@jax.jit
def _get_snr(Vtb, Vtb_var, num_samples):
    return jnp.sqrt(
        jnp.abs(num_samples * (jnp.conj(Vtb) * Vtb) / (Vtb_var + 1e-14))
    ).ravel()

@jax.jit(static_argnums=(4,))
def _get_vtb_var(centered_data_1, centered_data_2, weights, eigenvectors, transformation):
    @jax.vmap
    def _get_covar_per_sample(A, B):
        return jnp.outer(jnp.conj(A), B)

    x = _get_covar_per_sample(centered_data_1, centered_data_2).squeeze(-1)
    x = transformation(x)
    rho = jnp.einsum("ij, ki -> kj", jnp.conjugate(eigenvectors), x)
    rho = rho - jnp.tensordot(weights, rho, axes=(0, 0))

    return jnp.tensordot(weights, jnp.abs(rho)**2, axes=(0, 0))

def _unpad(x, pad_size):
    return x[:-pad_size] if pad_size else x

def _snr_step(
        grad_log_psi: SampledObs, o_loc: SampledObs, 
        V, Vtb, transformation, pad_size
    ):
    if isinstance(grad_log_psi, LazySampledObs):
        raise NotImplementedError(
            "PinvSNR's signal-to-noise regularization (snr_tol != 0) requires a densely "
            "materialized Jacobian, but grad_log_psi is a LazySampledObs (a batched/lazy "
            "Jacobian). Construct the Observable/objective function with "
            "batched_jacobian=False to enable SNR regularization, or set "
            "snr_tol=0 to disable it."
        )
    if grad_log_psi is None or o_loc is None or transformation is None:
        missing = [
            name for name, value in (
                ("grad_log_psi", grad_log_psi), ("o_loc", o_loc), ("transformation", transformation)
            ) if value is None
        ]
        raise ValueError(
            f"PinvSNR has snr_tol != 0, but the following required argument(s) for the "
            f"exact SNR computation were not given: {', '.join(missing)}. Pass "
            "grad_log_psi, o_loc, and transformation to PinvSNR.__call__, "
            "or set snr_tol=0 to disable SNR regularization."
        )

    if pad_size != 0:
        grad_log_psi.transform(
            lambda x: jnp.pad(x, ((0, 0), (0, pad_size)), mode="constant")
        )

    Vtb_var = _unpad(
        _get_vtb_var(
            grad_log_psi._centered_obs, 
            o_loc._centered_obs, 
            o_loc.weights, 
            V, 
            transformation
        ),
        pad_size
    )

    if pad_size != 0:
        grad_log_psi.transform(lambda x: x[:,:-pad_size])

    return _get_snr(Vtb, Vtb_var, o_loc.effective_num_samples)

class Pinv(AbstractSolver):
    """
    Pseudo-inverse solver based on an eigenvalue decomposition of `A`.

    Solves ``A @ x = b`` as ``V (ev^-1 * (V^H b))``, i.e. without ever assembling
    ``A^+`` itself. Besides saving the extra ``V diag(ev^-1) V^H`` product, this keeps
    the eigenvectors in whatever (possibly sharded, possibly padded) layout the
    diagonalization produced them in; see :func:`jVMC_exp.solver.util.diagonalize`.

    Parameters
    ----------
    pinv_cutoff : float, default=1e-14
        Relative eigenvalue cutoff. Eigenvalues with
        :math:`|\\lambda_i / \\lambda_{max}| \\leq` `pinv_cutoff` are treated as zero
        and dropped from the pseudo-inverse. Note that :class:`PinvSNR` uses this same
        name for the *minimum* of its adaptively lowered cutoff, and reserves
        ``pinv_tol`` for a target residual -- the two classes' ``pinv_tol`` are not
        interchangeable.

    diagonalization_mode : {"device", "distributed", "host"}, default="device"
        Backend used to diagonalize `A`; forwarded to
        :func:`jVMC_exp.solver.util.diagonalize`.

    T_A : int, optional
        Tile size forwarded to :func:`jVMC_exp.solver.util.diagonalize`
        for ``diagonalization_mode="distributed"``. If not given, a tile
        size is chosen automatically.
    """
    def __init__(
            self, pinv_cutoff=1e-14,
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None
        ):
        self._pinv_cutoff = pinv_cutoff
        self._diagonalization_mode = diagonalization_mode
        self._T_A = T_A

    @property
    def pinv_cutoff(self):
        return self._pinv_cutoff
    
    @property
    def _needs_dense_matrix(self) -> bool:
        return True

    def _diagonalize(self, A, b, pad_size):
        ev, V = diagonalize(
            A, pad_size, mode=self._diagonalization_mode, T_A=self._T_A
        )
        ev = _unpad(ev, pad_size)
        if jnp.max(jnp.abs(ev)) < 1e-14:
            raise RuntimeError(
                f"Largest eigenvalue of the matrix A to invert is {jnp.max(jnp.abs(ev))}. "
                "A is most likely highly ill-conditioned/zero "
            )
        
        Vtb = _unpad(
            jnp.dot(jnp.transpose(jnp.conj(V)), jnp.pad(b, (0, pad_size))),
            pad_size
        )

        return ev, V, Vtb

    def __call__(self, A, b, *, pad_size=0, **kwargs):
        ev, V, Vtb = self._diagonalize(A, b, pad_size)
        
        inv_ev = jnp.where(jnp.abs(ev / ev[-1]) > self.pinv_cutoff, 1. / ev, 0.)
        b_norm = jnp.linalg.norm(b) 
        residual = jnp.linalg.norm((inv_ev * ev - 1) * Vtb) / b_norm

        x = _unpad(
            jnp.dot(V, jnp.pad((inv_ev * Vtb), (0, pad_size))),
            pad_size
        )

        info = dict(
            condition_number=(ev[-1] / jnp.min(jnp.abs(ev))).item(),
            residual=residual.item(),
        )

        return x, info

class PinvSNR(Pinv):
    """
    Pseudo-inverse solver based on an eigenvalue decomposition of the covariance
    matrix.

    The inverse is regularized using two smooth filters:

    - an eigenvalue cutoff controlled by ``pinv_cutoff``, which suppresses
      ill-conditioned directions of the covariance matrix;
    - a signal-to-noise ratio (SNR) cutoff controlled by ``snr_tol``, which
      suppresses statistically unresolved update directions. This regularization
      is skipped by default. Change snr_tol to activate it; doing so requires
      passing ``grad_log_psi``, ``o_loc``, and ``transformation`` to
      :meth:`__call__`, and requires a densely materialized (non-batched)
      Jacobian.

    The effective eigenvalue cutoff is chosen adaptively such that the residual
    force discarded by the regularization is below ``pinv_tol`` whenever
    possible.

    Parameters
    ----------
    snr_tol : float, default=0
        Minimum signal-to-noise ratio of an eigenmode before it contributes
        significantly to the update.

    pinv_tol : float, default=1e-14
        Target residual. The solver decreases the eigenvalue cutoff until the
        fraction of discarded force falls below this threshold or the minimum
        cutoff ``pinv_cutoff`` is reached.

    pinv_cutoff : float, default=1e-8
        Minimum allowed relative eigenvalue cutoff.

    diagonalization_mode : {"device", "distributed", "host"}, default="device"
        Backend used to diagonalize the covariance matrix; forwarded to
        :func:`jVMC_exp.solver.util.diagonalize`. ``"device"`` uses
        ``jax.numpy.linalg.eigh``; ``"distributed"`` uses a distributed
        eigensolver across the device mesh, falling back to ``"device"``
        if the prerequisites for distributed execution aren't met;
        ``"host"`` uses ``numpy.linalg.eigh``.

    T_A : int, optional
        Tile size forwarded to :func:`jVMC_exp.solver.util.diagonalize`
        for ``diagonalization_mode="distributed"``. If not given, a tile
        size is chosen automatically.
    """
    def __init__(
            self, snr_tol=0, pinv_tol=1e-14, pinv_cutoff=1e-8, 
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None
        ):
        super().__init__(pinv_cutoff, diagonalization_mode, T_A)

        self._snr_tol = snr_tol
        self._pinv_tol = pinv_tol

    @property
    def snr_tol(self):
        return self._snr_tol
    
    @property
    def pinv_tol(self):
        return self._pinv_tol

    def __call__(
            self, A, b,
            *,
            grad_log_psi: None | SampledObs = None, o_loc: None | SampledObs = None,
            transformation=None, pad_size=0, exact_sampler=False,
            **kwargs
        ):
        """
        Solve ``A @ x = b`` for ``x`` via a regularized pseudo-inverse of `A`.

        Parameters
        ----------
        A : array_like
        b : array_like
        grad_log_psi : SampledObs, optional
            Dense per-sample log-derivatives :math:`O_k(s_n)`. Required,
            together with `o_loc` and `transformation`, to compute the exact
            SNR-based regularization whenever `snr_tol` is nonzero and
            `exact_sampler` is False. Must be a dense `SampledObs` -- a
            `LazySampledObs` (batched Jacobian) raises `NotImplementedError`,
            since the exact SNR needs the materialized per-sample Jacobian.
        o_loc : SampledObs, optional
            Per-sample local estimator (e.g. local energies) that
            `grad_log_psi` is correlated against to build `b`. Required
            together with `grad_log_psi` and `transformation` for the SNR
            regularization.
        transformation : callable, optional
            The same real/imaginary-part transform used to build `b` from
            the raw force estimator (``Evolution._rhs_trans_fn``). Required
            together with `grad_log_psi` and `o_loc` for the SNR
            regularization; must be identical to the transform used
            upstream to compute `b`, since applying the wrong transform (or
            applying it after projecting onto `A`'s eigenbasis instead of
            before) silently gives an incorrect variance whenever that
            eigenbasis is complex.
        pad_size : int
            Number of trailing rows/columns `A` needs to be padded with so
            that its size is compatible with the diagonalization backend;
            see :func:`jVMC_exp.solver.util.diagonalize`.
        exact_sampler : bool
            If True, disables the SNR-based regularization.
        **kwargs
            Ignored; accepted for interface compatibility with other
            solvers.

        Returns
        -------
        x : jax.Array
            The regularized pseudo-inverse update ``A^+ @ b``.
        info : dict
            Diagnostic information about the solve: ``residual``,
            ``pinv_cutoff``, ``snr``, ``condition_number``, ``spectrum``
            (eigenvalues of `A`), and ``effective_rank``.
        """
        # Keep V padded so that it doesn't get replicated on devices
        ev, V, Vtb = self._diagonalize(A, b, pad_size)

        snr = None
        if not exact_sampler and self.snr_tol:
            snr = _snr_step(
                grad_log_psi, o_loc, V, Vtb, transformation, pad_size
            )

        # Discard eigenvalues below numerical precision
        invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

        b_norm = jnp.linalg.norm(b) 
        residual, cutoff, pinvEv, effective_rank = self._regularize(
            snr, ev, invEv, Vtb, b_norm, exact_sampler
        )

        x = _unpad(
            jnp.dot(V, jnp.pad((pinvEv * Vtb), (0, pad_size))),
            pad_size
        )

        info = dict(
            residual=residual.item(),
            pinv_cutoff=cutoff.item(),
            snr=snr,
            condition_number=(ev[-1] / jnp.min(jnp.abs(ev))).item(),
            spectrum=ev,
            effective_rank=effective_rank.item()
        )

        return x, info

    @jax.jit(static_argnums=(0, 6))
    def _regularize(self, snr, eigenvalues, invEv, Vtb, b_norm, exact_sampler):
        def _step(cutoff):
            cutoff = jnp.maximum(0.8 * cutoff, self.pinv_cutoff)
            regularizer = smooth_cutoff_fn(jnp.abs(eigenvalues / eigenvalues[-1]), cutoff)

            if not exact_sampler and snr is not None:
                regularizer = regularizer * smooth_cutoff_fn(snr, self.snr_tol)
            
            pinvEv = invEv * regularizer
            residual = jnp.linalg.norm((pinvEv * eigenvalues - 1) * Vtb) / b_norm

            return residual, cutoff, pinvEv, jnp.mean(regularizer)

        def _cond(state):
            residual, cutoff, _, _ = state
            return (residual > self.pinv_tol) & (cutoff > self.pinv_cutoff)

        return jax.lax.while_loop(_cond, lambda s: _step(s[1]), _step(1e-2))