import jax
import jax.numpy as jnp
from typing import Callable

from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.sampler import ExactSampler
from jVMC_exp.stats import SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput, AbstractObjectiveFunction
from jVMC_exp.sharding_config import sharded, MESH, REPLICATED_SHARDING
from jVMC_exp.util import OutputManager
from jVMC_exp.solver.base import AbstractSolver
from jVMC_exp.solver import Pinv

def _interleave_re_im(arr):
    """
    Real array of shape (2N, ...) with the real and imaginary part of each row of ``arr`` next
    to each other to reduce device communication.
    """
    return jnp.stack([jnp.real(arr), jnp.imag(arr)], axis=1).reshape((2 * arr.shape[0],) + arr.shape[1:])

_concat_nonholo = jax.jit(_interleave_re_im)

@jax.jit(static_argnums=(3,))
def _normalize_batch(batch, weights, mean, concat):
    batch = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), batch - mean)
    if concat:
        batch = _interleave_re_im(batch)

    return batch

@jax.jit
def _take_columns(matrix, positions):
    return jnp.take(matrix, positions, axis=1)

def _redistribute(grad):
    """
    Turns the local block of ``grad``, sharded along the sample axis, into all its samples
    and this device's shard of the parameters.
    """
    return jax.lax.all_to_all(grad, 'devices', split_axis=1, concat_axis=0, tiled=True)

def _row_blocks(n_local_rows):
    """
    Number of blocks in which the rows of the tangent kernel owned by a device are computed:
    the largest number not larger than the number of devices that divides ``n_local_rows``.
    """
    return max(k for k in range(1, MESH.size + 1) if n_local_rows % k == 0)

def _blockwise_tangent_kernel(grad_l, grad_r=None):
    """
    Rows of ``grad_l @ conj(grad_r).T`` owned by this device, where ``grad_l`` and ``grad_r``
    are the local blocks of matrices sharded along the sample axis. ``grad_r=None`` means
    ``grad_r = grad_l``.

    After `_redistribute`, each device holds a partial sum over its own shard of the
    parameters, and the partial sums are reduced across devices one block of rows at a time.
    With as many blocks as devices, the partial sum of a block has as many rows as the share
    of the kernel of one device, so no device ever holds a matrix of the size of the full kernel.
    """
    n_devices = MESH.size
    n_local_rows, n_params = grad_l.shape
    n_blocks = _row_blocks(n_local_rows)
    block = n_local_rows // n_blocks
    dtype = jnp.result_type(grad_l, grad_l if grad_r is None else grad_r)

    if grad_r is None:
        grad_r = _redistribute(grad_l)
        grad_blocks = grad_r.reshape(n_devices, n_blocks, block, grad_r.shape[1])
        block_rows = lambda j: jax.lax.dynamic_index_in_dim(
            grad_blocks, j, axis=1, keepdims=False
        ).reshape(n_devices * block, grad_r.shape[1])
    else:
        grad_r = _redistribute(grad_r)
        # Redistributing grad_l one block at a time avoids holding all of it, redistributed,
        # next to grad_r. Block j of the rows of every device arrives in device order.
        grad_blocks = grad_l.reshape(n_blocks, block, n_params)
        block_rows = lambda j: _redistribute(
            jax.lax.dynamic_index_in_dim(grad_blocks, j, axis=0, keepdims=False)
        )

    def add_block(j, T):
        # conj(grad_l) @ grad_r.T is the complex conjugate of the kernel: conjugating the
        # reduced block, instead of grad_r, avoids a copy of grad_r
        local = jax.lax.dot_general(jnp.conj(block_rows(j)), grad_r, (((1,), (1,)), ((), ())))
        local = jnp.conj(jax.lax.psum_scatter(local, 'devices', scatter_dimension=0, tiled=True))

        return jax.lax.dynamic_update_slice_in_dim(T, local, j * block, axis=0)

    T = jnp.zeros((n_local_rows, grad_r.shape[0]), dtype)
    T = jax.lax.pcast(T, ('devices',), to='varying')

    return jax.lax.fori_loop(0, n_blocks, add_block, T)

class MinSR(AbstractOptimizer):
    """
    This class provides functionality for energy minimization via MinSR.

    See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``psi``: The variational wave function (``NQS`` instance) being optimized.
        * ``diagonalShift``: Regularization parameter :math:`\\lambda`, the diagonal shift added \
        to the tangent kernel before inversion, see above. May be a constant or a callable \
        ``step -> value``, updated at each step via ``update_hyperparams``.
        * ``solver``: Solver used to invert the tangent kernel, e.g. \
        :class:`jVMC_exp.solver.Pinv`. Its ``pinv_cutoff`` plays the role of the \
        regularization parameter :math:`\\epsilon_{SVD}`, and its \
        ``diagonalization_mode``/``T_A`` select the (optionally distributed) \
        eigensolver backend. Only dense solvers are supported: MinSR materializes the \
        tangent kernel, so a matrix-free solver such as :class:`jVMC_exp.solver.CG` is \
        rejected at construction. The signal-to-noise regularization of \
        :class:`jVMC_exp.solver.PinvSNR` is rejected as well, since it estimates the \
        sampling variance of a Monte Carlo *average*, whereas MinSR's right hand side \
        holds one local estimator per sample.
        * ``resample_stepper``: Whether the sampler resamples at every stepper substep.
    """
    def __init__(
            self, sampler: AbstractSampler, psi: NQS,
            diagonalShift=1e-3, solver: AbstractSolver=Pinv(),
            resample_stepper=True, output_manager: OutputManager | None = None
        ):
        self.diag_shift = diagonalShift

        if not solver._needs_dense_matrix:
            raise ValueError(
                f"Solver {solver.__class__.__name__} is not compatible with MinSR, "
                "which requires a dense tangent kernel."
            )
        if getattr(solver, "snr_tol", 0):
            raise ValueError(
                f"Solver {solver.__class__.__name__} has a non-zero SNR tolerance, "
                "which is not compatible with MinSR."
            )
        self._solver = solver

        self._concat = (not psi.holomorphic) and (jnp.issubdtype(psi.out_dtype, jnp.complexfloating))
        num_params = psi.numParameters * (2 if not psi.realParams else 1)
        self._params_pad_size = (- num_params) % MESH.shape["devices"]

        super().__init__(
            sampler, psi, resample_stepper, use_cross_valiadation=False, output_manager=output_manager
        )

        self._solver_state = dict(
            exact_sampler=isinstance(self.sampler, ExactSampler),
            pad_size=0
        )

    @property
    def solver_state(self):
        return self._solver_state

    @property
    def solver(self):
        return self._solver

    @property
    def diag_shift(self):
        return self._diag_shift

    @diag_shift.setter
    def diag_shift(self, value):
        self._diag_shift_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_shift = self._diag_shift_fn(0)

    @property
    def _needs_grad(self):
        return False

    def update_hyperparams(self, step):
        self._diag_shift = self._diag_shift_fn(step)

    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Dispatches on the type of ``objective_function_output.grad_log_psi``: a dense
        ``SampledObs`` is turned into a single tangent kernel via ``_get_tangent_kernel``, while a
        ``LazySampledObs`` (``batched_jacobian=True``) is processed batch pair by batch pair so
        that the full Jacobian is never materialized densely.
        """
        o_loc = objective_function_output.o_loc._normalized_obs.flatten()
        if self._params_pad_size != 0:
            objective_function_output.grad_log_psi.transform(
                lambda x: jnp.pad(x, ((0, 0), (0, self._params_pad_size)), mode="constant")
            )

        if isinstance(objective_function_output.grad_log_psi, SampledObs):
            grad = objective_function_output.grad_log_psi._normalized_obs
            if self._concat:
                grad = _concat_nonholo(grad)

            T = self._get_tangent_kernel(grad)

        else:
            # T is assembled with rows and columns in the original sample order (with the real
            # and imaginary rows of each sample next to each other if self._concat), like o_loc.
            # Each row block is written in place on the devices that hold its samples.
            grad = objective_function_output.grad_log_psi
            layout = grad.layout.scaled(2 if self._concat else 1)
            positions = jax.device_put(layout.batch_positions(), REPLICATED_SHARDING)
            T = None
            batches_l = iter(grad.observations)
            for l, weights_l in enumerate(grad._weights):
                batch_l = _normalize_batch(next(batches_l), weights_l, grad.mean, self._concat)

                T_row = []
                batches_r = iter(grad.observations)
                for weights_r in grad._weights:
                    batch_r = _normalize_batch(next(batches_r), weights_r, grad.mean, self._concat)
                    T_row.append(self._get_tangent_kernel(batch_l, batch_r))
                # The columns come in batch order: move them to the original sample order
                T_row = _take_columns(jnp.concatenate(T_row, axis=1), positions)

                if T is None:
                    T = layout.alloc(T_row)
                T = layout.put(T, T_row, l)

        if self._concat:
            o_loc = _concat_nonholo(o_loc)

        if self.diag_shift > 1e-15:
            idx = jnp.arange(T.shape[0])
            T = T.at[idx, idx].add(self.diag_shift)

        solution, self._additional_info = self.solver(T, o_loc, **self.solver_state)
        del T

        if isinstance(objective_function_output.grad_log_psi, SampledObs):
            update = - jnp.conj(jnp.transpose(grad)) @ solution
        else:
            update = 0
            batches = iter(grad.observations)
            for weights, solution_batch in zip(grad._weights, layout.split(solution)):
                grad_batch = _normalize_batch(next(batches), weights, grad.mean, self._concat)
                update -= jnp.conj(jnp.transpose(grad_batch)) @ solution_batch

        if self._params_pad_size != 0:
            objective_function_output.grad_log_psi.transform(
                lambda x: x[:, :-self._params_pad_size]
            )
            update = update[:-self._params_pad_size]
    
        return update

    def cross_validation(self, objective_function: AbstractObjectiveFunction, **objective_function_kwargs):
        raise NotImplementedError
    
    def _update_meta_data(self):
        self.meta_data = dict(**self._additional_info)

    def _get_tangent_kernel(self, grad_l, grad_r=None):
        """
        Dispatches to the single- or two-operand tangent kernel depending on whether a second
        (distinct) gradient block is given, avoiding a redundant ``all_to_all`` of the same data
        when computing a self-kernel (``grad_r is None`` or, for the batched path, the diagonal
        ``l == r`` blocks).
        """
        if grad_r is None:
            return self._get_single_tangent_kernel(grad_l, batch_size=None)
        return self._get_double_tangent_kernel(grad_l, grad_r, batch_size=None)

    @sharded(use_vmap=False)
    def _get_single_tangent_kernel(self, grad, *, batch_size):
        """
        Computes ``grad @ conj(grad).T``, sharded across devices along the sample axis, without
        ever materializing the full (samples x parameters) ``grad`` on any single device.

        ``grad`` arrives sharded along the sample axis with the (padded) parameter axis local to
        each device. Computing the kernel directly in that layout would need cross-device pairs
        of samples, forcing an implicit all-gather of the whole matrix under automatic sharding.
        Instead, ``all_to_all`` trades which axis is sharded (each device ends up with all
        samples but only its own shard of parameters), so the local matmul is a valid partial
        sum over that parameter shard; ``psum_scatter`` then reduces and re-shards these partial
        sums into the exact, correctly-sharded kernel, one block of rows at a time so that the
        full kernel is never materialized on any device (see `_blockwise_tangent_kernel`).
        """
        return _blockwise_tangent_kernel(grad)

    @sharded(use_vmap=False)
    def _get_double_tangent_kernel(self, grad_l, grad_r, *, batch_size):
        """
        Same as ``_get_single_tangent_kernel``, but for the cross term ``grad_l @ conj(grad_r).T``
        between two distinct gradient blocks (used for off-diagonal batches in the
        ``LazySampledObs`` path). Both operands must be redistributed the same way so that
        matching parameter shards land on the same device for both.
        """
        return _blockwise_tangent_kernel(grad_l, grad_r)