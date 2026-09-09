import jax
import jax.numpy as jnp
from typing import Callable

from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.sampler import ExactSampler
from jVMC_exp.stats import SampledObs, _reshape_in_batches
from jVMC_exp.vqs import NQS
from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput, AbstractObjectiveFunction
from jVMC_exp.sharding_config import sharded, MESH
from jVMC_exp.util import OutputManager
from jVMC_exp.solver.base import AbstractSolver
from jVMC_exp.solver import Pinv

@jax.jit
def _concat_nonholo(arr):
    """
    Returns a real array correctly sharded on the first dimension
    """
    return jnp.concatenate([jnp.real(arr), jnp.imag(arr)], axis=0)

@jax.jit(static_argnums=(3,))
def _normalize_batch(batch, weights, mean, concat):
    batch = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), batch - mean)
    if concat:
        batch = jnp.concatenate([jnp.real(batch), jnp.imag(batch)], axis=0)

    return batch

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

        self._concat = (not psi.holomorphic) and (not psi.realParams)
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
                o_loc = _concat_nonholo(o_loc)

            T = self._get_tangent_kernel(grad)

        else:
            grad = objective_function_output.grad_log_psi
            T = []
            o_loc_batch = []
            start = 0
            for batch_l, weights_l in zip(grad.observations, grad._weights):
                batch_l = _normalize_batch(batch_l, weights_l, grad.mean, self._concat)
    
                if self._concat:
                    size = weights_l.shape[0]
                    o_loc_batch.append(_concat_nonholo(o_loc[start:start + size]))
                    start += size
    
                T_batch = []
                for batch_r, weights_r in zip(grad.observations, grad._weights):
                    batch_r = _normalize_batch(batch_r, weights_r, grad.mean, self._concat)
                    T_batch.append(self._get_tangent_kernel(batch_l, batch_r))

                T.append(jnp.concatenate(T_batch, axis=1))
    
            T = jnp.concatenate(T)
            o_loc = jnp.concatenate(o_loc_batch) if self._concat else o_loc

        if self.diag_shift > 1e-15:
            idx = jnp.arange(T.shape[0])
            T = T.at[idx, idx].add(self.diag_shift)

        T, self._additional_info = self.solver(T, o_loc, **self.solver_state)

        if isinstance(objective_function_output.grad_log_psi, SampledObs):
            update = - jnp.conj(jnp.transpose(grad)) @ T
        else:
            T = _reshape_in_batches(
                T,
                grad._batch_size if self.psi.holomorphic else 2 * grad._batch_size
            )
    
            update = 0
            for grad_batch, weights, T_batch in zip(grad.observations, grad._weights, T):
                grad_batch = _normalize_batch(grad_batch, weights, grad.mean, self._concat)
                update -= jnp.conj(jnp.transpose(grad_batch)) @ T_batch

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
        sums into the exact, correctly-sharded kernel.
        """
        grad = jax.lax.all_to_all(grad, 'devices', split_axis=1, concat_axis=0, tiled=True)
        local = grad @ jnp.conj(jnp.transpose(grad))

        return jax.lax.psum_scatter(local, 'devices', tiled=True)

    @sharded(use_vmap=False)
    def _get_double_tangent_kernel(self, grad_l, grad_r, *, batch_size):
        """
        Same as ``_get_single_tangent_kernel``, but for the cross term ``grad_l @ conj(grad_r).T``
        between two distinct gradient blocks (used for off-diagonal batches in the
        ``LazySampledObs`` path). Both operands must be redistributed the same way so that
        matching parameter shards land on the same device for both.
        """
        grad_l = jax.lax.all_to_all(grad_l, 'devices', split_axis=1, concat_axis=0, tiled=True)
        grad_r = jax.lax.all_to_all(grad_r, 'devices', split_axis=1, concat_axis=0, tiled=True)
        local = grad_l @ jnp.conj(jnp.transpose(grad_r))

        return jax.lax.psum_scatter(local, 'devices', tiled=True)