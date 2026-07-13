import jax
import jax.numpy as jnp
from typing import Callable

from jVMC_exp.sampler import AbstractSampler
from jVMC_exp.vqs import NQS
from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import AbstractObjectiveFunction, ObjectiveFunctionOutput
from jVMC_exp.sharding_config import DEVICE_SPEC, REPLICATED_SPEC, REPLICATED_SHARDING, MESH, sharded
from jVMC_exp.stats import BatchedJacobian
from jVMC_exp.util.output_manager import OutputManager

@jax.jit
def _concat_nonholo(arr):
    return jnp.concatenate([jnp.real(arr), jnp.imag(arr)], axis=0)

@jax.jit
def _solve_tangent(tangent, o_loc, diag_shift, pinv_tol):
    tangent = tangent + diag_shift * jnp.eye(tangent.shape[-1], dtype=tangent.dtype)
    tangent = jnp.linalg.pinv(tangent, rtol=pinv_tol, hermitian=True)
    return tangent @ o_loc

class MinSR(AbstractOptimizer):
    """
    This class provides functionality for energy minimization via MinSR.

    See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``pinv_tol``: Regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``diagonalShift``: Regularization parameter :math:`\\lambda`, see below.
        * ``full_batched``: Recompute Jacobian blocks on demand instead of materializing the full Jacobian.
    """
    def __init__(
            self,
            sampler: AbstractSampler,
            psi: NQS,
            pinv_tol=1e-14,
            diagonalShift=1e-3,
            output_manager: OutputManager | None = None,
            full_batched: bool = False,
        ):
        self.pinv_tol = pinv_tol
        self.diag_shift = diagonalShift
        self.full_batched = bool(full_batched)

        num_params = psi.numParameters * (2 if not psi.realParams else 1)
        num_devices = MESH.shape["devices"]
        self._params_pad_size = int((num_devices - num_params % num_devices) % num_devices)

        super().__init__(
            sampler,
            psi,
            use_cross_valiadation=False,
            output_manager=output_manager,
        )

    @property
    def diag_shift(self):
        return self._diag_shift
    
    @diag_shift.setter
    def diag_shift(self, value):
        self._diag_shift_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_shift = self._diag_shift_fn(0)

    def update_hyperparams(self, step):
        self._diag_shift = self._diag_shift_fn(step)

    def _objective_function_kwargs(self, objective_function: AbstractObjectiveFunction):
        return dict(
            jacobian_mode="batched" if self.full_batched else "dense",
            compute_grad=False,
        )

    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Uses the technique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """
        if isinstance(objective_function_output.grad_log_psi, BatchedJacobian):
            return self._get_update_batched(objective_function_output)

        grad_log_psi = objective_function_output.grad_log_psi._normalized_obs  # (Ns, Np)
        o_loc = objective_function_output.o_loc._normalized_obs.reshape(-1)    # (Ns,)
        
        if not self.psi.holomorphic and not self.psi.realParams:
            grad_log_psi = _concat_nonholo(grad_log_psi)
            o_loc = _concat_nonholo(o_loc)

        update = self._solve(
            grad_log_psi, 
            o_loc, 
            diag_shift=self.diag_shift,
            pinv_tol=self.pinv_tol, 
            batch_size=None
        ).flatten()

        update = update[:-self._params_pad_size] if self._params_pad_size > 0 else update
       
        return jnp.array(jax.experimental.multihost_utils.process_allgather(update, tiled=True))

    def _get_update_batched(self, objective_function_output: ObjectiveFunctionOutput):
        grad_log_psi = objective_function_output.grad_log_psi
        if not self.psi.holomorphic and not self.psi.realParams:
            grad_log_psi = grad_log_psi.real_imag_doubled()
        tangent = grad_log_psi.tangent_kernel()
        o_loc = grad_log_psi.normalized_observable(objective_function_output.o_loc)
        y = _solve_tangent(
            jax.device_put(tangent, REPLICATED_SHARDING),
            o_loc,
            diag_shift=self.diag_shift,
            pinv_tol=self.pinv_tol,
        )
        y = jax.block_until_ready(y)
        update = -grad_log_psi.conj_transpose_matvec(y)

        return jnp.array(jax.experimental.multihost_utils.process_allgather(update, tiled=True))
    
    @sharded(use_vmap=False, in_specs=(DEVICE_SPEC, REPLICATED_SPEC))
    def _solve(self, gradients, o_loc, *, diag_shift, pinv_tol, batch_size):
        gradients = jnp.concatenate([
            gradients, 
            jnp.zeros((gradients.shape[0], self._params_pad_size))], axis=1
        )
        gradients = jax.lax.all_to_all(gradients, 'devices', split_axis=1, concat_axis=0, tiled=True)
        y = gradients @ jnp.conj(jnp.transpose(gradients))          # (Ns, Ns)
        y = jax.lax.psum(y, 'devices')
        y = y + diag_shift * jnp.eye(y.shape[-1])

        y = jnp.linalg.pinv(y, rtol=pinv_tol, hermitian=True)
        y = y @ o_loc                                               # (Ns,)
        
        return -1 * jnp.conj(jnp.transpose(gradients)) @ y          # (Np,)

    def cross_validation(self):
        raise NotImplementedError
    
    def _update_meta_data(self):
        pass
