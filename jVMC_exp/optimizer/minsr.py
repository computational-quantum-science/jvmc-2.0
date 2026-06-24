import jax.numpy as jnp
from typing import Callable

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput

class MinSR(AbstractOptimizer):
    """
    This class provides functionality for energy minimization via MinSR.

    See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``pinvTol``: Regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``diagonalSchift``: Regularization parameter :math:`\\lambda`, see below.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """
    def __init__(self, sampler, psi, pinv_tol=1e-14, diagonalShift=1e-3):
        self.pinvTol = pinv_tol
        self.diag_shift = diagonalShift
        self.diagonalShift = self._diag_shift_fn(0)

        super().__init__(sampler, psi, use_cross_valiadation=False)

    @property
    def diag_shift(self):
        return self._diag_shift
    
    @diag_shift.setter
    def diag_shift(self, value):
        self._diag_shift_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_shift = self._diag_shift_fn(0)

    def update_hyperparams(self, step):
        self.diagonalShift = self._diag_shift_fn(step)

    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Uses the techique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """
        o_loc = objective_function_output.o_loc
        grad_log_psi = objective_function_output.grad_log_psi

        if self.psi.holomorphic:
            T = grad_log_psi.tangent_kernel # TODO: No regularization??
            T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)
            return - grad_log_psi._normalized_obs.conj().T @ T_inv @ o_loc._normalized_obs.squeeze()

        gradients_all = jnp.concatenate([jnp.real(grad_log_psi._normalized_obs), jnp.imag(grad_log_psi._normalized_obs)])
        o_loc_all = jnp.concatenate([jnp.real(o_loc._normalized_obs), jnp.imag(o_loc._normalized_obs)]).squeeze()

        T = gradients_all @ gradients_all.T
        T = T + self.diagonalShift * jnp.eye(T.shape[-1])
        T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)

        return - gradients_all.T @ T_inv @ o_loc_all
    
    def cross_validation(self):
        raise NotImplementedError
    
    def _update_meta_data(self):
        pass


###### TODO: still giving problems on bigger systems

# import jax
# import jax.numpy as jnp
# from typing import Callable

# from jVMC_exp.sampler import AbstractSampler
# from jVMC_exp.vqs import NQS
# from jVMC_exp.optimizer.base import AbstractOptimizer
# from jVMC_exp.objective_function.base import ObjectiveFunctionOutput
# from jVMC_exp.sharding_config import DEVICE_SPEC, REPLICATED_SPEC, MESH, sharded

# @jax.jit
# def _concat_nonholo(arr):
#     return jnp.concatenate([jnp.real(arr), jnp.imag(arr)], axis=0)

# class MinSR(AbstractOptimizer):
#     """
#     This class provides functionality for energy minimization via MinSR.

#     See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

#     Initializer arguments:
#         * ``sampler``: A sampler object.
#         * ``pinv_tol``: Regularization parameter :math:`\\epsilon_{SVD}`, see above.
#         * ``diagonalSchift``: Regularization parameter :math:`\\lambda`, see below.
#         * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
#     """
#     def __init__(self, sampler: AbstractSampler, psi: NQS, pinv_tol=1e-14, diagonalShift=1e-3):
#         self.pinv_tol = pinv_tol
#         self.diag_shift = diagonalShift

#         num_params = psi.numParameters * (2 if not psi.realParams else 1)
#         num_devices = MESH.shape["devices"]
#         self._params_pad_size = int((num_devices - num_params % num_devices) % num_devices)

#         super().__init__(sampler, psi, use_cross_valiadation=False)

#     @property
#     def diag_shift(self):
#         return self._diag_shift
    
#     @diag_shift.setter
#     def diag_shift(self, value):
#         self._diag_shift_fn = value if isinstance(value, Callable) else lambda step: value
#         self._diag_shift = self._diag_shift_fn(0)

#     def update_hyperparams(self, step):
#         self._diag_shift = self._diag_shift_fn(step)

#     def get_update(self, objective_function_output: ObjectiveFunctionOutput):
#         """
#         Uses the techique proposed in arXiv:2302.01941 to compute the updates.
#         Efficient only if number of samples :math:`\\ll` number of parameters.
#         """
#         # FIX this with normalized obs not centered (!!!)
#         grad_log_psi = objective_function_output.grad_log_psi._normalized_obs  # (Ns, Np)
#         o_loc = objective_function_output.o_loc._normalized_obs.reshape(-1)    # (Ns,)
        
#         if not self.psi.holomorphic and not self.psi.realParams:
#             grad_log_psi = _concat_nonholo(grad_log_psi)
#             o_loc = _concat_nonholo(o_loc)

#         update = self._solve(
#             grad_log_psi, 
#             o_loc, 
#             diag_shift=self.diag_shift,
#             pinv_tol=self.pinv_tol, 
#             batch_size=None
#         ).flatten()

#         update = update[:-self._params_pad_size] if self._params_pad_size > 0 else update
       
#         return jnp.array(jax.experimental.multihost_utils.process_allgather(update, tiled=True))
    
#     @sharded(use_vmap=False, in_specs=(DEVICE_SPEC, REPLICATED_SPEC))
#     def _solve(self, gradients, o_loc, *, diag_shift, pinv_tol, batch_size):
#         gradients = jnp.concatenate([
#             gradients, 
#             jnp.zeros((gradients.shape[0], self._params_pad_size))], axis=1
#         )
#         gradients = jax.lax.all_to_all(gradients, 'devices', split_axis=1, concat_axis=0, tiled=True)
#         y = gradients @ jnp.conj(jnp.transpose(gradients))          # (Ns, Ns)
#         y = jax.lax.psum(y, 'devices')
#         y = y + diag_shift * jnp.eye(y.shape[-1])

#         y = jnp.linalg.pinv(y, rtol=pinv_tol, hermitian=True)
#         y = y @ o_loc                                               # (Ns,)
        
#         return -1 * jnp.conj(jnp.transpose(gradients)) @ y          # (Np,)

#     def cross_validation(self):
#         raise NotImplementedError
    
#     def _update_meta_data(self):
#         pass