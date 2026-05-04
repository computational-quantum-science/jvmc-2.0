import jax.numpy as jnp

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
    def __init__(self, sampler, psi, pinvTol=1e-14, diagonalShift=1e-3):
        self.pinvTol = pinvTol
        self.diagonalShift = diagonalShift

        super().__init__(sampler, psi, use_cross_valiadation=False)

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
