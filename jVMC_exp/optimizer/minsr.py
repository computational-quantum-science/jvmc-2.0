import jax.numpy as jnp

from jVMC_exp.stats import SampledObs
from jVMC_exp.optimizer.base import AbstractOptimizer

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
    def __init__(
            self, sampler, psi, stepper, output_manager=None,
            pinvTol=1e-14, diagonalShift=1e-3, diagonalizeOnDevice=True
        ):
        self.pinvTol = pinvTol
        self.diagonalShift = diagonalShift
        self.diagonalizeOnDevice = diagonalizeOnDevice

        super().__init__(sampler, psi, stepper, output_manager, use_cross_valiadation=False)

    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        """
        Uses the techique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """
        if self.psi.holomorphic:
            T = gradients.tangent_kernel
            T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)
            return - gradients._normalized_obs.conj().T @ T_inv @ Eloc._normalized_obs.squeeze()

        gradients_all = jnp.concatenate([jnp.real(gradients._normalized_obs), jnp.imag(gradients._normalized_obs)])
        Eloc_all = jnp.concatenate([jnp.real(Eloc._normalized_obs), jnp.imag(Eloc._normalized_obs)]).squeeze()

        T = gradients_all @ gradients_all.T
        T = T + self.diagonalShift * jnp.eye(T.shape[-1])
        T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)

        return - gradients_all.T @ T_inv @ Eloc_all
    
    def cross_validation(self):
        raise NotImplementedError
    
    def _update_meta_data(self):
        pass
