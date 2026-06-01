import jax
import jax.numpy as jnp
from typing import Callable
from functools import partial

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput
from jVMC_exp.sharding_config import DEVICE_SPEC, MESH

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
    def __init__(self, sampler, psi, pinvTol=1e-14, diagonalShift=1e-3, centered=True):
        self.pinvTol = pinvTol
        self._diag_shift_fn = diagonalShift if isinstance(diagonalShift, Callable) else lambda step: diagonalShift
        self.diagonalShift = self._diag_shift_fn(0)
        self.holomorphic = psi.holomorphic
        self.real = psi.realParams
        self.centered = centered

        super().__init__(sampler, psi, use_cross_valiadation=False)

    def update_hyperparams(self, step):
        self.diagonalShift = self._diag_shift_fn(step)

    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Uses the techique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """

        @partial(jax.jit, static_argnames=["padding"])
        @partial(jax.shard_map, mesh=MESH, in_specs=(DEVICE_SPEC, None, None, None, None), out_specs=DEVICE_SPEC)
        def _solve(gradients, eloc, diagonalShift, pinvTol, padding):
            gr = jnp.concatenate([gradients, jnp.zeros((gradients.shape[0], padding))], axis=1)
            gr = jax.lax.all_to_all(gr, 'devices', split_axis=1, concat_axis=0, tiled=True)
            y = gr @ jnp.conj(jnp.transpose(gr)) # (Ns,Ns)
            y = jax.lax.psum(y, 'devices')
            y = y + diagonalShift * jnp.eye(y.shape[-1])
            y = jnp.linalg.pinv(y, rtol=pinvTol, hermitian=True)
            y = y @ eloc # (Ns,)
            y = -1 * jnp.conj(jnp.transpose(gr)) @ y # (Np,)
            return y
        
        gradients_sampobs = objective_function_output.grad_log_psi
        eloc_sampobs = objective_function_output.o_loc
        
        if self.centered:
            gradients_data = gradients_sampobs._centered_obs # (Ns,Np)
            eloc_data = eloc_sampobs._centered_obs.reshape(-1) # (Ns,)
        else:
            gradients_data = gradients_sampobs.observations # (Ns,Np)
            eloc_data = eloc_sampobs.observations.reshape(-1) # (Ns,)

        # padding for all_to_all: the number of parameters needs to be divisible by the number of devices
        a = self.psi.numParameters * (2 if not self.real else 1)
        b = jax.device_count()
        self.missingSize = int((b - a % b) % b)

        if not self.holomorphic and not self.real:
            gradients_data = jnp.concatenate([jnp.real(gradients_data), jnp.imag(gradients_data)], axis=0)
            eloc_data = jnp.concatenate([jnp.real(eloc_data), jnp.imag(eloc_data)], axis=0)
        
        update = _solve(gradients_data, eloc_data, self.diagonalShift, self.pinvTol, self.missingSize)
        update = (update[:-self.missingSize] if self.missingSize > 0 else update)
        update = jnp.array(jax.experimental.multihost_utils.process_allgather(update, tiled=True)).reshape(-1)
        return update
    
    def cross_validation(self):
        raise NotImplementedError
    
    def _update_meta_data(self):
        pass
