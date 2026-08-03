import jax
import jax.numpy as jnp
from typing import Callable

from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.stats import SampledObs, LazySampledObs, _normalize, _reshape_in_batches
from jVMC_exp.vqs import NQS
from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput, AbstractObjectiveFunction
from jVMC_exp.sharding_config import DEVICE_SPEC, REPLICATED_SPEC, MESH, sharded

@jax.jit
def _concat_nonholo(arr):
    """
    Returns a real array correctly sharded on the first dimension
    """
    return jnp.concatenate([jnp.real(arr), jnp.imag(arr)], axis=0)

class MinSR(AbstractOptimizer):
    """
    This class provides functionality for energy minimization via MinSR.

    See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``pinv_tol``: Regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``diagonalSchift``: Regularization parameter :math:`\\lambda`, see below.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """
    def __init__(
            self, sampler: AbstractSampler, psi: NQS,
            pinv_tol=1e-14, diagonalShift=1e-3,
            resample_stepper=True,
        ):
        self.pinv_tol = pinv_tol
        self.diag_shift = diagonalShift

        num_params = psi.numParameters * (2 if not psi.realParams else 1)
        num_devices = MESH.shape["devices"]
        self._params_pad_size = int((num_devices - num_params % num_devices) % num_devices)
        self._concat = (not psi.holomorphic) and (not psi.realParams)

        super().__init__(sampler, psi, resample_stepper, use_cross_valiadation=False)

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
        Uses the technique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """
        grad_log_psi = objective_function_output.grad_log_psi
        o_loc = objective_function_output.o_loc._normalized_obs.reshape(-1)

        if isinstance(grad_log_psi, LazySampledObs):
            return self._solve_lazy(grad_log_psi, o_loc)
        
        gradients = grad_log_psi._get_normalized_obs_and_consume()
        if self._concat:
            gradients = _concat_nonholo(gradients)
            o_loc = _concat_nonholo(o_loc)
        update = self._solve(
            gradients, o_loc,
            diag_shift=self.diag_shift, pinv_tol=self.pinv_tol, batch_size=None
        ).flatten()

        update = update[:-self._params_pad_size] if self._params_pad_size > 0 else update

        return jnp.array(jax.experimental.multihost_utils.process_allgather(update, tiled=True))

    def cross_validation(self, objective_function: AbstractObjectiveFunction, **objective_function_kwargs):
        raise NotImplementedError

    def _update_meta_data(self):
        pass

    @sharded(use_vmap=False, in_specs=(DEVICE_SPEC, REPLICATED_SPEC))
    def _solve(self, gradients, o_loc, *, diag_shift, pinv_tol, batch_size):
        gradients = jnp.concatenate([
            gradients,
            jnp.zeros((gradients.shape[0], self._params_pad_size), dtype=gradients.dtype)], axis=1
        )
        gradients = jax.lax.all_to_all(gradients, 'devices', split_axis=1, concat_axis=0, tiled=True)
        y = gradients @ jnp.conj(jnp.transpose(gradients))          # (Ns, Ns)
        y = jax.lax.psum(y, 'devices')
        y = y + diag_shift * jnp.eye(y.shape[-1], dtype=y.dtype)

        y = jnp.linalg.pinv(y, rtol=pinv_tol, hermitian=True)
        y = y @ o_loc                                               # (Ns,)

        return -1 * jnp.conj(jnp.transpose(gradients)) @ y          # (Np,)

    def _solve_lazy(self, grad: LazySampledObs, o_loc):
        def normalize_batch(batch, weights):
                batch = _normalize(batch, weights, grad.mean)
                if self._concat:
                    batch = _concat_nonholo(batch)
        
                return batch
        
        y = []
        for batch_l, weights_l in zip(grad.observations, grad._weights):
            batch_l = normalize_batch(batch_l, weights_l)

            y_batch = []
            for batch_r, weights_r in zip(grad.observations, grad._weights):
                batch_r = normalize_batch(batch_r, weights_r)
                y_batch.append(batch_l @ jnp.conj(jnp.transpose(batch_r)))

            y.append(jnp.concatenate(y_batch, axis=1))
        y = jnp.concatenate(y)

        if self._concat:
            pieces = []
            start = 0
            for weights in grad._weights:
                size = weights.shape[0]
                pieces.append(_concat_nonholo(o_loc[start:start + size]))
                start += size

            o_loc = jnp.concatenate(pieces)

        y = y + self.diag_shift * jnp.eye(y.shape[-1], dtype=y.dtype)
        y = jnp.linalg.pinv(y, rtol=self.pinv_tol, hermitian=True)
        y = y @ o_loc

        y = _reshape_in_batches(
            y, 
            grad._batch_size if self.psi.holomorphic else 2 * grad._batch_size
        )

        update = 0
        for grad_batch, weights, y_batch in zip(grad.observations, grad._weights, y):
            grad_batch = normalize_batch(grad_batch, weights)
            update += jnp.conj(jnp.transpose(grad_batch)) @ y_batch

        return -update