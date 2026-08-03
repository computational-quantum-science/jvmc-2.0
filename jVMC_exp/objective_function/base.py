from abc import ABC, abstractmethod
from chex import dataclass
import jax

from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler import AbstractSampler
from jVMC_exp.sharding_config import sharded
from jVMC_exp.util.grads import pick_gradient

@dataclass
class ObjectiveFunctionOutput():
    o_loc: SampledObs | None = None
    grad_log_psi: SampledObs | LazySampledObs | None = None
    grad: jax.Array | None = None
    grad_var_re: jax.Array | None = None
    grad_var_im: jax.Array | None = None
    grad_cov_re_im: jax.Array | None = None

class AbstractObjectiveFunction(ABC):
    @abstractmethod
    def __call__(self, sampler: AbstractSampler, **kwargs) -> SampledObs:
        pass

    @abstractmethod
    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool, **kwargs) -> ObjectiveFunctionOutput:
        pass

class Observable(AbstractObjectiveFunction):
    def __init__(self, operator: AbstractOperator, batched_jacobian: bool = False):
        self._operator = operator
        self._batched_jacobian = batched_jacobian
        # TODO: might add a batch size here so that one can have different batch sizes

    @property
    def operator(self):
        return self._operator
    
    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return sampler(self.operator, **op_kwargs)
    
    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        if self._batched_jacobian:
            grad_log_psi = LazySampledObs(sampler.psi.lazy_gradients(sampler.samples), sampler.weights)
        else:
            grad_log_psi = SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

        if compute_grad:
            grad, grad_var_re, grad_var_im, grad_cov_re_im = grad_log_psi.get_covar_and_covar_var(o_loc)
            return ObjectiveFunctionOutput(
                o_loc=o_loc, grad=grad,
                grad_var_re=grad_var_re, grad_var_im=grad_var_im, grad_cov_re_im=grad_cov_re_im,
                grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

class Estimator(AbstractObjectiveFunction):
    def __init__(self, estimator_fn: callable, batched_jacobian: bool = False):
        self._estimator_fn = estimator_fn
        self._is_grad_init = False
        self._batched_jacobian = batched_jacobian
        
    @property
    def estimator_fn(self):
        return self._estimator_fn
    
    def __call__(self, sampler: AbstractSampler):
        observations = self.estimator_fn(sampler.psi.parameters, sampler.samples)

        return SampledObs(observations, sampler.weights)
    
    def value_and_grad(self, sampler: AbstractSampler, **kwargs):
        if not self._is_grad_init:
            _, _, self._grad_fn, _ = pick_gradient(self.estimator_fn, sampler.psi.parameters, sampler.samples[0])
            self._is_grad_init = True

        value = self(sampler)
        if self._batched_jacobian:
            grad_obs = LazySampledObs(
                self._lazy_grad_fn_sh(
                    sampler.samples, 
                    parameters=sampler.psi.parameters, 
                    batch_size=sampler.psi.batchSize
                ), 
                sampler.weights
            )
        else:
            grad_obs = SampledObs(
                self._grad_fn_sh(
                    sampler.samples, 
                    parameters=sampler.psi.parameters, 
                    batch_size=sampler.psi.batchSize
                ), 
                sampler.weights
            )

        return ObjectiveFunctionOutput(o_loc=value, grad=grad_obs.mean)

    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)
    
    @sharded(automatic_sharding=True, yield_iter=True) # TODO: Set flag to False once jax problem is solved
    def _lazy_grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)
    
class ParametricObservable(AbstractObjectiveFunction):
    """
    Objective function for an observable O(θ, s) that depends explicitly on
    both the network parameters θ and the configuration s.
    """
    def __init__(
            self, 
            operator: AbstractOperator, 
            estimator_fn: callable, 
            batched_jacobian: bool = False
        ):
        self._observable = Observable(operator, batched_jacobian)
        self._estimator = Estimator(estimator_fn, batched_jacobian)

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return self._observable(sampler, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        obs_out = self._observable.value_and_grad(sampler, compute_grad, **op_kwargs)
        o_loc = obs_out.o_loc
        grad_log_psi = obs_out.grad_log_psi

        if compute_grad:
            grad = obs_out.grad + self._estimator.value_and_grad(sampler, compute_grad=compute_grad).grad

            return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi)
        
        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)