from abc import ABC, abstractmethod
from chex import dataclass

from jVMC_exp.stats import SampledObs
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler import AbstractSampler
from jVMC_exp.sharding_config import sharded
from jVMC_exp.util.grads import pick_gradient

@dataclass
class ObjectiveFunctionOutput():
    o_loc: SampledObs | None = None
    grad: SampledObs | None = None
    grad_log_psi: SampledObs | None = None

    def get_subset(self, start=None, end=None, step=None):
        return ObjectiveFunctionOutput(
            o_loc=self.o_loc.get_subset(start=start, end=end, step=step) if self.o_loc is not None else None,
            grad=self.grad.get_subset(start=start, end=end, step=step) if self.grad is not None else None,
            grad_log_psi=self.grad_log_psi.get_subset(start=start, end=end, step=step ) if self.grad_log_psi is not None else None
        )
    
    def transform(self, element_wise_fn=lambda x: x, linear_map=None):
        return ObjectiveFunctionOutput(
            o_loc=self.o_loc.transform(element_wise_fn, linear_map) if self.o_loc is not None else None,
            grad=self.grad.transform(element_wise_fn, linear_map) if self.grad is not None else None,
            grad_log_psi=self.grad_log_psi.transform(element_wise_fn, linear_map) if self.grad_log_psi is not None else None
        )

class AbstractObjectiveFunction(ABC):
    @abstractmethod
    def __call__(self, sampler: AbstractSampler, **kwargs) -> SampledObs:
        pass

    @abstractmethod
    def value_and_grad(self, sampler: AbstractSampler, **kwargs) -> ObjectiveFunctionOutput:
        pass

class Observable(AbstractObjectiveFunction):
    def __init__(self, operator: AbstractOperator):
        self._operator = operator

    @property
    def operator(self):
        return self._operator
    
    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return sampler(self.operator, **op_kwargs)
    
    def value_and_grad(self, sampler: AbstractSampler, compute_grad_covar: bool = True, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        grad_log_psi = SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

        if compute_grad_covar:
            grad_obs = grad_log_psi.get_covar_obs(o_loc)
            return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad_obs, grad_log_psi=grad_log_psi)
        else:
            return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

class Estimator(AbstractObjectiveFunction):
    def __init__(self, estimator_fn: callable):
        self._estimator_fn = estimator_fn
        self._is_grad_init = False
        
    @property
    def estimator_fn(self):
        return self._estimator_fn
    
    def __call__(self, sampler: AbstractSampler):
        observations = self.estimator_fn(sampler.psi.parameters, sampler.samples)

        return SampledObs(observations, sampler.weights)
    
    def value_and_grad(self, sampler: AbstractSampler):
        if not self._is_grad_init:
            _, _, self._grad_fn, _ = pick_gradient(self.estimator_fn, sampler.psi.parameters, sampler.samples[0])
            self._is_grad_init = True

        value = self(sampler)
        grad = SampledObs(self._get_estimator_grad(
            sampler.samples, 
            parameters=sampler.psi.parameters, 
            batch_size=sampler.psi.batchSize
        ), sampler.weights)

        return ObjectiveFunctionOutput(o_loc=value, grad=grad)

    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _get_estimator_grad(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)
    
class ParametricObservable(AbstractObjectiveFunction):
    """
    Objective function for an observable O(θ, s) that depends explicitly on
    both the network parameters θ and the configuration s.
    """
    def __init__(self, operator: AbstractOperator, estimator_fn: callable):
        self._observable = Observable(operator)
        self._estimator = Estimator(estimator_fn)

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return self._observable(sampler, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        grad_log_psi = SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

        term1 = grad_log_psi.get_covar(o_loc).ravel()
        estimator_out = self._estimator.value_and_grad(sampler)
        grad = SampledObs(term1 + estimator_out.grad.observations, sampler.weights)

        return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi)