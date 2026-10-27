from abc import ABC, abstractmethod
from chex import dataclass
import jax

from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.sharding_config import sharded
from jVMC_exp.util.grads import pick_gradient

@dataclass
class ObjectiveFunctionOutput():
    """
    Output of ``AbstractObjectiveFunction.value_and_grad``.

    o_loc:        local losses ell_theta(s_i) on the sampler's configurations.
    grad_log_psi: logarithmic derivatives Gamma_k(s_i) = d_k log psi_theta(s_i).
    grad:         complex force vector F, such that d_k L = 2 Re F_k.
    """
    o_loc: SampledObs | None = None
    grad_log_psi: SampledObs | LazySampledObs | None = None
    grad: jax.Array | None = None

class AbstractObjectiveFunction(ABC):
    """
    Loss L(theta) = E_{s ~ p_theta}[ell_theta(s)], with real parameters theta and
    E[ell] real for all theta.

    ``value_and_grad`` returns the complex force vector

        F_k = Cov_p(Gamma_k^*, ell) + R_k / 2,
        R_k = E_p[d_k Re(ell)] - 2 E_p[Im(Gamma_k) Im(ell)],

    so that d_k L = 2 Re F_k. Subclasses differ only in the choice of ell and,
    therefore, in the form of R_k.
    """
    def __init__(self, batched_jacobian: bool = False):
        self._batched_jacobian = batched_jacobian

    @property
    def batched_jacobian(self) -> bool:
        return self._batched_jacobian

    @abstractmethod
    def __call__(self, sampler: AbstractSampler, **kwargs) -> SampledObs:
        pass

    @abstractmethod
    def value_and_grad(
        self, sampler: AbstractSampler, compute_grad: bool = True, **kwargs
    ) -> ObjectiveFunctionOutput:
        pass

    def _grad_log_psi(self, sampler: AbstractSampler) -> SampledObs | LazySampledObs:
        """Gamma_k(s_i), returned lazily if ``batched_jacobian`` is set."""
        if self._batched_jacobian:
            return LazySampledObs(
                sampler.psi.lazy_gradients(sampler.samples), sampler.weights
            )
        return SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

class Observable(AbstractObjectiveFunction):
    """
    L = <O> for a theta-independent Hermitian operator O, with ell = O_loc.
    Here R_k = 0, so F_k = Cov_p(Gamma_k^*, O_loc).
    """
    def __init__(self, operator: AbstractOperator, batched_jacobian: bool = False):
        super().__init__(batched_jacobian)
        self._operator = operator
        # TODO: might add a batch size here so that one can have different batch sizes

    @property
    def operator(self):
        return self._operator

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return sampler(self.operator, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        o_loc = self(sampler, **op_kwargs)
        grad_log_psi = self._grad_log_psi(sampler)

        if compute_grad:
            grad = grad_log_psi.get_covar(o_loc)
            return ObjectiveFunctionOutput(
                o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

class Estimator(AbstractObjectiveFunction):
    """
    ell_theta(s) = f(s, theta), an arbitrary real-valued function of the
    configuration and the parameters, given as ``estimator_fn(parameters, samples)``.

    Since Im(ell) = 0, R_k = E_p[d_k f], so

        F_k = Cov_p(Gamma_k^*, f) + E_p[d_k f] / 2,

    with d_k f obtained by automatic differentiation.
    """
    def __init__(self, estimator_fn: callable, batched_jacobian: bool = False):
        super().__init__(batched_jacobian)
        self._estimator_fn = estimator_fn
        self._grad_fn = None

    @property
    def estimator_fn(self):
        return self._estimator_fn

    def __call__(self, sampler: AbstractSampler, **kwargs):
        observations = self.estimator_fn(sampler.psi.parameters, sampler.samples)
        return SampledObs(observations, sampler.weights)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **kwargs):
        value = self(sampler)
        grad_log_psi = self._grad_log_psi(sampler)

        if compute_grad:
            grad = grad_log_psi.get_covar(value) + 0.5 * self.explicit_grad(sampler)
            return ObjectiveFunctionOutput(
                o_loc=value, grad=grad, grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=value, grad_log_psi=grad_log_psi)

    def explicit_grad(self, sampler: AbstractSampler) -> jax.Array:
        if self._grad_fn is None:
            _, _, self._grad_fn, _ = pick_gradient(
                self.estimator_fn, sampler.psi.parameters, sampler.samples[0]
            )

        grad_kwargs = dict(
            parameters=sampler.psi.parameters, batch_size=sampler.psi.batchSize
        )
        if self._batched_jacobian:
            grad_obs = LazySampledObs(
                self._lazy_grad_fn_sh(sampler.samples, **grad_kwargs), sampler.weights
            )
        else:
            grad_obs = SampledObs(
                self._grad_fn_sh(sampler.samples, **grad_kwargs), sampler.weights
            )

        return grad_obs.mean

    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)

    @sharded(automatic_sharding=True, yield_iter=True) # TODO: Set flag to False once jax problem is solved
    def _lazy_grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)

class ParametricObservable(AbstractObjectiveFunction):
    """
    L = <O(theta)> for an operator O(theta) that is Hermitian for all theta and
    whose matrix elements depend explicitly on the parameters, ell = O_loc(s; theta).

    The implicit dependence through psi_theta is treated as for ``Observable``,
    while the explicit one gives R_k = E_p[(d_k O)_loc], so

        F_k = Cov_p(Gamma_k^*, O_loc) + E_p[(d_k O)_loc] / 2.

    ``estimator_fn(parameters, samples)`` must encode only the explicit parameter
    dependence of the matrix elements: its derivative with respect to theta has to
    be (d_k O)_loc(s) = sum_s' d_k O_ss'(theta) psi(s')/psi(s), i.e. the wave-function
    ratios must be treated as constants (e.g. via ``jax.lax.stop_gradient``).
    """
    def __init__(
            self,
            operator: AbstractOperator,
            estimator_fn: callable,
            batched_jacobian: bool = False
        ):
        super().__init__(batched_jacobian)
        self._observable = Observable(operator, batched_jacobian)
        self._explicit = Estimator(estimator_fn, batched_jacobian)

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        return self._observable(sampler, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        obs_out = self._observable.value_and_grad(sampler, compute_grad, **op_kwargs)

        if compute_grad:
            grad = obs_out.grad + 0.5 * self._explicit.explicit_grad(sampler)
            return ObjectiveFunctionOutput(
                o_loc=obs_out.o_loc, grad=grad, grad_log_psi=obs_out.grad_log_psi
            )

        return obs_out