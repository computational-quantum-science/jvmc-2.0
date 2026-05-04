import jax
import jax.numpy as jnp
import warnings

from jVMC_exp.sampler import AbstractSampler
from jVMC_exp.operator.discrete.base import AbstractOperator
from jVMC_exp.operator.discrete.branch_free import IdentityOperator
from jVMC_exp.stats import SampledObs
from .base import AbstractObjectiveFunction, ObjectiveFunctionOutput

class MovingAverage:
    """Simple moving average used for the adaptive control-variate coefficient."""

    def __init__(self, width=10, init=0.0):
        self.values = jnp.full((width,), init)

    @jax.jit(static_argnums=(0,))
    def update(self, x):
        self.values = jnp.roll(self.values, -1)
        self.values = self.values.at[-1].set(x)

        return jnp.mean(self.values)

def _weighted_mean(values, weights):
    """Return the weighted sample mean as a scalar when possible."""
    mean = SampledObs(values, weights).mean
    return jnp.squeeze(mean)

@jax.jit
def _get_adapt_control_variate(x):
    return x, jnp.abs(x)**2, jnp.abs(x)**4, x * jnp.abs(x)**2

@jax.jit
def _get_control_variate(x):
    return x, jnp.abs(x)**2, None, None

class Infidelity(AbstractObjectiveFunction):
    def __init__(
        self,
        reference_sampler: AbstractSampler,
        operator: AbstractOperator | None=None,
        conjugated_operator: AbstractOperator | None=None,
        control_variate: str | None = None,
        moving_average_width: int=1,
        l_dim: int=2
    ):
        self.control_variate = control_variate
        self.moving_average_width = moving_average_width
        self._reference_sampler = reference_sampler
        self._control_variate_coeff = -0.5

        if conjugated_operator is None and operator is not None:
            warnings.warn(
                "No ConjugatedOperator provided; assuming Operator is Hermitian.",
                stacklevel=2,
            )
            if self.control_variate is not None:
                warnings.warn(
                    "Control variates are disabled when the operator is assumed Hermitian.",
                    stacklevel=2,
                )
                self.control_variate = None
        self._operator = operator or IdentityOperator(l_dim, 0)
        self._conjugated_operator = conjugated_operator or self.operator

        self._reset_cached_observables()

    @property
    def reference_sampler(self):
        return self._reference_sampler
    
    @property
    def operator(self):
        return self._operator

    @property
    def conjugated_operator(self):
        return self._conjugated_operator
    
    @property
    def control_variate(self):
        return self._control_variate
    
    @control_variate.setter
    def control_variate(self, value: str | None):
        if value is not None: 
            value = value.lower()
            if value not in ['simple', 'adaptive']:
                raise ValueError(f'Control variate can be either None, "Simple" or "Adaptive", got {value}')
            
            self._control_variate_fn = _get_control_variate if value == 'simple' else _get_adapt_control_variate
        
        self._control_variate = value

    @property
    def moving_average_width(self):
        return self._moving_average_width
    
    @moving_average_width.setter
    def moving_average_width(self, value):
        self._moving_average = MovingAverage(value)
        self._moving_average_width = value

    def __call__(
            self, sampler: AbstractSampler, 
            sample_ref_state: bool=True, 
            *, 
            control_variate_coeff=None, 
            **kwargs
        ) -> SampledObs:
        r"""
        Compute the local infidelity estimator on ``samples``.

        Args:
            samples: Sampled computational-basis configurations.
            psi: Trial variational state.
            logPsiS: Optional logarithmic amplitudes :math:`\log \psi(s)`.
                They are computed from ``psi`` when omitted.
            psi_p: Born probabilities of ``samples``. Required when
                ``adaptCV=True``.
            sample_chi: Whether to resample the reference state before updating
                the cached :math:`F^\chi` estimator.
            **kwargs: Optional estimator parameters. Currently only ``CVc`` is
                used to override the adaptive control-variate coefficient.

        Returns:
            The local infidelity estimator
            :math:`1 - F^\psi_{\rm loc}(s) \langle F^\chi_{\rm loc} \rangle`.
        """
        if sample_ref_state:
            self.reference_sampler.sample()

        o_loc = self.conjugated_operator.get_O_loc(
            self.reference_sampler.samples, sampler.net, logPsiS=self.reference_sampler.logPsi
        )

        if self.control_variate is not None:
            ref_f_loc, ref_f_loc_cv, self._ref_f2_loc_cv, self._ref_f_loc_f2_loc_cv = self._control_variate_fn(o_loc)
            self._ref_f_loc = _weighted_mean(ref_f_loc, self.reference_sampler.weights)
            self._ref_f_loc_cv = _weighted_mean(ref_f_loc_cv, self.reference_sampler.weights)
        else:
            self._ref_f_loc = _weighted_mean(o_loc, self.reference_sampler.weights)

        o_loc = self.operator.get_O_loc(
            sampler.samples, self.reference_sampler.net, logPsiS=sampler.logPsi
        )

        if self.control_variate is not None:
            self._psi_f_loc, self._psi_f_loc_cv, self._psi_f2_loc_cv, self._psi_f_loc_f2_loc_cv = self._control_variate_fn(o_loc)

            if self.control_variate == 'adaptive':
                if control_variate_coeff is not None:
                    self._control_variate_coeff = control_variate_coeff
                else:
                    self._control_variate_coeff = self._get_control_variate_coeff(sampler.weights)

            correction = self._control_variate_coeff * (self._psi_f_loc_cv * self._ref_f_loc_cv - 1.0)

        else:
            self._psi_f_loc = o_loc
            correction = 0

        infidelity_loc = 1.0 - self._psi_f_loc * self._ref_f_loc

        return SampledObs(infidelity_loc - correction, sampler.weights)
    
    def value_and_grad(
            self, 
            sampler: AbstractSampler, 
            sample_ref_state: bool=True, 
            **kwargs
        ) -> ObjectiveFunctionOutput:
        value = self(sampler, sample_ref_state=sample_ref_state, **kwargs)
        grad_log_psi = SampledObs(sampler.net.gradients(sampler.samples), sampler.weights)
        f_loc = SampledObs(self._psi_f_loc, sampler.weights)
        grad = grad_log_psi.get_covar_obs(f_loc)
        grad._observations *= - 2.0 * self._ref_f_loc

        return ObjectiveFunctionOutput(o_loc=value, grad=grad, grad_log_psi=grad_log_psi)
    
    def _reset_cached_observables(self):
        self._ref_f_loc = None
        self._ref_f_loc_cv = None
        self._ref_f2_loc_cv = None
        self._ref_f_loc_f2_loc_cv = None
        
        self._psi_f_loc = None
        self._psi_f_loc_cv = None
        self._psi_f2_loc_cv = None
        self._psi_f_loc_f2_loc_cv = None
        self._var_f2 = None

    def _get_control_variate_coeff(self, psi_p):
        r"""
        Update the adaptive control-variate coefficient.

        The coefficient is estimated from the covariance of the fidelity and its
        squared magnitude according to the control-variate construction used in
        infidelity minimization.
        """
        ref_f_loc_f2_loc_cv = _weighted_mean(self._ref_f_loc_f2_loc_cv, self.reference_sampler.weights)
        psi_f_loc_f2_loc_cv = _weighted_mean(self._psi_f_loc_f2_loc_cv, psi_p)
        psi_f_loc_cv = _weighted_mean(self._psi_f_loc_cv, psi_p)
        psi_f_loc = _weighted_mean(self._psi_f_loc, psi_p)
        ref_f2_loc_cv = _weighted_mean(self._ref_f2_loc_cv, self.reference_sampler.weights)
        psi_f2_loc_cv = _weighted_mean(self._psi_f2_loc_cv, psi_p)

        covar_ff2 = ref_f_loc_f2_loc_cv * psi_f_loc_f2_loc_cv - self._ref_f_loc_cv * self._ref_f_loc * psi_f_loc_cv * psi_f_loc
        self._var_f2 = ref_f2_loc_cv * psi_f2_loc_cv - (self._ref_f_loc_cv * psi_f_loc_cv) ** 2
        
        return self._moving_average.update(-jnp.abs(jnp.real(covar_ff2) / jnp.real(self._var_f2)))