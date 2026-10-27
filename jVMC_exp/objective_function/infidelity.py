import jax
import jax.numpy as jnp
import warnings

from jVMC_exp.vqs import NQS
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.operator.discrete.branch_free import Operator
from jVMC_exp.operator.discrete.branch_free import IdentityOperator
from jVMC_exp.stats import SampledObs
from jVMC_exp.sharding_config import sharded, MESH
from .base import AbstractObjectiveFunction, ObjectiveFunctionOutput

@jax.jit
def _moving_average_update(values, x):
    values = jnp.roll(values, -1).at[-1].set(x)
    return values, jnp.mean(values)

class MovingAverage:
    """Simple moving average used for the adaptive control-variate coefficient."""

    def __init__(self, width=10, init=0.0):
        self.values = jnp.full((width,), init)

    def update(self, x):
        self.values, mean = _moving_average_update(self.values, x)
        return mean

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
    r"""
    Infidelity between the trial state :math:`|\psi_\theta\rangle` (primary sampler)
    and a reference state :math:`|\chi\rangle` (``reference_sampler``),

    .. math::
        \mathcal I = 1 - \frac{\langle\psi_\theta|\hat O|\chi\rangle\langle\chi|\hat O^\dagger|\psi_\theta\rangle}
                              {\langle\psi_\theta|\psi_\theta\rangle\langle\chi|\chi\rangle}.

    The local loss is :math:`\ell(s) = 1 - R^\psi_{\rm loc}(s)\langle R^\chi_{\rm loc}\rangle_\chi` with
    :math:`R^\psi_{\rm loc}(s) = (\hat O\chi)(s)/\psi_\theta(s)`, :math:`s\sim|\psi_\theta|^2`, and
    :math:`R^\chi_{\rm loc}(s) = (\hat O^\dagger\psi_\theta)(s)/\chi(s)`, :math:`s\sim|\chi|^2`.
    This is the local estimator of the theta-independent Hermitian operator
    :math:`1 - \hat O|\chi\rangle\langle\chi|\hat O^\dagger/\langle\chi|\chi\rangle`, hence :math:`R_k=0` and

    .. math::
        F_k = -\,\mathrm{Cov}_p(\Gamma_k^*, R^\psi_{\rm loc})\,\langle R^\chi_{\rm loc}\rangle_\chi .

    Control variates only modify the estimate of :math:`\mathcal I`, not the force.
    """
    def __init__(
        self,
        reference_sampler: AbstractSampler,
        operator: Operator | None = None,
        conjugated_operator: Operator | None = None,
        control_variate: str | None = None,
        moving_average_width: int = 1,
        l_dim: int = 2,
        batched_jacobian: bool = False
    ):
        super().__init__(batched_jacobian)

        if operator is None and conjugated_operator is not None:
            raise ValueError(
                "A ConjugatedOperator was provided without an Operator."
            )

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
            self,
            sampler: AbstractSampler,
            sample_ref_state: bool = True,
            *,
            control_variate_coeff=None,
            **kwargs
        ) -> SampledObs:
        r"""
        Compute the local infidelity estimator on the configurations held by ``sampler``.

        Args:
            sampler: Sampler of the trial state :math:`|\psi_\theta\rangle`.
            sample_ref_state: Whether to resample the reference state before
                estimating :math:`\langle R^\chi_{\rm loc}\rangle_\chi`.
            control_variate_coeff: Optional override of the adaptive
                control-variate coefficient.

        Returns:
            The local infidelity estimator
            :math:`1 - R^\psi_{\rm loc}(s)\langle R^\chi_{\rm loc}\rangle_\chi`,
            minus the control-variate correction if enabled.
        """
        if sample_ref_state:
            self.reference_sampler.sample()

        o_loc = self._get_f_loc(
            self.conjugated_operator, self.reference_sampler.samples,
            self.reference_sampler.logPsi, sampler.psi
        )

        if self.control_variate is not None:
            ref_f_loc, ref_f_loc_cv, self._ref_f2_loc_cv, self._ref_f_loc_f2_loc_cv = self._control_variate_fn(o_loc)
            self._ref_f_loc = _weighted_mean(ref_f_loc, self.reference_sampler.weights)
            self._ref_f_loc_cv = _weighted_mean(ref_f_loc_cv, self.reference_sampler.weights)
        else:
            self._ref_f_loc = _weighted_mean(o_loc, self.reference_sampler.weights)

        o_loc = self._get_f_loc(
            self.operator, sampler.samples,
            sampler.logPsi, self.reference_sampler.psi
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
            compute_grad: bool = True,
            *,
            sample_ref_state: bool = True,
            **kwargs
        ) -> ObjectiveFunctionOutput:
        value = self(sampler, sample_ref_state=sample_ref_state, **kwargs)
        grad_log_psi = self._grad_log_psi(sampler)

        if compute_grad:
            f_loc = SampledObs(-self._ref_f_loc * self._psi_f_loc, sampler.weights)
            grad = grad_log_psi.get_covar(f_loc)

            return ObjectiveFunctionOutput(
                o_loc=value, grad=grad, grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=value, grad_log_psi=grad_log_psi)

    def _get_f_loc(self, operator: Operator, s, log_phi_s, psi: NQS):
        r"""
        Returns the local estimator :math:`\sum_{s'} O_{s,s'} \psi(s') / \phi(s)` of
        :math:`\langle\phi|O|\psi\rangle / \langle\phi|\phi\rangle`, where ``s`` is sampled
        from :math:`|\phi|^2` and ``log_phi_s`` is :math:`\log\phi(s)`.
        """
        evals_per_device = psi.batchSize // MESH.size
        chunk_size = max(1, min(operator.n_conn + 1, evals_per_device))

        return self._get_f_loc_sh(
            s, log_phi_s,
            parameters=psi.eval_parameters,
            operator=operator,
            psi=psi,
            chunk_size=chunk_size,
            batch_size=MESH.size * (evals_per_device // chunk_size)
        )

    @sharded(static_kwarg_names=("operator", "psi", "chunk_size"))
    def _get_f_loc_sh(
        self, s, log_phi_s, *,
        parameters, operator: Operator, psi: NQS, chunk_size, batch_size
    ):
        s_p, mat_els, mat_el_diag = operator._get_conn_elements(s, {})
        s_p = jnp.concatenate([s[None], s_p], axis=0)
        mat_els = jnp.concatenate([mat_el_diag[None], mat_els], axis=0)

        log_psi_s_p = jax.lax.map(
            lambda x: psi.apply_fun(parameters, x), s_p, batch_size=chunk_size
        ).astype(psi.out_dtype)

        return jnp.sum(mat_els * jnp.exp(log_psi_s_p - log_phi_s))

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
        Update the adaptive control-variate coefficient
        :math:`c = -|\mathrm{Re}\,\mathrm{Cov}(A, |A|^2)| / \mathrm{Var}(|A|^2)`, with
        :math:`A = R^\psi_{\rm loc}(x) R^\chi_{\rm loc}(y)` for independent :math:`x, y`,
        smoothed by a moving average.
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