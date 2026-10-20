import warnings
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from functools import partial

from jVMC_exp.util.util import has_callable_attr
from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import (
    MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING, 
)
from jVMC_exp.sampler.propose import AbstractProposer, AbstractProposeCont
from jVMC_exp.sampler.base import AbstractMCSampler
from jVMC_exp import global_defs

class MCSampler(AbstractMCSampler):
    """
    A sampler class.

    This abstract class provides functionality to sample computationally basis.
    """
    def _init_state(self):
        initializer = lambda key, shape, dtype: jax.random.bernoulli(key, 0.5, shape).astype(dtype)
        
        return self._init_state_general(initializer, global_defs.DT_SAMPLES)
    
class MCSamplerCont(AbstractMCSampler):
    def __init__(
        self, psi: NQS, updateProposer: None | AbstractProposeCont, key=None, 
        numChains=32, numSamples=128, thermalizationSweeps=10, sweepSteps=None, 
        initState=None, mu=2, logProbFactor=0.5
    ):
        if sweepSteps is None:
            sweepSteps = updateProposer.geometry.n_particles * updateProposer.geometry.n_dim
        super().__init__(psi, updateProposer, key , numChains, numSamples, 
                         thermalizationSweeps, sweepSteps, initState, mu, logProbFactor)
        
    def _init_state(self):
        return self._init_state_general(self.updateProposer.geometry.uniform_populate, global_defs.DT_SAMPLES_CONT)

class CutoffSampler(MCSampler):
    """
    A cutoff-based MCMC sampler class.

    This class provides functionality to sample computational basis states from
    a cutoff version of the distribution

        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    During the Markov chain update, the acceptance probability
    :math:`\\mu \\, \\text{Re}(\\log\\psi)` is clipped from below at
    ``cutoff = mu * max_s Re(log psi(s)) + log(eps)``. Configurations whose
    target probability would otherwise be smaller than the cutoff are sampled
    with the clipped probability instead. The returned weight for each
    sample is multiplied by the importance ratio
    :math:`r(s) = |\\psi(s)|^{\\mu} / \\max\\{|\\psi(s)|^{\\mu}, e^{\\text{cutoff}}\\}`
    so that estimators stay unbiased.

    Sampling is automatically distributed across processes and locally
    available devices via JAX sharding (see ``jVMC_exp.sharding_config``).

    Initializer arguments:
        * ``net``: Network defining the probability distribution.
        * ``updateProposer``: An instance of
            :class:`jVMC_exp.propose.AbstractProposer`.
        * ``eps``: Cutoff parameter, in the closed interval :math:`[0, 1]`.
            The cutoff is set to ``maxLogPsi + log(eps)``.
        * ``key``: PRNG key (``jax.random.PRNGKey`` or seed ``int``).
        * ``numChains``: Number of Markov chains run in parallel.
        * ``numSamples``: Default number of samples returned by
            :meth:`sample`.
        * ``thermalizationSweeps``: Sweeps used for thermalization.
        * ``sweepSteps``: Proposed updates per sweep.
        * ``initState``: Optional initial chain configurations.
        * ``mu``: Exponent of the target distribution :math:`|\\psi|^{\\mu}`.
        * ``logProbFactor``: Factor for log-probabilities (``0.5`` for pure
            wave functions, ``1.0`` for POVMs).
        * ``maxLogPsi``: Optional initial value of
            :math:`\\mu\\,\\max_s\\text{Re}(\\log\\psi(s))`. If ``None``, a
            quick bootstrap MC run is used to estimate it.
        * ``bootstrap_kwargs``: Optional dict of keyword arguments
            forwarded to the bootstrap :class:`MCSampler`. Use this to
            reduce the cost of the bootstrap (e.g., smaller
            ``numSamples``) or to override its key.
    """
    def __init__(
            self, psi: NQS, updateProposer: AbstractProposer, eps,
            key=None, numChains=32, numSamples=128,
            thermalizationSweeps=10, sweepSteps=None, initState=None,
            mu=2, logProbFactor=0.5, maxLogPsi=None, bootstrap_kwargs=None
        ):
        if psi.is_generator:
            raise RuntimeError(
                "CutoffSampler does not support generator (autoregressive) nets; "
                "use 'MCSampler' for those."
            )
        if not isinstance(updateProposer, AbstractProposer):
            raise RuntimeError(
                "'updateProposer' must be an instance of 'jVMC_exp.propose.AbstractProposer'."
            )
        if updateProposer._use_custom_thermalization:
            raise RuntimeError(
                "CutoffSampler does not support proposers with custom "
                "thermalization (e.g. RWM, MALA)."
            )

        super().__init__(
            psi, updateProposer, key=key, numChains=numChains, numSamples=numSamples, 
            thermalizationSweeps=thermalizationSweeps, sweepSteps=sweepSteps, 
            initState=initState, mu=mu, logProbFactor=logProbFactor
        )

        if maxLogPsi is None:
            maxLogPsi = self._bootstrap_max_log_psi(bootstrap_kwargs or {})

        self._maxLogPsi = jnp.asarray(maxLogPsi)
        self.eps = eps
        self.ratio = None
        self.renorm = None

        if has_callable_attr(self.psi.net, "eval_real"):
            def log_prob_fun(p, s):
                return (self.mu * self.psi.apply_fun(p, s, method=self.psi.net.eval_real)
                        .astype(global_defs.DT_OUT_REAL))
        else:
            def log_prob_fun(p, s):
                return (self.mu * jnp.real(self.psi.apply_fun(p, s))
                        .astype(global_defs.DT_OUT_REAL))
        self._log_prob_fun_jsh = jax.jit(
            jax.shard_map(
                jax.vmap(log_prob_fun, in_axes=(None, 0)),
                mesh=MESH,
                in_specs=(REPLICATED_SPEC, DEVICE_SPEC),
                out_specs=DEVICE_SPEC
            )
        )

        if self.psi.eval_ratio:
            if jax.process_index() == 0:
                warnings.warn(
                    "CutoffSampler does not support nets with `eval_ratio` method; "
                    "using the default log probability function."
                )
        else:
            def get_ratio(log_prob, log_prob_correction, params, state, new_state, cutoff):
                new_log_prob = jnp.maximum(log_prob_fun(params, new_state), cutoff)

                return jnp.exp(new_log_prob - log_prob + log_prob_correction), new_log_prob
        self._get_ratio = get_ratio

    @property
    def eps(self): 
        return self._eps

    @eps.setter
    def eps(self, value):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("`eps` must be in the closed interval [0, 1].")
        self._eps = float(value)
        self._update_cutoff()

    @property
    def maxLogPsi(self):
        return self._maxLogPsi

    @maxLogPsi.setter
    def maxLogPsi(self, value):
        self._maxLogPsi = jnp.asarray(value)
        self._update_cutoff()

    @property
    def cutoff(self):
        return self._cutoff

    def _update_cutoff(self):
        """Recompute the cutoff from the current `eps` and `maxLogPsi`."""
        self._cutoff = self._maxLogPsi + jnp.log(self._eps)

    def _bootstrap_max_log_psi(self, bootstrap_kwargs):
        """Run a small MCSampler to obtain an initial estimate of max log psi."""
        defaults = dict(
            key=random.PRNGKey(4321),
            numChains=25,
            sweepSteps=int(np.prod(self.sampleShape)),
            numSamples=self.numSamples,
            thermalizationSweeps=50,
            mu=self.mu,
            logProbFactor=self.logProbFactor,
        )
        defaults.update(bootstrap_kwargs)
        bootstrap = MCSampler(self.psi, updateProposer=self.updateProposer, **defaults)
        _, coeffs, _ = bootstrap.sample()

        return jnp.max(self.mu * jnp.real(coeffs))

    def _get_samples_mcmc(self):
        if not self._is_state_initialized:
            self._init_state()
        self.updateProposer.update_arg(self.psi)
        self.logProb = jnp.maximum(
            self._log_prob_fun_jsh(self.psi.sampler_parameters, self.states),
            self.cutoff
        )
        self.numProposed = jax.device_put(
            jnp.zeros((self.numChains,), dtype=np.int64), DEVICE_SHARDING
        )
        self.numAccepted = jax.device_put(
            jnp.zeros((self.numChains,), dtype=np.int64), DEVICE_SHARDING
        )

        numSamplesStr = str(self._samplePerChain)

        if numSamplesStr not in self._get_samples_jsh:
            get_samples = partial(
                self._get_samples,
                sweepFunction=partial(self._sweep, get_ratio=self._get_ratio),
                updateProposer=self.updateProposer,
                numSamples=self._samplePerChain,
                thermSweeps=self.thermalizationSweeps,
                sweepSteps=self.sweepSteps,
                sampleShape=self.sampleShape,
            )

            self._get_samples_jsh[numSamplesStr] = jax.jit(
                jax.shard_map(
                    get_samples,
                    mesh=MESH,
                    in_specs=(REPLICATED_SPEC,) * 2 + (DEVICE_SPEC,) * 5 + (self.updateProposer.arg_in_specs,), 
                    out_specs=((DEVICE_SPEC,) * 5, DEVICE_SPEC) + (self.updateProposer.arg_in_specs,),
                )
            )

        (self._states, self.logProb, self._key, self.numProposed, self.numAccepted), configs, self.updateProposer._arg = \
            self._get_samples_jsh[numSamplesStr](
                    self.psi.sampler_parameters, self.cutoff, self.states, self.logProb, self.key,
                    self.numProposed, self.numAccepted, self.updateProposer._arg
                )
        
        coeffs = self.psi(configs)

        # Importance sampling weights for cutoff distribution
        logPMu = self.mu * jnp.real(coeffs)
        logRatio = jnp.minimum(logPMu - self.cutoff, 0.0)
        logWeights = (1.0 / self.logProbFactor - self.mu) * jnp.real(coeffs) + logRatio
        p = jnp.exp(logWeights - jnp.max(logWeights))
        self.ratio = jnp.exp(logRatio)
        self.renorm = self.numSamples / jnp.sum(self.ratio)
        self.maxLogPsi = jnp.max(logPMu)

        return configs, coeffs, p / jnp.sum(p)

    def _get_samples(
            self, params, cutoff, states, logProb, key, numProposed, numAccepted, updateProposerArg, 
            numSamples, thermSweeps, sweepSteps, updateProposer, sweepFunction, sampleShape
        ):
        if thermSweeps is not None:
            states, logProb, key, numProposed, numAccepted = sweepFunction(
                states, logProb, key, numProposed, numAccepted, cutoff, params,
                thermSweeps * sweepSteps, updateProposer, updateProposerArg
            )

        def scan_fun(c, x):
            states, logProb, key, numProposed, numAccepted = sweepFunction(
                c[0], c[1], c[2], c[3], c[4], cutoff, params, sweepSteps,
                updateProposer, updateProposerArg
            )
            return (states, logProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(
            scan_fun, (states, logProb, key, numProposed, numAccepted),
            None, length=numSamples
        )
        return meta, jnp.swapaxes(configs, 0, 1).reshape((-1,) + sampleShape), updateProposerArg

    def _sweep(
            self, states, logProb, key, numProposed, numAccepted, cutoff, params,
            numSteps, updateProposer, updateProposerArg, get_ratio
        ):
        def perform_mc_update_single_chain(state, logProb, key_single, ProposerArg):
            # Generate update proposal
            proposerKey, newKey = random.split(key_single)
            newState, log_prob_correction = updateProposer(proposerKey, state, ProposerArg)

            # Compute acceptance probability
            P, newLogProb = get_ratio(logProb, log_prob_correction, params, state, newState, cutoff)
            
            # Roll dice
            acceptKey, newKey = random.split(newKey)
            accepted = random.bernoulli(acceptKey, P)

            # Perform update if accepted
            newState = jax.lax.cond(accepted, lambda: newState, lambda: state)
            newLogProb = jax.lax.cond(accepted, lambda: newLogProb, lambda: logProb)

            return newState, newLogProb, newKey, accepted

        def sweep_step(carry, _):
            states, logProb, keys, numProposed, numAccepted = carry

            newStates, newLogProb, newKeys, accepted = jax.vmap(
                perform_mc_update_single_chain,
                in_axes=(0, 0, 0, self.updateProposer.arg_in_axes)
            )(states, logProb, keys, updateProposerArg)

            numProposed = numProposed + 1
            numAccepted = numAccepted + accepted

            return (newStates, newLogProb, newKeys, numProposed, numAccepted), None

        (states, logProb, key, numProposed, numAccepted), _ = jax.lax.scan(
            sweep_step, 
            (states, logProb, key, numProposed, numAccepted),
            None, 
            length=numSteps
        )

        return states, logProb, key, numProposed, numAccepted

def _logit(x):
    return jnp.log(x) - jnp.log1p(-x)


class TargetFractionCutoffSampler(CutoffSampler):
    """
    A cutoff sampler whose cutoff is set by the fraction of samples it clips.

    Like :class:`CutoffSampler`, the Markov chain samples the flattened law

        :math:`q(s) \\propto \\max\\{|\\psi(s)|^{\\mu}, e^{c}\\}`

    and the returned weights undo the flattening. The cutoff, however, is not
    tied to :math:`\\max_s |\\psi(s)|^{\\mu}`. It is

        :math:`c = r + \\delta`,

    where the reference :math:`r` is the importance-weighted mean of
    :math:`\\mu\\,\\text{Re}\\log\\psi` over the last batch of samples, and the
    offset :math:`\\delta` is adjusted after every call to :meth:`sample` such that
    the fraction of samples on the flat part of :math:`q` stays close to
    ``targetFraction``.

    Why not a fixed ``eps`` relative to the maximum: the flat part of :math:`q`
    carries the weight :math:`e^{c}` times the number of configurations below the
    cutoff, which is up to :math:`2^N`. Once the state has spread out, a cutoff at
    a fixed ratio to the maximum lies above the bulk of :math:`|\\psi|^{\\mu}`, the
    chains drift onto the plateau where their weights are tiny, and the effective
    sample size collapses. The sample maximum is in addition an extreme value that
    fluctuates from batch to batch and feeds back into the next cutoff. The
    fraction of samples below the cutoff is the weight of the plateau in
    :math:`q`, an average over the whole batch, and directly bounds the loss of
    effective sample size.

    Controller: with :math:`f` the fraction of the last batch below the cutoff
    that batch was sampled with,

        :math:`\\delta \\leftarrow \\min\\{\\delta + \\text{clip}(\\gamma\\,
        [\\text{logit}(f_{\\text{target}}) - \\text{logit}(f)],
        -\\Delta_{\\downarrow}, \\Delta_{\\uparrow}),\\ \\delta_{\\max}\\}`.

    :math:`f` is clipped to :math:`[1/(2n), 1 - 1/(2n)]` for a batch of :math:`n`
    samples, so an empty or a full plateau gives a finite step. The new cutoff
    applies from the next call on: the weights of each batch are computed with
    the cutoff its chains were run with, so every batch is unbiased on its own.

    Initializer arguments (beyond those of :class:`MCSampler`):
        * ``targetFraction``: Target fraction of samples below the cutoff, in the
            open interval :math:`(0, 1)`.
        * ``damping``: Gain :math:`\\gamma \\in [0, 1]` of the controller. ``0``
            freezes the offset.
        * ``maxIncrease``, ``maxDecrease``: Largest change of :math:`\\delta` per
            call upwards and downwards. Raising the cutoff is what can make the
            plateau blow up, so the default step up is smaller than the one down.
        * ``maxLogOffset``: Upper bound :math:`\\delta_{\\max}` on the offset.
        * ``logOffset``, ``reference``: Initial :math:`\\delta` and :math:`r`. If
            either is ``None``, both are obtained from a bootstrap run, see
            :meth:`initialize_cutoff`.
        * ``bootstrap_kwargs``: Keyword arguments for the bootstrap
            :class:`MCSampler`.

    Diagnostics of the last call to :meth:`sample`: ``last_cutoff``,
    ``fraction_below`` and ``effective_sample_fraction`` (Kish effective sample
    size divided by the number of samples).

    The ``eps`` and ``maxLogPsi`` attributes inherited from :class:`CutoffSampler`
    do not enter the cutoff.
    """
    def __init__(
            self, psi: NQS, updateProposer: AbstractProposer, targetFraction=0.02,
            key=None, numChains=32, numSamples=128,
            thermalizationSweeps=10, sweepSteps=None, initState=None,
            mu=2, logProbFactor=0.5, damping=0.3, maxIncrease=0.5, maxDecrease=2.0,
            maxLogOffset=0.0, logOffset=None, reference=None, bootstrap_kwargs=None
        ):
        if not (0.0 <= float(damping) <= 1.0):
            raise ValueError("`damping` must be in the closed interval [0, 1].")
        if float(maxIncrease) < 0 or float(maxDecrease) < 0:
            raise ValueError("`maxIncrease` and `maxDecrease` must be non-negative.")

        self.targetFraction = targetFraction
        self.damping = float(damping)
        self.maxIncrease = float(maxIncrease)
        self.maxDecrease = float(maxDecrease)
        self.maxLogOffset = float(maxLogOffset)

        # Placeholders, the parent's __init__ already evaluates the cutoff.
        self._reference = jnp.asarray(0.0)
        self._logOffset = jnp.asarray(0.0)

        # eps and maxLogPsi are passed only to skip the parent's bootstrap.
        super().__init__(
            psi, updateProposer, eps=1.0, key=key, numChains=numChains,
            numSamples=numSamples, thermalizationSweeps=thermalizationSweeps,
            sweepSteps=sweepSteps, initState=initState, mu=mu,
            logProbFactor=logProbFactor, maxLogPsi=0.0
        )

        self.last_cutoff = None
        self.fraction_below = None
        self.effective_sample_fraction = None

        if reference is None or logOffset is None:
            self.initialize_cutoff(bootstrap_kwargs)
        if reference is not None:
            self.reference = reference
        if logOffset is not None:
            self.logOffset = logOffset

    @property
    def targetFraction(self):
        return self._targetFraction

    @targetFraction.setter
    def targetFraction(self, value):
        if not (0.0 < float(value) < 1.0):
            raise ValueError("`targetFraction` must be in the open interval (0, 1).")
        self._targetFraction = float(value)

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, value):
        self._reference = jnp.asarray(value)
        self._update_cutoff()

    @property
    def logOffset(self):
        return self._logOffset

    @logOffset.setter
    def logOffset(self, value):
        value = jnp.asarray(value)
        if not jnp.isfinite(value):
            raise ValueError("`logOffset` must be finite.")
        self._logOffset = jnp.minimum(value, self.maxLogOffset)
        self._update_cutoff()

    def _update_cutoff(self):
        self._cutoff = self._reference + self._logOffset

    def initialize_cutoff(self, bootstrap_kwargs=None):
        """(Re-)initialize the cutoff from a plain MC run with the current parameters.

        The reference is set to the weighted mean of :math:`\\mu\\,\\text{Re}\\log\\psi`
        and the cutoff to just below the smallest sampled value, so that the first
        batch is (almost) not clipped and the controller raises the cutoff from
        there. Call this after the parameters have been changed by hand, e.g. after
        loading a state, when the current cutoff is meaningless.
        """
        defaults = dict(
            key=random.PRNGKey(4321),
            numChains=25,
            sweepSteps=int(np.prod(self.sampleShape)),
            numSamples=min(self.numSamples, 1000),
            thermalizationSweeps=50,
            mu=self.mu,
            logProbFactor=self.logProbFactor,
        )
        defaults.update(bootstrap_kwargs or {})
        bootstrap = MCSampler(self.psi, updateProposer=self.updateProposer, **defaults)
        _, coeffs, p = bootstrap.sample()
        logPMu = self.mu * jnp.real(coeffs)

        self._reference = jnp.sum(p * logPMu) / jnp.sum(p)
        self.logOffset = jnp.min(logPMu) - self._reference - 1.0

    def _next_log_offset(self, fraction, num_samples):
        f = jnp.clip(fraction, 0.5 / num_samples, 1.0 - 0.5 / num_samples)
        step = self.damping * (_logit(self.targetFraction) - _logit(f))
        step = jnp.clip(step, -self.maxDecrease, self.maxIncrease)

        return jnp.minimum(self._logOffset + step, self.maxLogOffset)

    def _get_samples_mcmc(self):
        cutoff = self.cutoff
        # The parent runs the chains and computes the weights with this cutoff.
        configs, coeffs, p = super()._get_samples_mcmc()

        logPMu = self.mu * jnp.real(coeffs)
        num_samples = logPMu.shape[0]
        self.last_cutoff = cutoff
        self.fraction_below = jnp.mean(logPMu < cutoff)
        self.effective_sample_fraction = 1.0 / (num_samples * jnp.sum(p ** 2))

        # Applies from the next call on.
        self._reference = jnp.sum(p * logPMu)
        self._logOffset = self._next_log_offset(self.fraction_below, num_samples)
        self._update_cutoff()

        return configs, coeffs, p
