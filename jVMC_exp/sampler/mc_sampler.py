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