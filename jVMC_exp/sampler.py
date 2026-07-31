from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from jVMC_exp.util.key_gen import format_key
from jVMC_exp.util.util import has_callable_attr
from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import (
    MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING, 
    distribute, broadcast_split_key
)
from jVMC_exp.propose import AbstractProposer, AbstractProposeCont

from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.propose import AbstractProposeCont, AbstractProposer
from jVMC_exp.sharding_config import (DEVICE_SHARDING, DEVICE_SPEC, MESH,
                                      REPLICATED_SPEC, broadcast_split_key,
                                      distribute)
from jVMC_exp.stats import SampledObs

from jVMC_exp import global_defs


class AbstractSampler(ABC):
    def __init__(self, psi: NQS):
        self._psi = psi
        self._samples = None
        self._logPsi = None
        self._weights = None

    @property
    def psi(self):
        return self._psi
    
    @property
    def sampleShape(self):
        return self.psi.sampleShape
    
    @property
    def logPsi(self):
        if self._samples is None:
            self.sample()
        
        return self._logPsi
    
    @property
    def weights(self):
        if self._samples is None:
            self.sample()

        return self._weights
    
    @property
    def samples(self):
        if self._samples is None:
            self.sample()

        return self._samples
    
    @abstractmethod
    def __call__(self, observable: AbstractOperator, **obs_kwargs) -> SampledObs:
        pass

    @abstractmethod
    def sample(self, numSamples=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Sample configurations from the distribution defined by the network.
        
        Args.:
            numSamples: Optional number of samples to generate. If None, the default number of samples is used.
        Returns:
            A tuple of (configs, logPsi, p), where:
            - configs: Sampled configurations.
            - logPsi: Logarithm of the wave function coefficients for the sampled configurations.
            - p: Normalized probabilities of the sampled configurations.
        '''
        pass

class AbstractMCSampler(AbstractSampler):
    """
    A sampler class.

    This abstract class provides functionality to sample states from \
    the distribution 

        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    For :math:`\\mu=2` this corresponds to sampling from the Born distribution. \
    :math:`0\\leq\\mu<2` can be used to perform importance sampling \
    (see `[arXiv:2108.08631] <https://arxiv.org/abs/2108.08631>`_).

    Sampling is automatically distributed accross processes and locally available devices.

    Initializer arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of configurations.
        * ``key``: An instance of ``jax.random.PRNGKey``. Alternatively, an ``int`` that will be used \
                   as seed to initialize a ``PRNGKey``.
        * ``updateProposer``: An instance of jVMC.proposer.AbstractProposer to propose updates for the MCMC algorithm. \
        It is called as ``updateProposer(key, config, **kwargs)``, where ``key`` is an instance of \
        ``jax.random.PRNGKey``, ``config`` is a computational basis configuration, and ``**kwargs`` \
        are optional additional arguments.
        * ``numChains``: Number of Markov chains, which are run in parallel.
        * ``updateProposerArg``: An optional argument that will be passed to the ``updateProposer`` \
        as ``kwargs``.
        * ``numSamples``: Default number of samples to be returned by the ``sample()`` member function.
        * ``thermalizationSweeps``: Number of sweeps to perform for thermalization of the Markov chain.
        * ``sweepSteps``: Number of proposed updates per sweep.
        * ``mu``: Parameter for the distribution :math:`p_{\\mu}(s)`, see above.
        * ``logProbFactor``: Factor for the log-probabilities, aquivalent to the exponent for the probability \
        distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs. In the POVM case, the \
        ``mu`` parameter must be set to 1.0, to sample the unchanged POVM distribution.
    """

    def __init__(
            self, psi: NQS, updateProposer: None | AbstractProposer, key=None, 
            numChains=32, numSamples=128, thermalizationSweeps=10, sweepSteps=None, 
            initState=None, mu=2, logProbFactor=0.5
        ):
        if (not psi.is_generator) and (not isinstance(updateProposer, AbstractProposer)):
            raise RuntimeError(
                "Instantiation of MCSampler: `updateProposer` is `None` and cannot be used for MCMC sampling. "
                "'updateProposer' must be an instance of 'jVMC.propose.AbstractProposer'."
            )
        super().__init__(psi)

        self.initial_states = initState
        if initState is not None: 
            self.initial_states = jnp.array(initState)
            if self.initial_states.shape[1:] != self.sampleShape:
                raise ValueError(
                    f"The provided initState has the wrog sample shape. "
                    f"Got {self.initial_states.shape[1:]}, while sampleShape is {self.sampleShape}."
                )
            elif numChains - self.initial_states[0] < 0:
                raise ValueError(
                    f"The number of chain in initState ({self.states.shape[0]}) "
                    f"is greater than the provided numChains ({numChains})."
                )

        self.logProbFactor = logProbFactor
        self.mu = mu
        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self._updateProposer = updateProposer

        self.key = key

        if sweepSteps is None:
            sweepSteps = self.sampleShape[-1]
        self.sweepSteps = sweepSteps
        self.thermalizationSweeps = thermalizationSweeps
        self.numSamples = numSamples
        self.numChains = numChains

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
            def get_ratio(log_prob, log_prob_correction, params, state, new_state):
                abs_ratio = jnp.abs(
                    self.psi.apply_fun(
                        params, state, new_state, method=self.psi.net.eval_ratio
                    )
                ).astype(global_defs.DT_OUT_REAL)

                return abs_ratio**self.mu * jnp.exp(log_prob_correction), log_prob
        else:
            def get_ratio(log_prob, log_prob_correction, params, state, new_state):
                new_log_prob = log_prob_fun(params, new_state)

                return jnp.exp(new_log_prob - log_prob + log_prob_correction), new_log_prob
        self._get_ratio = get_ratio

    @property
    def thermalizationSweeps(self):
        return self._thermalizationSweeps
    
    @thermalizationSweeps.setter
    def thermalizationSweeps(self, value):
        self._thermalizationSweeps = value
        self._get_samples_jsh = {}
    
    @property
    def sweepSteps(self):
        return self._sweepSteps
    
    @sweepSteps.setter
    def sweepSteps(self, value):
        self._sweepSteps = value
        self._get_samples_jsh = {}

    @property
    def updateProposer(self):
        return self._updateProposer

    @property
    def numChains(self):
        return self._numChains
    
    @numChains.setter
    def numChains(self, value):
        if value > self.psi.batchSize:
            Warning(
                f"numChains ({value}) is larger than the batch size ({self.psi.batchSize}), "
                "which may lead to an out-of-memory error. "
                "Automatically setting numChains = batchSize."
            )
            value = self.psi.batchSize
            if self.initial_states is not None:
                if value - self.initial_states[0] < 0:
                    self.initial_states = self.initial_states[:value]
        if self.numSamples < value:
            raise ValueError(
                f"The provided number of chains {value} is bigger "
                f"than the number of samples {self.numSamples}."
            )

        self._numChains = value
        self._is_state_initialized = False
        self._get_samples_jsh = {}
        self._randomize_samples_jsh = {}

    @property
    def key(self):
        return self._key
    
    @key.setter
    def key(self, value):
        self._key = format_key(value)
        self._is_state_initialized = False
    
    def __call__(
            self, 
            observable: AbstractOperator, 
            num_samples: int | None=None,
            resample: bool = False,
            **obs_kwargs 
    ) -> SampledObs:
        """
        Evaluate an observable using Monte Carlo samples of the current variational state.

        This method draws (or reuses) samples from the underlying sampler and computes
        the local estimator of the given observable. Resampling is triggered if no
        samples are currently stored, if the requested number of samples differs from
        the stored one, or if `resample=True`.

        Parameters
        ----------
        observable : AbstractOperator

        num_samples : int or None, optional
            Number of Monte Carlo samples to draw. If None, the previously used
            number of samples is reused.

        resample : bool, optional
            If True, forces resampling even if cached samples are available.

        **obs_kwargs
            Additional keyword arguments forwarded to
            `observable.get_O_loc(...)`.

        Returns
        -------
        SampledObs
        """
        needs_resample = (
            self.samples is None
            or (num_samples is not None and num_samples != self.numSamples)
            or resample
        )
        if needs_resample:
            self._samples, self._logPsi, self._weights = self.sample(num_samples)
        raw_data = observable.get_O_loc(self.samples, self.psi, logPsiS=self.logPsi, **obs_kwargs)

        return SampledObs(raw_data, self.weights)
  
    def reset(self, key=None):
        self.key = key

    @abstractmethod
    def _init_state(self):
        pass

    def _init_state_general(self, initializer, dtype):
        master_key = self.key[0] if self.key.ndim > 1 else self.key
        all_keys = broadcast_split_key(master_key, self.numChains + 1)
        keys = all_keys[:-1]
        initStateKey = all_keys[-1]
        self._key = jax.device_put(keys, DEVICE_SHARDING)

        if self.initial_states is not None:
            self.states = self.initial_states.astype(dtype)
            res = self.numChains - self.states.shape[0]
            if res > 0:
                pad = initializer(initStateKey, (res,) + self.sampleShape, dtype)
                self.states = jnp.concat([self.initial_states, pad])
        else:
            self.states = initializer(initStateKey, (self.numChains,) + self.sampleShape, dtype)
        self.states = jax.device_put(self.states, DEVICE_SHARDING)
        
        self.updateProposer.init_arg(self.psi, self.numChains)

        self._is_state_initialized = True
    
    def _distribute_sampling(self):
        """
        Distribute MCMC sampling tasks across sharded devices.

        This method ensures that the number of chains and samples per chain are 
        compatible with the device mesh used for JAX sharding. It adjusts chain 
        and sample counts as needed to maintain uniform distribution across devices.

        Device constraints:
            - The number of chains must be >= total_devices
            - The number of chains must be divisible by total_devices
            - Each chain generates the same number of samples

        The method performs the following adjustments:
            1. If numChains < total_devices: increases numChains to total_devices
            2. If numChains is not divisible by total_devices: rounds up to the 
               next multiple of total_devices
            3. Rounds up samples per chain using ceiling division to ensure 
               at least numSamples total samples are generated
        """
        self._numChains = distribute(self.numChains, 'chains')
        
        # Use ceiling division to ensure at least numSamples total samples
        self._samplePerChain = (self.numSamples + self.numChains - 1) // self.numChains
        totalSamples = self._samplePerChain * self.numChains
        
        if totalSamples > self.numSamples:
            print(f"INFO: Total samples adjusted: {self.numSamples} -> {totalSamples}")
        self.numSamples = totalSamples

    def sample(self, numSamples=None):
        """
        Generate random samples from wave function.

        If supported by ``net``, direct sampling is peformed. Otherwise, MCMC is run \
        to generate the desired number of samples. For direct sampling the real part \
        of ``net`` needs to provide a ``sample()`` member function that generates \
        samples from :math:`p_{\\mu}(s)`.

        Sampling is automatically distributed accross MPI processes and available \
        devices. In that case the number of samples returned might exceed ``numSamples``.

        Arguments:
            * ``numSamples``: Number of samples to generate. When running multiple processes \
            or on multiple devices per process, the number of samples returned is \
            ``numSamples`` or more. If ``None``, the default number of samples is returned \
            (see ``set_number_of_samples()`` member function).
            * ``multipleOf``: This argument allows to choose the number of samples returned to \
            be the smallest multiple of ``multipleOf`` larger than ``numSamples``. This feature \
            is useful to distribute a total number of samples across multiple processors in such \
            a way that the number of samples per processor is identical for each processor.

        Returns:
            Samples drawn from :math:`p_{\\mu}(s)`.
        """

        if numSamples is not None:
            samples_tmp = self.numSamples 
            self.numSamples = numSamples

        if self.psi.is_generator:
            configs, logPsi, p = self._get_samples_gen()
        else:
            configs, logPsi, p = self._get_samples_mcmc()
             
        if numSamples is not None:
            self.numSamples = samples_tmp

        self._samples = configs
        self._logPsi = logPsi
        self._weights = p

        return configs, logPsi, p

    def _get_samples_gen(self):
        self._key, sample_key = random.split(self.key)
        samples = self.psi.sample(self.numSamples, sample_key)
        
        return samples, self.psi(samples), jnp.ones(self.numSamples) / self.numSamples

    def _get_samples_mcmc(self):
        self._distribute_sampling()
        if not self._is_state_initialized:
            self._init_state()
        self.updateProposer.update_arg(self.psi)
        self.logProb = self._log_prob_fun_jsh(self.psi.sampler_parameters, self.states)
        self.numProposed = jax.device_put(jnp.zeros((self.numChains,), dtype=np.int64), DEVICE_SHARDING)
        self.numAccepted = jax.device_put(jnp.zeros((self.numChains,), dtype=np.int64), DEVICE_SHARDING)

        numSamplesStr = str(self._samplePerChain)

        # check whether _get_samples is already compiled for given number of samples.
        # partial over numSamples, thermSweeps and sweepSteps is needed cause these must be static
        # at the level of the shard_map, as they are used in the jax loops.
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
                    in_specs=(REPLICATED_SPEC,) + (DEVICE_SPEC,) * 5 + (self.updateProposer.arg_in_specs,),
                    out_specs=((DEVICE_SPEC,) * 5, DEVICE_SPEC) + (self.updateProposer.arg_in_specs,)
                )
            )

        (self.states, self.logProb, self._key, self.numProposed, self.numAccepted), configs, self.updateProposer._arg =\
            self._get_samples_jsh[numSamplesStr](
                self.psi.sampler_parameters, self.states, self.logProb, self.key, 
                self.numProposed, self.numAccepted, self.updateProposer._arg
            )

        coeffs = self.psi(configs)
        p = jnp.exp((1.0 / self.logProbFactor - self.mu) * jnp.real(coeffs))

        return configs, coeffs, p / jnp.sum(p)

    def _get_samples(    
            self, params, states, logProb, key, numProposed, numAccepted, updateProposerArg,
            numSamples, thermSweeps, sweepSteps, updateProposer, sweepFunction, sampleShape
        ):
        # Thermalize
        if self.thermalizationSweeps is not None:
            if updateProposer._use_custom_thermalization:
                states, logProb, key, numProposed, numAccepted, updateProposerArg = updateProposer._custom_therm_fun(
                    states, logProb, key, numProposed, numAccepted, params, 
                    sweepSteps, thermSweeps, sweepFunction, updateProposerArg
                )
            else:
                states, logProb, key, numProposed, numAccepted = sweepFunction(
                    states, logProb, key, numProposed, numAccepted, params, 
                    thermSweeps * sweepSteps, updateProposer, updateProposerArg
                )

        # Collect samples
        def scan_fun(c, x):
            states, logProb, key, numProposed, numAccepted = sweepFunction(
                c[0], c[1], c[2], c[3], c[4], params, sweepSteps, updateProposer, updateProposerArg
            )

            return (states, logProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logProb, key, numProposed, numAccepted), None, length=numSamples)

        # Reshape in from (numChains, numSamplesPerChain, sampleShape) to (numChains * numSamplesPerChain, sampleShape)
        return meta, configs.reshape((configs.shape[0] * configs.shape[1],) + sampleShape), updateProposerArg

    def _sweep(
            self, states, logProb, key, numProposed, numAccepted, params, 
            numSteps, updateProposer, updateProposerArg, get_ratio
        ):
        def perform_mc_update_single_chain(state, logProb, key_single, ProposerArg):
            # Generate update proposal
            proposerKey, newKey = random.split(key_single)
            newState, log_prob_correction = updateProposer(proposerKey, state, ProposerArg)  
            
            # Compute acceptance probability
            P, newLogProb = get_ratio(logProb, log_prob_correction, params, state, newState)
            
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
        
    def acceptance_ratio(self):
        """Get acceptance ratio.

        Returns:
            Acceptance ratio observed in the last call to ``sample()``.
        """

        numProp = jnp.sum(self.numProposed)
        if numProp > 0:
            return jnp.sum(self.numAccepted) / numProp

        return jnp.array([0.])

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

class CutoffSampler(AbstractSampler):
    """A cutoff-based MCMC sampler class.

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

    def __init__(self, psi: NQS, updateProposer: AbstractProposer, eps,
                 key=None, numChains=32, numSamples=128,
                 thermalizationSweeps=10, sweepSteps=None, initState=None,
                 mu=2, logProbFactor=0.5, maxLogPsi=None, bootstrap_kwargs=None):
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

        super().__init__(psi)

        self.initial_states = initState
        if initState is not None:
            self.initial_states = jnp.array(initState)
            if self.initial_states.shape[1:] != self.sampleShape:
                raise ValueError(f"The provided initState has the wrong sample shape. "
                                 f"Got {self.initial_states.shape[1:]}, while sampleShape is {self.sampleShape}.")
            elif numChains - self.initial_states.shape[0] < 0:
                raise ValueError(f"The number of chains in initState ({self.initial_states.shape[0]}) "
                                 f"is greater than the provided numChains ({numChains}).")

        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self.mu = mu
        self.logProbFactor = logProbFactor
        self._updateProposer = updateProposer

        self.key = key

        if sweepSteps is None:
            sweepSteps = self.sampleShape[-1]
        self.sweepSteps = sweepSteps
        self.thermalizationSweeps = thermalizationSweeps
        self.numSamples = numSamples
        self.numChains = numChains

        # Log-probability of a single configuration, log q_0(s) = mu * Re log psi(s).
        # Same convention as AbstractMCSampler, so that `mu` and `eval_real` are
        # handled identically.
        if has_callable_attr(self.psi.net, "eval_real"):
            def log_prob_fun(p, s):
                return (self.mu * self.psi.apply_fun(p, s, method=self.psi.net.eval_real)
                        .astype(global_defs.DT_OUT_REAL))
        else:
            def log_prob_fun(p, s):
                return (self.mu * jnp.real(self.psi.apply_fun(p, s))
                        .astype(global_defs.DT_OUT_REAL))
        self._log_prob_fun = log_prob_fun
        self._log_prob_fun_jsh = jax.jit(
            jax.shard_map(
                jax.vmap(log_prob_fun, in_axes=(None, 0)),
                mesh=MESH,
                in_specs=(REPLICATED_SPEC, DEVICE_SPEC),
                out_specs=DEVICE_SPEC
            )
        )

        # Cutoff bookkeeping
        if not (0.0 <= float(eps) <= 1.0):
            raise ValueError("`eps` must be in the closed interval [0, 1].")
        self._eps = float(eps)

        if maxLogPsi is None:
            maxLogPsi = self._bootstrap_max_log_psi(bootstrap_kwargs or {})
        self._maxLogPsi = jnp.asarray(maxLogPsi)
        self._cutoff = self._maxLogPsi + jnp.log(self._eps)

    # ---- Properties with cache-invalidating setters ----

    @property
    def thermalizationSweeps(self):
        return self._thermalizationSweeps

    @thermalizationSweeps.setter
    def thermalizationSweeps(self, value):
        self._thermalizationSweeps = value
        self._get_samples_jsh = {}

    @property
    def sweepSteps(self):
        return self._sweepSteps

    @sweepSteps.setter
    def sweepSteps(self, value):
        self._sweepSteps = value
        self._get_samples_jsh = {}

    @property
    def updateProposer(self):
        return self._updateProposer

    @property
    def numChains(self):
        return self._numChains

    @numChains.setter
    def numChains(self, value):
        self._numChains = value
        self._is_state_initialized = False
        self._get_samples_jsh = {}

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = format_key(value)
        self._is_state_initialized = False

    # ---- Cutoff bookkeeping ----

    @property
    def eps(self):
        return self._eps

    @property
    def maxLogPsi(self):
        return self._maxLogPsi

    @property
    def cutoff(self):
        return self._cutoff

    def update_cutoff(self):
        """Recompute the cutoff from the current `eps` and `maxLogPsi`."""
        self._cutoff = self._maxLogPsi + jnp.log(self._eps)

    def update_eps(self, eps):
        """Update `eps` and recompute the cutoff."""
        if not (0.0 <= float(eps) <= 1.0):
            raise ValueError("`eps` must be in the closed interval [0, 1].")
        self._eps = float(eps)
        self.update_cutoff()

    def update_maxLogPsi(self, maxLogPsi):
        """Update `maxLogPsi` and recompute the cutoff."""
        self._maxLogPsi = jnp.asarray(maxLogPsi)
        self.update_cutoff()

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

    # ---- AbstractSampler interface ----

    def __call__(
        self,
        observable: AbstractOperator,
        num_samples: int | None = None,
        resample: bool = False,
        **obs_kwargs,
    ) -> SampledObs:
        """Evaluate an observable using Monte Carlo samples of the current variational state."""
        needs_resample = (
            self.samples is None
            or (num_samples is not None and num_samples != self.numSamples)
            or resample
        )
        if needs_resample:
            self._samples, self._logPsi, self._weights = self.sample(num_samples)
        raw_data = observable.get_O_loc(self.samples, self.psi, logPsiS=self.logPsi, **obs_kwargs)

        return SampledObs(raw_data, self.weights)

    def reset(self, key=None):
        self.key = key

    def sample(self, numSamples=None, parameters=None):
        """Generate cutoff-MCMC samples and the cutoff-corrected weights."""
        if numSamples is not None:
            samples_tmp = self.numSamples
            self.numSamples = numSamples
        if parameters is not None:
            parameters_tmp = self.psi.parameters
            self.psi.parameters = parameters

        configs, logPsi, p = self._get_samples_mcmc()

        if numSamples is not None:
            self.numSamples = samples_tmp
        if parameters is not None:
            self.psi.parameters = parameters_tmp

        self._samples = configs
        self._logPsi = logPsi
        self._weights = p

        return configs, logPsi, p

    # ---- Internal MCMC machinery ----

    def _distribute_sampling(self):
        """Adjust chain and sample counts to match the device mesh (mirror of AbstractMCSampler)."""
        self._numChains = distribute(self.numChains, 'chains')

        self._samplePerChain = (self.numSamples + self.numChains - 1) // self.numChains
        totalSamples = self._samplePerChain * self.numChains

        if totalSamples > self.numSamples:
            print(f"INFO: Total samples adjusted: {self.numSamples} -> {totalSamples}")
        self.numSamples = totalSamples

    def _init_state(self):
        master_key = self.key[0] if self.key.ndim > 1 else self.key
        all_keys = broadcast_split_key(master_key, self.numChains + 1)
        keys = all_keys[:-1]
        initStateKey = all_keys[-1]
        self._key = jax.device_put(keys, DEVICE_SHARDING)

        if self.initial_states is not None:
            self.states = self.initial_states.astype(global_defs.DT_SAMPLES)
            res = self.numChains - self.states.shape[0]
            if res > 0:
                pad = jax.random.bernoulli(
                    initStateKey, 0.5, (res,) + self.sampleShape
                ).astype(global_defs.DT_SAMPLES)
                self.states = jnp.concat([self.states, pad])
        else:
            self.states = jax.random.bernoulli(
                initStateKey, 0.5, (self.numChains,) + self.sampleShape
            ).astype(global_defs.DT_SAMPLES)
        self.states = jax.device_put(self.states, DEVICE_SHARDING)

        self.updateProposer.init_arg(self.psi, self.numChains)
        self._is_state_initialized = True

    def _get_samples_mcmc(self):
        self._distribute_sampling()
        if not self._is_state_initialized:
            self._init_state()
        self.updateProposer.update_arg(self.psi)

        self.logProb = self._log_prob_fun_jsh(self.psi.sampler_parameters, self.states)
        self.logProb = jnp.maximum(self.logProb, self.cutoff)
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
                sweepFunction=self._sweep,
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
                    in_specs=(REPLICATED_SPEC,) + (DEVICE_SPEC,) * 5
                             + (self.updateProposer.arg_in_specs, REPLICATED_SPEC),
                    out_specs=((DEVICE_SPEC,) * 5, DEVICE_SPEC)
                             + (self.updateProposer.arg_in_specs,),
                )
            )

        (self.states, self.logProb, self._key, self.numProposed, self.numAccepted), configs, \
            self.updateProposer._arg = self._get_samples_jsh[numSamplesStr](
                self.psi.sampler_parameters, self.states, self.logProb, self.key,
                self.numProposed, self.numAccepted, self.updateProposer._arg, self.cutoff
            )

        coeffs = self.psi(configs)
        logPMu = self.mu * jnp.real(coeffs)

        # Importance ratio p_target(s) / q(s), evaluated in log space to avoid
        # overflow: ratio = |psi|^mu / max{|psi|^mu, exp(cutoff)} = exp(min(0, logPMu - cutoff)).
        logRatio = jnp.minimum(logPMu - self.cutoff, 0.0)
        # Target distribution is |psi|^(1/logProbFactor), the sampled one carries |psi|^mu.
        logWeights = (1.0 / self.logProbFactor - self.mu) * jnp.real(coeffs) + logRatio
        weights = jnp.exp(logWeights - jnp.max(logWeights))
        weights = weights / jnp.sum(weights)

        self.ratio = jnp.exp(logRatio)
        self.renorm = self.numSamples / jnp.sum(self.ratio)

        self.update_maxLogPsi(jnp.max(logPMu))

        return configs, coeffs, weights

    def _get_samples(self, params, states, logProb, key, numProposed, numAccepted,
                     updateProposerArg, cutoff,
                     numSamples, thermSweeps, sweepSteps, updateProposer,
                     sweepFunction, sampleShape):
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
        return (meta,
                configs.reshape((configs.shape[0] * configs.shape[1],) + sampleShape),
                updateProposerArg)

    def _sweep(self, states, logProb, key, numProposed, numAccepted, cutoff, params,
               numSteps, updateProposer, updateProposerArg):
        def perform_mc_update_single_chain(state, logProb, key_single, ProposerArg):
            proposerKey, newKey = random.split(key_single)
            newState, log_prob_correction = updateProposer(proposerKey, state, ProposerArg)

            # Clip the log-probability from below at the cutoff: the chain samples
            # from q(s) ~ max{|psi(s)|^mu, exp(cutoff)}.
            newLogProb = jnp.maximum(self._log_prob_fun(params, newState), cutoff)
            P = jnp.exp(newLogProb - logProb + log_prob_correction)

            acceptKey, newKey = random.split(newKey)
            accepted = random.bernoulli(acceptKey, P)

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
            sweep_step, (states, logProb, key, numProposed, numAccepted),
            None, length=numSteps
        )
        return states, logProb, key, numProposed, numAccepted

    def acceptance_ratio(self):
        """Get acceptance ratio.

        Returns:
            Acceptance ratio observed in the last call to ``sample()``.
        """
        numProp = jnp.sum(self.numProposed)
        if numProp > 0:
            return jnp.sum(self.numAccepted) / numProp

        return jnp.array([0.])

class ExactSampler(AbstractSampler):
    """
    Class for full enumeration of basis states.

    This class generates a full basis of the many-body Hilbert space. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling.

    Initialization arguments:
        * ``net``: Network defining the probability distribution.
        * ``lDim``: Local Hilbert space dimension.
        * ``logProbFactor``: Factor for the log-probabilities, aquivalent to the exponent for the probability \
        distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs.
    """

    def __init__(self, psi: NQS, lDim=2, logProbFactor=0.5):
        super().__init__(psi)

        self._lDim = lDim
        self._logProbFactor = logProbFactor
        self._lastNorm = 0.

        self.get_probabilities = jax.jit(
            jax.shard_map(
                lambda logPsi, lastNorm : jnp.exp(jnp.real(logPsi - lastNorm) / self.logProbFactor),
                mesh=MESH,
                in_specs=(DEVICE_SPEC, REPLICATED_SPEC),
                out_specs=DEVICE_SPEC
            )
        )

    @property
    def num_sites(self):
        return jnp.prod(jnp.asarray(self.sampleShape))
    
    @property
    def num_states(self):
        return self.lDim ** self.num_sites
    
    @property
    def lDim(self):
        return self._lDim
    
    @property
    def logProbFactor(self):
        return self._logProbFactor
    
    @cached_property
    def basis(self):
        adjusted_dof = distribute(self.num_states)
        int_repr = jax.device_put(jnp.arange(adjusted_dof, dtype=global_defs.DT_SAMPLES), DEVICE_SHARDING)

        def get_basis(int_repr, n_sites):
            def make_state(int_repr, n_sites):
                def scan_fun(c, x):
                    locState = c % self.lDim
                    c = (c - locState) // self.lDim
                    
                    return c, locState
                _, state = jax.lax.scan(scan_fun, int_repr, jnp.arange(n_sites))

                return state[::-1].reshape(self.psi.sampleShape)
            basis = jax.vmap(make_state, in_axes=(0, None))(int_repr, n_sites)

            return basis

        return jax.jit(
            jax.shard_map(partial(get_basis, n_sites=self.num_sites), mesh=MESH, in_specs=DEVICE_SPEC, out_specs=DEVICE_SPEC)
        )(int_repr)[:self.num_states]
    
    def __call__(self, observable: AbstractOperator, **obs_kwargs) -> SampledObs:
        raw_data = observable.get_O_loc(self.samples, self.psi, logPsiS=self.logPsi, **obs_kwargs)

        return SampledObs(raw_data, self.weights)
    
    def sample(self, numSamples=None):
        """
        Return all computational basis states.

        Sampling is automatically distributed accross processes and available devices.

        Arguments:
            * ``numSamples``: Dummy argument to provide identical interface as the ``MCSampler`` class.

        Returns:
            ``configs, logPsi, p``: All computational basis configurations, \
            corresponding wave function coefficients, and probabilities :math:`|\\psi(s)|^2` (normalized).
        """

        logPsi = self.psi(self.basis)
        p = self.get_probabilities(logPsi, self._lastNorm)
        norm = jnp.sum(p)
        p = p / norm
        self._lastNorm += self.logProbFactor * jnp.log(norm)

        self._samples = self.basis
        self._logPsi = logPsi
        self._weights = p

        return self.basis, logPsi, p 
