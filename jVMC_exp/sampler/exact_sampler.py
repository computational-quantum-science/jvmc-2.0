import jax
import jax.numpy as jnp
from functools import partial, cached_property

from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import (
    MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING, 
    distribute
)
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.stats import SampledObs
from jVMC_exp import global_defs
from jVMC_exp.sampler.base import AbstractSampler

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
        self.numSamples = self.num_states

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