import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax.random as random 
from functools import partial

from jVMC_exp.geometry import AbstractGeometry
from jVMC_exp.util.key_gen import format_key
from jVMC_exp.sharding_config import DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING
from jVMC_exp.global_defs import DT_SAMPLES_CONT

def shard_array_across_chains(arr, numChains, dtype):
    arr_atleast_1d = jnp.atleast_1d(arr)
    sharded_arr = jnp.stack([arr_atleast_1d] * numChains, axis=0, dtype=dtype)
    return jax.device_put(sharded_arr.reshape((numChains,) + arr_atleast_1d.shape), DEVICE_SHARDING)

class AbstractProposer(ABC):
    def __init__(self, use_custom_thermalization=False):
        self._use_custom_thermalization = use_custom_thermalization
        self._arg = None

    @abstractmethod
    def __call__(self, key, s, arg):
        """
        Proposal move.

        Args:
            key: An instance of ``jax.random.PRNGKey``.
            s: The configuration.
            info: Additional info for the proposal.

        Returns:
            s_prime: The new proposed configuration.
            log_prob_correction: Correction given by non-symmetric proposal moves. 
                                 Return 0 as second arg if no correctionn is needed. 
        """
        pass

    @property
    def arg_in_axes(self):
        """
        Return a dictionary with the same shape as self._arg,
        which indicates which arguments of the proposer must be vmapped across.
        """
        return None

    @property
    def arg_in_specs(self):
        """
        Return a dictionary with the same shape as self._arg,
        which indicates which arguments of the proposer must be sharded across
        different devices.
        """
        return jax.tree_util.tree_map(
            lambda x: DEVICE_SPEC if x is not None else REPLICATED_SPEC, 
            self.arg_in_axes, 
            is_leaf=lambda x: x is None
        ) 

    def init_arg(self, vqs, numChains):
        """
        Initialize all the arguments of the proposer with the right shape 
        for sharding across devices and chains.
        """
        return None

    def update_arg(self, vqs):
        """
        Update the arguments of the proposer that depend on the network.
        """
        pass
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted, 
                          params, sweepSteps, thermSweeps, sweepFunction):
        """
        Optional custom thermalization function.

        Subclasses can override this method if
        `use_custom_thermalization=True`.
        """
        if self._use_custom_thermalization:
            raise NotImplementedError(
                "Custom thermalization was enabled, "
                "but '_custom_therm_fun' is not implemented."
            )
        else:
            raise RuntimeError(
                "Custom thermalization not enabled for this proposer."
            )

class AbstractProposeCont(AbstractProposer):
    def __init__(self, geometry: AbstractGeometry, use_custom_thermalization=False):
        if not isinstance(geometry, AbstractGeometry):
            raise ValueError("The argument 'geometry' has to be an instance of the class 'jVMC.geometry.AbstractGeometry'.")
        self._geometry = geometry
        super().__init__(use_custom_thermalization)

    @property
    def geometry(self):
        return self._geometry

class SpinFlip(AbstractProposer):
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        idx = random.randint(key, (1,), 0, s.size)[0]
        idx = jnp.unravel_index(idx, s.shape)
        update = (s[idx] + 1) % 2

        return s.at[idx].set(update), 0
    
class POVMOutcome(AbstractProposer):
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        key, subkey = random.split(key)
        idx = random.randint(subkey, (1,), 0, s.size)[0]
        idx = jnp.unravel_index(idx, s.shape)
        update = (s[idx] + random.randint(key, (1,), 0, 3) % 4)

        return s.at[idx].set(update), 0

class SpinFlipZ2(AbstractProposer):
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        idxKey, flipKey = jax.random.split(key)
        idx = random.randint(idxKey, (1,), 0, s.size)[0]
        idx = jnp.unravel_index(idx, s.shape)
        update = (s[idx] + 1) % 2
        s = s.at[idx].set(update)
        # On average, do a global spin flip every 30 updates to
        # reflect Z_2 symmetry
        doFlip = random.randint(flipKey, (1,), 0, 5)[0]

        return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s), 0
    
class SpinFlipZeroMag(AbstractProposer):
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        # propose spin flips that stay in the zero magnetization sector
        idxKeyUp, idxKeyDown, flipKey = jax.random.split(key, num=3)

        # can't use jnp.where because then it is not jit-compilable
        # find indices based on cumsum
        bound_up = jax.random.randint(idxKeyUp, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)
        bound_down = jax.random.randint(idxKeyDown, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)

        id_up = jnp.searchsorted(jnp.cumsum(s), bound_up)
        id_down = jnp.searchsorted(jnp.cumsum(1 - s), bound_down)

        idx_up = jnp.unravel_index(id_up, s.shape)
        idx_down = jnp.unravel_index(id_down, s.shape)

        s = s.at[idx_up[0], idx_up[1]].set(0)
        s = s.at[idx_down[0], idx_down[1]].set(1)

        # On average, do a global spin flip every 30 updates to
        # reflect Z_2 symmetry
        doFlip = random.randint(flipKey, (1,), 0, 5)[0]

        return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)

class RWM(AbstractProposeCont):
    def __init__(self, geometry, sigma=None, use_thermalization=True, adapt_rate=0.1, target_rate=0.5):
        super().__init__(geometry, use_thermalization)
        self._adapt_rate = adapt_rate
        self._target_rate = target_rate
        self._sigma = sigma

    @property
    def adapt_rate(self):
        return self._adapt_rate
    
    @property
    def target_rate(self):
        return self._target_rate
            
    @property
    def arg_in_axes(self):
        return {"sigma": 0}
    
    def init_arg(self, vqs, numChains):
        if self._sigma is None:
            sigma = jnp.array(self.geometry.extent) * 0.1
        else: 
            sigma = jnp.asarray(self._sigma) if not isinstance(self._sigma, jax.Array) else self._sigma
            if sigma.size == 1:
                sigma = jnp.full(self.geometry.n_dim, sigma.item())
            elif sigma.size != self.geometry.n_dim:
                raise ValueError(
                    f"'sigma' must have the same dimension as 'geometry.n_dim' "
                    f"(expected {self.geometry.n_dim}, got {sigma.size})."
                )
        sigma = shard_array_across_chains(jnp.asarray(sigma), numChains, dtype=DT_SAMPLES_CONT)

        self._arg = {"sigma": sigma}
    
    def __call__(self, key, s, arg):
        """
        Proposal move for random walk metropolis (RWM).
        
        Args:
            key: An instance of ``jax.random.PRNGKey``.
            s: The configuration.

        Return: 
            s': The new proposed configuration. 
        """
        sigma = jnp.tile(arg["sigma"], self.geometry.n_particles)
        dx = (sigma * jax.random.normal(format_key(key), s.shape, dtype=s.dtype))   
        return self.geometry.apply_PBC(s + dx), 0
    
    def _get_new_sigma(self, sigma, acceptance_rate):
        new_sigma = sigma * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
        return jnp.clip(new_sigma, max=jnp.array(self.geometry.extent) / 2)
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, arg):
        def therm_sweep_fun(carry, i):
            states, logAccProb, key, numProposed, numAccepted, arg = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(
                states, logAccProb, key, numProposed, numAccepted, params, sweepSteps, self, arg
            )
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(sweepSteps, 1)
            # Vectorize across chains
            new_sigma = jax.vmap(self._get_new_sigma)(arg["sigma"], acceptance_rate)
            arg = {**arg, "sigma": new_sigma}

            new_carry = (states, logAccProb, key, numProposed, numAccepted, arg)
            return new_carry, None 

        carry = (states, logAccProb, key, numProposed, numAccepted, arg)
        final_carry, _ = jax.lax.scan(therm_sweep_fun, carry, jnp.arange(thermSweeps))
        return final_carry
    
class MALA(AbstractProposeCont):
    def __init__(self, geometry, appy_fun, tau=None, use_thermalization=True, adapt_rate=0.1, target_rate=0.5, mu=2):
        super().__init__(geometry, use_thermalization)
        self._adapt_rate = adapt_rate
        self._target_rate = target_rate
        self._apply_fun = appy_fun
        self._mu = mu
        self._tau = tau
        
    @property
    def apply_fun(self):
        return self._apply_fun

    @property
    def mu(self):
        return self._mu

    @property
    def adapt_rate(self):
        return self._adapt_rate
    
    @property
    def target_rate(self):
        return self._target_rate

    @property
    def arg_in_axes(self):
        return {"params": None, "tau": 0}
    
    def init_arg(self, vqs, numChains):
        # TODO: implement version for tau in more dims
        if self._tau is None:
            tau = jnp.array(self.geometry.extent)[0] * 0.01
        elif self._tau < 0:
            raise ValueError('"tau" can not be negative.')
        else:
            tau = self._tau
        tau = shard_array_across_chains(jnp.asarray(tau), numChains, dtype=DT_SAMPLES_CONT)

        self._arg = {"params": vqs.parameters, "tau": tau}
    
    def update_arg(self, vqs):
        if self._arg is None:
            raise RuntimeError("Must call init_arg before update_arg.")
        self._arg["params"] = vqs.parameters
    
    def __call__(self, key, s, arg):
        tau = arg["tau"]
        params = arg["params"]
        log_prob_fun = partial(lambda x, p: self.mu * jnp.real(self.apply_fun(p, x)), p=params)
        log_prob_fun_grad = jax.grad(log_prob_fun)

        xi = jax.random.normal(format_key(key), s.shape, dtype=s.dtype)
        drift = tau * log_prob_fun_grad(s)
        dx = drift + jnp.sqrt(2 * tau).astype(s.dtype) * xi
        sp = s + dx

        log_q_s_sp = - jnp.sum((s - sp - tau * log_prob_fun_grad(sp)) ** 2) / (4 * tau)
        log_q_sp_s = - jnp.sum(xi ** 2) * 0.5
        log_prob_correction = log_q_s_sp - log_q_sp_s

        return self.geometry.apply_PBC(sp), log_prob_correction.squeeze()
    
    def _get_new_tau(self, tau, acceptance_rate):
        new_tau = tau * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
        return jnp.clip(new_tau, max=jnp.array(self.geometry.extent)[0] / 2) # TODO: change when tau gets more dims
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, arg):
        def therm_sweep_fun(carry, i):
            states, logAccProb, key, numProposed, numAccepted, arg = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(
                states, logAccProb, key, numProposed, numAccepted, params, sweepSteps, self, arg
            )
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(sweepSteps, 1)
            # Vectorize across chains
            new_tau = jax.vmap(self._get_new_tau)(arg["tau"], acceptance_rate)
            arg = {**arg, "tau": new_tau}

            new_carry = (states, logAccProb, key, numProposed, numAccepted, arg)
            return new_carry, None 

        carry = (states, logAccProb, key, numProposed, numAccepted, arg)
        final_carry, _ = jax.lax.scan(therm_sweep_fun, carry, jnp.arange(thermSweeps))
        return final_carry