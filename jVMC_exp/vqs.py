import jax
from jax import grad 
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import flax
from flax.core.frozen_dict import freeze
import numpy as np
import flax.linen as nn
import collections
from math import isclose
from typing import Tuple

from jVMC_exp.nets.sym_wrapper import avgFun_Coefficients_Exp, SymNet
from jVMC_exp.nets.two_nets_wrapper import TwoNets
from jVMC_exp.util.key_gen import generate_seed, format_key
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC, DEVICE_SHARDING, REPLICATED_SHARDING, REPLICATED_SPEC
from jVMC_exp.sharding_config import distribute, broadcast_split_key, sharded

def flat_gradient_real(fun, params, arg):
    """
    Grad for fun: Real -> Real
    """
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]

    return jnp.concatenate(g)

def flat_gradient(fun, params, arg):
    """
    Grad for fun: Real -> Complex 
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_nonholo(fun, params, arg):
    """
    Grad for fun: Complex -> Complex (not holomorphic) 
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_holo(fun, params, arg):
    """
    Grad for fun: Complex -> Complex (holomorphic) 
    """
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j * x.ravel()], g))[0]

    return jnp.concatenate(g)

def dict_gradient(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_map(lambda x: x.ravel(), gr)
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_map(lambda x: x.ravel(), gi)

    return tree_map(lambda x,y: x + 1.j*y, gr, gi)

def dict_gradient_real(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_map(lambda x: x.ravel(), g)

    return g
        
class NQS:
    """
        Initializes NQS class.
        
        This class can operate in two modi:
            #. Single-network ansatz
                Quantum state of the form :math:`\\psi_\\theta(s)\\equiv\\exp(r_\\theta(s))`, \
                where the network :math:`r_\\theta` is
                a) holomorphic, i.e., parametrized by complex valued parameters :math:`\\vartheta`.
                b) non-holomorphic, i.e., parametrized by real valued parameters :math:`\\theta`.
            #. Two-network ansatz
                Quantum state of the form 
                :math:`\\psi_\\theta(s)\\equiv\\exp(r_{\\theta_r}(s)+i\\varphi_{\\theta_\\phi}(s))` \
                with an amplitude network :math:`r_{\\theta_{r}}` and a phase network \
                :math:`\\varphi_{\\theta_\\phi}` \
                parametrized by real valued parameters :math:`\\theta_r,\\theta_\\phi`.
        Args:       
            * ``net``: Variational network or tuple of networks.
                A network has to be registered as pytree node and provide \
                a ``__call__`` function for evaluation. \
                If a tuple of two networks is given, the first is used for the logarithmic \
                amplitude and the second for the phase of the wave function coefficient.
            * ``logarithmic``: Boolean variable indicating, whether the ANN returns logarithmic \
                (:math:`\\log\\psi_\\theta(s)`) or plain (:math:`\\psi_\\theta(s)`) wave function coefficients.
            * ``batchSize``: Batch size for batched network evaluation. Choice \
                of this parameter impacts performance: with too small values performance \
                is limited by memory access overheads, too large values can lead \
                to "out of memory" issues.
            * ``seed``: Seed for the PRNG to initialize the network parameters.
            * ``orbit``: Orbit which defining the symmetry operations (instance of ``util.symmetries.LatticeSymmetry``). \
                If this argument is given, the wave function is symmetrized to be invariant under symmetry operations.
            * ``avgFun``: Reduction operation for the symmetrization.
        """

    def __init__(self, net: nn.Module | Tuple[nn.Module, nn.Module], sampleShape, batchSize: int | None = None, batchSize_per_device: int | None = None, 
                 logarithmic=True, seed: None | int = None, orbit=None, avgFun=avgFun_Coefficients_Exp):
        if isinstance(net, collections.abc.Iterable):
            if len(net) != 2:
                raise ValueError(f"If a tuple is passed for 'net', this must have len 2. Got {len(net)}.") 
            if not isinstance(net[0], nn.Module) or not isinstance(net[1], nn.Module):
                raise ValueError("The argument 'net' has to be an instance of flax.nn.Module.")
            net = TwoNets(net)
        else:
            if not isinstance(net, nn.Module):
                raise ValueError("The argument 'net' has to be an instance of flax.nn.Module.")
            
        self._isGenerator = False
        if "sample" in dir(net):
            if callable(net.sample):
                self._isGenerator = True
        if orbit is not None:
            net = SymNet(net=net, orbit=orbit, avgFun=avgFun)
        self._net = net

        if isinstance(sampleShape, tuple):
            self._sampleShape = sampleShape
        else:
            self._sampleShape = (sampleShape,)
        
        num_devices = MESH.size
        if (batchSize is None) == (batchSize_per_device is None):
            raise ValueError("Exactly one of 'batchSize' or 'batchSize_per_device' must be specified")
        if batchSize is None:
            batchSize = batchSize_per_device * num_devices
        elif batchSize % num_devices != 0:
            raise ValueError(f"The batch size ({batchSize}) has to be divisible by the number of devices ({num_devices})")
        self._batchSize = batchSize

        self._logarithmic = logarithmic
        
        self.init_net(seed)
        if self.logarithmic:
            self.apply_fun = self.net.apply
        else:
            def apply_fun(parameters, s, method=None):
                return jnp.log(self.net.apply(parameters, s, method))
            self.apply_fun = apply_fun

        self._append_gradients_dict_single = lambda x, y: tree_map(lambda a, b: jnp.concatenate((a, 1.j * b)), x, y)
        self._append_gradients_dict = lambda x, y: tree_map(lambda a, b: jnp.concatenate((a[:, :], 1.j * b[:, :]), axis=1), x, y)
        self._append_gradients_dict_jsh = jax.jit(
            jax.shard_map(
                self._append_gradients_dict, 
                mesh=MESH, 
                in_specs=(DEVICE_SPEC, DEVICE_SPEC), 
                out_specs=DEVICE_SPEC
            )
        )

    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        if hasattr(value, "shape"):
            if len(value) != len(self.parameters_flat):
                raise ValueError(
                    f"The given number of parameters ({len(value)}) "
                    f"does not match the existing one ({len(self.parameters_flat)})"
                )
            value = self._param_unflatten(value)
        if 'params' not in value.keys():
            value = {'params': value}
        if isinstance(self.parameters, flax.core.frozen_dict.FrozenDict):
            value = freeze(value)
        if jax.tree_util.tree_structure(value) != jax.tree_util.tree_structure(self.parameters):
            raise ValueError(
                "Parameter tree structure mismatch.\n"
                f"  Expected: {jax.tree_util.tree_structure(self.parameters)}\n"
                f"  Received: {jax.tree_util.tree_structure(value)}"
            )

        self._parameters = value

    @property
    def params(self):
        return self.parameters['params']
    
    @params.setter
    def params(self, value):
        self.parameters = value

    @property
    def parameters_flat(self):
        """
        Get variational parameters.
        
        Returns:
            Array holding current values of all variational parameters.
        """
        if not self._realParams:
            return jnp.concatenate([jnp.concatenate([p.ravel().real, p.ravel().imag]) for p in tree_flatten(self.params)[0]])
        return jnp.concatenate([p.ravel() for p in tree_flatten(self.params)[0]])

    @property
    def batchSize(self):
        return self._batchSize
        
    @property
    def net(self):
        return self._net
    
    @property
    def sampleShape(self):
        return self._sampleShape
    
    @property
    def logarithmic(self):
        return self._logarithmic

    @property
    def is_generator(self):
        return self._isGenerator
    
    @property
    def holomorphic(self):
        return self._holomorphic
    
    @property
    def realParams(self):
        return self._realParams
    
    @property
    def flat_gradient_function(self):
        return self._flat_gradient_function
    
    @property
    def dict_gradient_function(self):
        return self._dict_gradient_function
    
    @property
    def paramShapes(self):
        return self._paramShapes
    
    @property
    def numParameters(self):
        return self._numParameters
        
    def init_net(self, seed: int | None):
        dummy_sample = jnp.ones(self.sampleShape)
        if seed == None:
            seed = generate_seed()
        self._parameters = jax.device_put(self.net.init(jax.random.PRNGKey(seed), dummy_sample), REPLICATED_SHARDING)
        self._out_dtype = self.net.apply(self._parameters, dummy_sample).dtype
        self._realParams = False
        self._holomorphic = False
        
        dtypes = [a.dtype for a in tree_flatten(self.parameters)[0]]
        if not all(d == dtypes[0] for d in dtypes):
            raise Exception("Network uses different parameter data types. This is not supported.")
        if dtypes[0] == np.single or dtypes[0] == np.double:
            self._realParams = True
        self._param_dtype = dtypes[0]

        # check Cauchy-Riemann condition to test for holomorphicity
        def make_flat(t):
            return jnp.concatenate([p.ravel() for p in tree_flatten(t)[0]])
        
        grads_r = make_flat(jax.grad(lambda a, b: jnp.real(self.net.apply(a,b)))(self.parameters, dummy_sample)["params"])
        grads_i = make_flat(jax.grad(lambda a, b: jnp.imag(self.net.apply(a,b)))(self.parameters, dummy_sample)["params"])
        if isclose(jnp.linalg.norm(grads_r - 1.j * grads_i) / grads_r.shape[0], 0.0, abs_tol=1e-14):
            self._holomorphic = True
            self._flat_gradient_function = flat_gradient_holo
            self._dict_gradient_function = dict_gradient_real
        else:
            if self._realParams:
                self._flat_gradient_function = flat_gradient
                self._dict_gradient_function = dict_gradient
            else:
                self._flat_gradient_function = flat_gradient_nonholo
                self._dict_gradient_function = dict_gradient_real

        self._paramShapes = [(p.size, p.shape) for p in tree_flatten(self.parameters["params"])[0]]
        self._netTreeDef = jax.tree_util.tree_structure(self.parameters["params"])
        self._numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters["params"])[0]]))
    
    def __call__(self, s):
        """
        Evaluate variational wave function.
        
        Compute the logarithmic wave function coefficients :math:`\\ln\\psi(s)` for \
        computational configurations :math:`s`.
        
        Args:
            * ``s``: Array of computational basis states.
        Returns:
            Logarithmic wave function coefficients :math:`\\ln\\psi(s)`.
        
        :meta public:
        """ 
        return self._apply_fun_sh(s, parameters=self.parameters, batch_size=self.batchSize)
    
    @sharded()
    def _apply_fun_sh(self, s, *, parameters, batch_size):
        return self.apply_fun(parameters, s)

    def gradients(self, s):
        """
        Compute gradients of logarithmic wave function.
        
        Compute gradient of the logarithmic wave function coefficients, \
        :math:`\\nabla\\ln\\psi(s)`, for computational configurations :math:`s`.
        
        Args:
            * ``s``: Array of computational basis states.
        Returns:
            A vector containing derivatives :math:`\\partial_{\\theta_k}\\ln\\psi(s)` \
            with respect to each variational parameter :math:`\\theta_k` for each \
            input configuration :math:`s`.
        """
        return self._gradients_sh(s, parameters=self.parameters, batch_size=self.batchSize)
    
    @sharded()
    def _gradients_sh(self, s, *, parameters, batch_size):
        return self.flat_gradient_function(self.apply_fun, parameters, s)
    
    def gradients_dict(self, s):
        result = self._gradients_dict_sh(s, parameters=self.parameters, batch_size=self.batchSize)

        if self.holomorphic:
            return self._append_gradients_dict_jsh(result, result)
        return result
    
    @sharded()
    def _gradients_dict_sh(self, s, *, parameters, batch_size):
        return self.dict_gradient_function(self.apply_fun, parameters, s)

    def grad_dict_to_vec_map(self):
        PTreeShape = []
        start = 0
        P = jnp.arange(2 * self.numParameters)
        for s in self.paramShapes:
            # Here we need to add the treatment for the complex non-holomorphic case
            if self.holomorphic:
                PTreeShape.append((P[start:start + 2 * s[0]]))
                start += 2 * s[0]
            else:
                PTreeShape.append(P[start:start + s[0]])
                start += s[0]
        
        return tree_unflatten(self._netTreeDef, PTreeShape)

    def get_sampler_net(self):
        """
        Get real part of NQS and current parameters

        This function returns a function that evaluates the real part of the NQS,
        :math:`\\text{Re}(\\log\\psi(s))`, and the current parameters.

        Returns:
            Real part of the NQS and current parameters
        """
        if "eval_real" in dir(self.net):
            if callable(self.net.eval_real):
                return lambda p, x: jnp.real(self.apply_fun(p, x, method=self.net.eval_real)), self.parameters
        else:
            return lambda p, x: jnp.real(self.apply_fun(p, x)), self.parameters
    
    def sample(self, numSamples, key=None, parameters=None):
        if self._isGenerator:
            params = parameters or self.parameters
            key = format_key(key) 
            if len(key.shape) > 1:
                key = key[0]
            keys = jax.device_put(broadcast_split_key(key, numSamples), DEVICE_SHARDING)

            return self._sample(keys, parameters=params, batch_size=self.batchSize)

        return None
    
    @sharded()
    def _sample(self, keys, *, parameters, batch_size):
        return self.net.apply(parameters, keys, method=self.net.sample)
 
    def _param_unflatten(self, P):
        """
        Reshape parameter array update according to net tree structure
        """
        if isinstance(P, dict):
            if 'params' in P.keys():
                return P['params']
            else:
                return P
        PTreeShape = []
        start = 0
        for s in self.paramShapes:
            if not self._realParams:
                PTreeShape.append((P[start:start + s[0]] + 1.j * P[start + s[0]:start + 2 * s[0]]).reshape(s[1]))
                start += 2 * s[0]
            else:
                PTreeShape.append(P[start:start + s[0]].reshape(s[1]))
                start += s[0]
        
        return tree_unflatten(self._netTreeDef, PTreeShape)