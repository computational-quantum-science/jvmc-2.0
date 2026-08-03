import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import flax
from flax.core.frozen_dict import freeze
import flax.linen as nn
import collections
from typing import Tuple
from functools import reduce
import copy
import flax.linen as nn
from flax import nnx

from jVMC_exp.nets.two_nets_wrapper import TwoNets
from jVMC_exp.symmetry import SymmetryProjector, ProjectedOrbitNet
from jVMC_exp.util.grads import pick_gradient
from jVMC_exp.util.key_gen import generate_seed, format_key
from jVMC_exp.util.util import has_callable_attr
from jVMC_exp import global_defs
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC, DEVICE_SHARDING, REPLICATED_SHARDING
from jVMC_exp.sharding_config import broadcast_split_key, sharded

def _check_dtype(dtype: jnp.dtype | None, dtype_ref: jnp.dtype, label: str):
    if dtype is None:
        return dtype_ref
    ref_is_real = jnp.issubdtype(dtype_ref, jnp.floating)
    given_is_real = jnp.issubdtype(dtype, jnp.floating) 
    if ref_is_real and not given_is_real:
        raise TypeError(
            f"The given dtype for {label} is complex but the parameters are real"
        )
    if not ref_is_real and given_is_real:
        raise TypeError(
            f"The given dtype for {label} is real but the parameters are complex"
        )
    
    return dtype

def _cast_pytree(pytree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), pytree)

def _check_model(model, nnx_init_kwargs=None):
    if isinstance(model, nn.Module):
        return model
    
    if isinstance(model, nnx.Module):
        raise ValueError(
            "Pass the NNX class, not an instance: "
            f"use {type(model).__name__} instead of {type(model).__name__}(...)"
        )
    
    if isinstance(model, type) and issubclass(model, nnx.Module):
        kwargs = nnx_init_kwargs or {}
        return nnx.bridge.to_linen(model, **kwargs)
    
    raise ValueError(f"Expected a flax.linen.Module instance or a flax.nnx.Module class, got {type(model)}")
        
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
        * ``net``: Variational network, tuple of networks, or ``flax.nnx.Module`` subclass. \
            A network has to be registered as pytree node and provide \
            a ``__call__`` function for evaluation. \
            If a tuple of two networks is given, the first is used for the logarithmic \
            amplitude and the second for the phase of the wave function coefficient. \
            If a ``flax.nnx.Module`` subclass is given, it will be automatically wrapped \
            into a ``flax.linen``-compatible module via ``flax.nnx.bridge.to_linen``. \
            In this case, ``nnx_init`` must also be provided.
        * ``logarithmic``: Boolean variable indicating, whether the ANN returns logarithmic \
            (:math:`\\log\\psi_\\theta(s)`) or plain (:math:`\\psi_\\theta(s)`) wave function coefficients.
        * ``batchSize``: Batch size for batched network evaluation. Choice \
            of this parameter impacts performance: with too small values performance \
            is limited by memory access overheads, too large values can lead \
            to "out of memory" issues.
        * ``seed``: Seed for the PRNG to initialize the network parameters.
        * ``orbit``: Symmetry projector defining the symmetry operations (instance of ``jVMC_exp.symmetry.SymmetryProjector``). \
            If this argument is given, the wave function is symmetrized to be invariant under symmetry operations.
        * ``symmetry_average``: Built-in symmetry average name or callable passed to ``ProjectedOrbitNet``.
        * ``mixed_precision``: If ``True``, low-precision parameter storage is allowed while public \
            amplitudes, ratios, flattened parameters, and gradients are cast to operator precision. \
            If ``False``, fp32/complex64 parameter storage raises an error.
        * ``nnx_init``: Dictionary of keyword arguments passed to the ``flax.nnx.Module`` constructor, \
            excluding ``rngs`` (which is handled internally). Required when ``net`` is a \
            ``flax.nnx.Module`` subclass or a tuple thereof; ignored otherwise. \
            If ``net`` is a tuple of two ``flax.nnx.Module`` subclasses, ``nnx_init`` must be \
            a tuple of two dictionaries, one per network.

    Example:
        Using a ``flax.linen`` model (unchanged behavior)::

            psi = NQS(RBMLinenModel(numHidden=4), sampleShape, batchSize=32, seed=0)

        Using a ``flax.nnx`` model::

            psi = NQS(RBMNNXModel, sampleShape, batchSize=32, seed=0,
                    nnx_init=dict(in_features=10, numHidden=4, bias=True))

        Using two ``flax.nnx`` models::

            psi = NQS((AmplitudeNNX, PhaseNNX), sampleShape, batchSize=32, seed=0,
                    nnx_init=(dict(in_features=10, numHidden=4),
                                dict(in_features=10, numHidden=4)))
    """
    def __init__(
            self, net: nn.Module | Tuple[nn.Module, nn.Module], 
            sampleShape: int | tuple, 
            batchSize: int | None = None, batchSize_per_device: int | None = None, 
            logarithmic=True, 
            seed: None | int = None, 
            orbit=None, symmetry_average="exp",
            sampler_dtype: jnp.dtype |None = None, 
            eval_dtype: jnp.dtype |None = None,
            grad_dtype: jnp.dtype |None = None,
            nnx_init=None
        ):
        if isinstance(net, collections.abc.Iterable):
            if len(net) != 2:
                raise ValueError(f"If a tuple is passed for 'net', this must have len 2. Got {len(net)}.") 
            if nnx_init is not None:
                if not isinstance(nnx_init, collections.abc.Iterable):
                    raise ValueError(f"If a tuple is passed for 'net', nnx_init can be either None or a tuple of len 2.")
                if len(nnx_init) != 2:
                    raise ValueError("If a tuple is passed for 'net', and nnx_init is not None, nnx_init must be a tuple of len 2."
                                     f"Got {len(nnx_init)}")
            else:
                nnx_init = (None, None)
            net = tuple(_check_model(n, i) for n, i in zip(net, nnx_init))
            net = TwoNets(net)
        else:
            net = _check_model(net, nnx_init)
            
        if orbit is not None:
            if not isinstance(orbit, SymmetryProjector):
                raise TypeError(
                    f"Orbit has to be an instance of jVMC_exp.symmetry.SymmetryProjector, "
                    f"got {orbit}"
                )
            net = ProjectedOrbitNet(base_net=net, symmetry=orbit, symmetry_average=symmetry_average)
        self._net = net

        if isinstance(self._net, ProjectedOrbitNet):
            self._is_generator = has_callable_attr(net.base_net, "sample")
        else:
            self._is_generator = has_callable_attr(net, "sample")
        
        self._eval_ratio = callable(getattr(net, "eval_ratio", None))

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

        if logarithmic:
            self.apply_fun = self.net.apply
        else:
            def apply_fun(parameters, s, method=None):
                return jnp.log(self.net.apply(parameters, s, method))
            self.apply_fun = apply_fun
        
        self.init_net(seed)

        self._sampler_dtype = _check_dtype(sampler_dtype, self.param_dtype, "sampler")
        self._eval_dtype = _check_dtype(eval_dtype, self.param_dtype, "eval")
        self._grad_dtype = _check_dtype(grad_dtype, self.param_dtype, "grad") 

        self._append_gradients_dict_single = lambda x, y: tree_map(
            lambda a, b: jnp.concatenate((a, 1.j * b)), x, y
        )
        self._append_gradients_dict = lambda x, y: tree_map(
            lambda a, b: jnp.concatenate((a[:, :], 1.j * b[:, :]), axis=1), x, y
        )
        self._append_gradients_dict_jsh = jax.jit(
            jax.shard_map(
                self._append_gradients_dict, 
                mesh=MESH, 
                in_specs=(DEVICE_SPEC, DEVICE_SPEC), 
                out_specs=DEVICE_SPEC
            )
        )

        self._frozen_params = None

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
        if jax.tree_util.tree_structure(value["params"]) != self._netTreeDef:
            raise ValueError(
                "Parameter tree structure mismatch.\n"
                f"  Expected: {self._netTreeDef}\n"
                f"  Received: {jax.tree_util.tree_structure(value['params'])}"
            )

        if self._frozen_params is not None:
            for frozen_path in self._frozen_params:
                original = reduce(lambda d, key: d[key], frozen_path, self.params)
                reduce(lambda d, k: d[k], frozen_path[:-1], value['params'])[frozen_path[-1]] = original
        if isinstance(self.parameters, flax.core.frozen_dict.FrozenDict):
            value = freeze(value)

        value = jax.tree_util.tree_map(lambda x: x.astype(self.param_dtype), value)
        self._parameters = copy.deepcopy(value)

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
        if not self.realParams:
            flat = jnp.concatenate([
                jnp.concatenate([p.ravel().real,p.ravel().imag,]) for p in tree_flatten(self.params)[0]
            ])
            return flat.astype(global_defs.DT_OUT_REAL)
        
        return jnp.concatenate([p.ravel() for p in tree_flatten(self.params)[0]]).astype(global_defs.DT_OUT_REAL)
    
    @property
    def frozen_parameters(self):
        return self._frozen_params

    @frozen_parameters.setter
    def frozen_parameters(self, labels: list[str] | None | str):
        if labels is not None:
            if isinstance(labels, str):
                labels = (labels,)
            try:
                reduce(lambda d, key: d[key], labels, self.parameters['params'])
            except KeyError as e:
                raise ValueError(f'The given label {e} does not exist in parameters')
            
            if self._frozen_params is None:
                self._frozen_params = []
            if labels not in self._frozen_params:
                self._frozen_params.append(labels)
        else:
            self._frozen_params = None

    @property
    def sampler_parameters(self):
        return _cast_pytree(self.parameters, self.sampler_dtype)
    
    @property
    def eval_parameters(self):
        return _cast_pytree(self.parameters, self.eval_dtype)
    
    @property
    def grad_parameters(self):
        return _cast_pytree(self.parameters, self.grad_dtype)
    
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
    def is_generator(self):
        return self._is_generator

    @property
    def eval_ratio(self):
        return self._eval_ratio
    
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
    
    @property
    def param_dtype(self):
        return self._param_dtype
    
    @property
    def sampler_dtype(self):
        return self._sampler_dtype
    
    @property
    def eval_dtype(self):
        return self._eval_dtype
    
    @property
    def grad_dtype(self):
        return self._grad_dtype
    
    @property
    def out_dtype(self):
        return self._out_dtype
        
    def init_net(self, seed: int | None):
        if seed == None:
            seed = generate_seed()

        # TODO: this is fragile and is only needed for holo check. 
        # Holo should be given by the user at init to make it more genera
        # Otherwise another optional function, which would take an input state as well, could be used to check holo
        dummy_sample = jax.random.normal(jax.random.PRNGKey(seed), self.sampleShape)
        self._parameters = jax.device_put(self.net.init(jax.random.PRNGKey(seed), dummy_sample), REPLICATED_SHARDING)
        self._netTreeDef = jax.tree_util.tree_structure(self.params)
        out = self.apply_fun(self.parameters, dummy_sample)
        is_complex = jnp.issubdtype(out.dtype, jnp.complexfloating)
        self._out_dtype = global_defs.DT_OUT_CPX if is_complex else global_defs.DT_OUT_REAL

        self._realParams, self._holomorphic, self._flat_gradient_function, self._dict_gradient_function = pick_gradient(
            self.apply_fun, self.parameters, dummy_sample
        )

        self._paramShapes = []
        self._numParameters = 0
        for p in tree_flatten(self.params)[0]:
            self._paramShapes.append((p.size, p.shape))
            self._numParameters += p.size 
        self._param_dtype = p.dtype

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
        return self._apply_fun_sh(
            s, parameters=self.eval_parameters, batch_size=self.batchSize
        ).astype(self.out_dtype)
    
    @sharded()
    def _apply_fun_sh(self, s, *, parameters, batch_size):
        return self.apply_fun(parameters, s)

    def call_ratio(self, s, sp):
        if self.eval_ratio:
            return self._apply_ratio_sh(
                s, sp, parameters=self.eval_parameters, batch_size=self.batchSize
            ).astype(self.out_dtype)

        return jnp.exp(self(sp) - self(s))
    
    @sharded()
    def _apply_ratio_sh(self, s, sp, *, parameters, batch_size):
        return self.apply_fun(parameters, s, sp, method=self.net.eval_ratio)

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
        return self._gradients_sh(
            s, parameters=self.grad_parameters, batch_size=self.batchSize
        ).astype(self.out_dtype)
    
    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _gradients_sh(self, s, *, parameters, batch_size):
        return self.flat_gradient_function(self.apply_fun, parameters, s)
    
    def lazy_gradients(self, s):
        return self._lazy_gradients_sh(s, parameters=self.grad_parameters, batch_size=self.batchSize)
    
    @sharded(automatic_sharding=True, yield_iter=True)
    def _lazy_gradients_sh(self, s, *, parameters, batch_size):
        return self.flat_gradient_function(self.apply_fun, parameters, s)
    
    def gradients_dict(self, s):
        result = self._gradients_dict_sh(s, parameters=self.grad_parameters, batch_size=self.batchSize)
        if self.holomorphic:
            result = self._append_gradients_dict_jsh(result, result)

        return _cast_pytree(result, self.out_dtype)
    
    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _gradients_dict_sh(self, s, *, parameters, batch_size):
        return self.dict_gradient_function(self.apply_fun, parameters, s)
    
    def sample(self, numSamples, key=None):
        if self.is_generator:
            key = format_key(key) 
            if len(key.shape) > 1:
                key = key[0]
            keys = jax.device_put(broadcast_split_key(key, numSamples), DEVICE_SHARDING)

            return self._sample(keys, parameters=self.sampler_parameters, batch_size=self.batchSize)

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