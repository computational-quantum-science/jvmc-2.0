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
import warnings

from jVMC_exp.nets.two_nets_wrapper import TwoNets
from jVMC_exp.symmetry_projector import SymmetryProjector, ProjectedOrbitNet
from jVMC_exp.util.grads import pick_gradient
from jVMC_exp.util.key_gen import generate_seed, format_key
import jVMC_exp.global_defs as global_defs
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC, DEVICE_SHARDING, REPLICATED_SHARDING
from jVMC_exp.sharding_config import broadcast_split_key, sharded


def _is_complex_dtype(dtype):
    return jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)

def _operator_dtype_for(dtype):
    return global_defs.DT_OPERATORS_CPX if _is_complex_dtype(dtype) else global_defs.DT_OPERATORS_REAL

def _is_low_precision_parameter_dtype(dtype):
    dtype = jnp.dtype(dtype)
    return dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64))

def check_model(model, nnx_init_kwargs=None):
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
        * ``orbit``: Symmetry projector defining the symmetry operations (instance of ``symmetry_projector.SymmetryProjector``). \
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
    def __init__(self, net: nn.Module | Tuple[nn.Module, nn.Module], sampleShape, batchSize: int | None = None, batchSize_per_device: int | None = None, 
                 logarithmic=True, seed: None | int = None, orbit=None, symmetry_average="exp", nnx_init=None, mixed_precision: bool = False):
        self._mixed_precision = bool(mixed_precision)
        self._warned_param_casts = set()
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
            net = tuple(check_model(n, i) for n, i in zip(net, nnx_init))
            net = TwoNets(net)
        else:
            net = check_model(net, nnx_init)
            
        if orbit is not None:
            if not isinstance(orbit, SymmetryProjector):
                raise TypeError(
                    "To symmetrize the NQS pass a jVMC_exp.symmetry_projector.SymmetryProjector."
                )
            net = ProjectedOrbitNet(base_net=net, symmetry=orbit, symmetry_average=symmetry_average)
        self._net = net
        if isinstance(net, ProjectedOrbitNet):
            self._isGenerator = callable(getattr(net.base_net, "sample", None))
        else:
            self._isGenerator = callable(getattr(net, "sample", None))
        
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

        self._logarithmic = logarithmic
        if self.logarithmic:
            self.apply_fun = self.net.apply
        else:
            def apply_fun(parameters, s, method=None):
                return jnp.log(self.net.apply(parameters, s, method))
            self.apply_fun = apply_fun
        self.init_net(seed)

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
        if hasattr(self, "paramDtypes"):
            value = self._cast_param_tree(value)
        if jax.tree_util.tree_structure(value) != jax.tree_util.tree_structure(self.parameters):
            raise ValueError(
                "Parameter tree structure mismatch.\n"
                f"  Expected: {jax.tree_util.tree_structure(self.parameters)}\n"
                f"  Received: {jax.tree_util.tree_structure(value)}"
            )

        if self._frozen_params is not None:
            for frozen_path in self._frozen_params:
                original = reduce(lambda d, key: d[key], frozen_path, self.params)
                reduce(lambda d, k: d[k], frozen_path[:-1], value['params'])[frozen_path[-1]] = original
        if isinstance(self.parameters, flax.core.frozen_dict.FrozenDict):
            value = freeze(value)

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
        if not self._realParams:
            return jnp.concatenate([
                jnp.concatenate([
                    p.ravel().real.astype(global_defs.DT_OPERATORS_REAL),
                    p.ravel().imag.astype(global_defs.DT_OPERATORS_REAL),
                ])
                for p in tree_flatten(self.params)[0]
            ])
        return jnp.concatenate([p.ravel().astype(global_defs.DT_OPERATORS_REAL) for p in tree_flatten(self.params)[0]])
    
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
    def eval_ratio(self):
        return self._eval_ratio

    @property
    def mixed_precision(self):
        return self._mixed_precision
    
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
        self.paramDtypes = [p.dtype for p in tree_flatten(self.parameters["params"])[0]]
        self._validate_parameter_dtypes()
        self._out_dtype = self.net.apply(self._parameters, dummy_sample).dtype
        
        self._realParams, self._holomorphic, self._flat_gradient_function, self._dict_gradient_function = pick_gradient(
            self.apply_fun, self.parameters, dummy_sample
        )

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
        return self._cast_output(self._apply_fun_sh(s, parameters=self.parameters, batch_size=self.batchSize))
    
    @sharded()
    def _apply_fun_sh(self, s, *, parameters, batch_size):
        return self.apply_fun(parameters, s)

    def call_ratio(self, s, sp):
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
        if self.eval_ratio:
            return self._apply_ratio_sh(s, sp, parameters=self.parameters, batch_size=self.batchSize)
        
        log_psi_s = self._cast_output(self._apply_fun_sh(s, parameters=self.parameters, batch_size=self.batchSize))
        log_psi_sp = self._cast_output(self._apply_fun_sh(sp, parameters=self.parameters, batch_size=self.batchSize))
        return self._cast_output(jnp.exp(log_psi_sp - log_psi_s))
    
    @sharded()
    def _apply_ratio_sh(self, s, sp, *, parameters, batch_size):
        if self._mixed_precision:
            log_psi_s = self._cast_output(self.apply_fun(parameters, s))
            log_psi_sp = self._cast_output(self.apply_fun(parameters, sp))
            return self._cast_output(jnp.exp(log_psi_sp - log_psi_s))
        return self._cast_output(self.apply_fun(parameters, s, sp, method=self.net.eval_ratio))

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
        return self._cast_output(self._gradients_sh(s, parameters=self.parameters, batch_size=self.batchSize))
    
    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _gradients_sh(self, s, *, parameters, batch_size):
        return self.flat_gradient_function(self.apply_fun, parameters, s)
    
    def gradients_dict(self, s):
        result = self._gradients_dict_sh(s, parameters=self.parameters, batch_size=self.batchSize)

        if self.holomorphic:
            result = self._append_gradients_dict_jsh(result, result)
        return tree_map(self._cast_output, result)
    
    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _gradients_dict_sh(self, s, *, parameters, batch_size):
        return self.dict_gradient_function(self.apply_fun, parameters, s)

    def grad_dict_to_vec_map(self):
        PTreeShape = []
        start = 0
        P = jnp.arange(2 * self.numParameters)
        for s in self.paramShapes:
            # TODO: Here we need to add the treatment for the complex non-holomorphic case
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
        if "eval_real" in dir(self.net) and callable(self.net.eval_real):
            return lambda p, x: jnp.real(self._cast_output(self.apply_fun(p, x, method=self.net.eval_real))), self.parameters
        elif self._eval_ratio:
            if self._mixed_precision:
                return lambda p, x, y: self._cast_output(
                    jnp.exp(self._cast_output(self.apply_fun(p, y)) - self._cast_output(self.apply_fun(p, x)))
                ), self.parameters
            return lambda p, x, y: self._cast_output(self.apply_fun(p, x, y, method=self.net.eval_ratio)), self.parameters
        
        return lambda p, x: jnp.real(self._cast_output(self.apply_fun(p, x))), self.parameters
    
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

    def _cast_output(self, x):
        x = jnp.asarray(x)
        return x.astype(_operator_dtype_for(x.dtype))

    def _validate_parameter_dtypes(self):
        unsupported = [
            dtype for dtype in self.paramDtypes
            if jnp.dtype(dtype) not in (
                jnp.dtype(jnp.float32),
                jnp.dtype(jnp.float64),
                jnp.dtype(jnp.complex64),
                jnp.dtype(jnp.complex128),
            )
        ]
        if unsupported:
            raise TypeError(f"Unsupported parameter dtypes: {unsupported}")
        if not self.mixed_precision and any(_is_low_precision_parameter_dtype(dtype) for dtype in self.paramDtypes):
            raise ValueError(
                "NQS has fp32/complex64 parameters but mixed_precision=False. "
                "Pass mixed_precision=True or initialize the network with fp64/complex128 parameters."
            )

    def _warn_if_unsafe_param_cast(self, source, target_dtype, leaf_idx):
        source = jnp.asarray(source)
        target_dtype = jnp.dtype(target_dtype)
        if _is_complex_dtype(source.dtype) and not _is_complex_dtype(target_dtype):
            imag_norm = float(jnp.max(jnp.abs(jnp.imag(source))))
            if imag_norm > 1e-12 and (leaf_idx, "imag") not in self._warned_param_casts:
                warnings.warn(
                    "Casting complex parameter update to a real parameter leaf discards a nonzero imaginary part.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_param_casts.add((leaf_idx, "imag"))
        if self.mixed_precision and source.dtype != target_dtype and target_dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64)):
            casted = source.astype(target_dtype).astype(source.dtype)
            scale = jnp.maximum(jnp.max(jnp.abs(source)), jnp.asarray(1.0, dtype=jnp.real(source).dtype))
            rel_loss = float(jnp.max(jnp.abs(source - casted)) / scale)
            if rel_loss > 1e-5 and (leaf_idx, "precision") not in self._warned_param_casts:
                warnings.warn(
                    f"Casting parameter update from {source.dtype} to stored dtype {target_dtype} "
                    f"loses relative precision {rel_loss:.3e}.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_param_casts.add((leaf_idx, "precision"))

    def _cast_param_tree(self, params):
        if not hasattr(self, "paramDtypes"):
            return params
        payload = params["params"] if isinstance(params, dict) and "params" in params else params
        leaves, treedef = tree_flatten(payload)
        if len(leaves) != len(self.paramDtypes):
            raise ValueError(f"Parameter tree leaf mismatch: got {len(leaves)}, expected {len(self.paramDtypes)}.")
        casted = []
        for idx, (p, dtype) in enumerate(zip(leaves, self.paramDtypes)):
            self._warn_if_unsafe_param_cast(p, dtype, idx)
            casted.append(jnp.asarray(p, dtype=dtype))
        payload = tree_unflatten(treedef, casted)
        return {"params": payload} if isinstance(params, dict) and "params" in params else payload
 
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
        
        return self._cast_param_tree(tree_unflatten(self._netTreeDef, PTreeShape))
