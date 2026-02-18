import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import PartitionSpec as P
from functools import partial, wraps
from jax.experimental.shard_map import shard_map
import os

USE_DISTRIBUTED = os.environ.get('JVMC_USE_DISTRIBUTED', 'false').lower() == 'true'

if USE_DISTRIBUTED:
    try:
        jax.distributed.initialize()
        num_processes = jax.process_count()
        num_devices = jax.device_count() 
        print(f"JAX distributed initialized: {num_processes} processes and {num_devices} devices.")
    except Exception as e:
        print(f"Failed to initialize JAX distributed: {e}")
else:
    print("Running in single-node mode (JVMC_USE_DISTRIBUTED not set)")

global_devices = mesh_utils.create_device_mesh((jax.device_count(),))
MESH = Mesh(global_devices, axis_names=("devices",))
DEVICE_SHARDING = NamedSharding(MESH, P("devices"))
REPLICATED_SHARDING = NamedSharding(MESH, P())
DEVICE_SPEC = P("devices")
REPLICATED_SPEC = P()

def distribute(global_size: int, label: str | None=None):
    """
    Adjust a global array size to be compatible with device sharding.

    This helper ensures that a quantity intended to be sharded across the
    JAX device mesh is compatible with the number of available devices.
    In particular, it enforces that the global size is at least the number
    of devices and divisible by it, so that each device receives an equal
    shard.

    Parameters
    ----------
    global_size : int
        Total number of elements before sharding (e.g. number of chains,
        walkers, or other per-device replicated objects).
    label : str
        Human-readable label used in warning messages to identify the
        adjusted quantity.

    Returns
    -------
    int
        A possibly increased size that is:
        - greater than or equal to the number of devices, and
        - exactly divisible by the number of devices.

    Notes
    -----
    If `global_size` is smaller than the number of devices, it is increased
    to match the device count. If it is not divisible by the device count,
    it is rounded up to the next multiple. In both cases, a warning is
    printed to notify the user of the adjustment.
    """
    total_devices = MESH.shape["devices"]

    if global_size < total_devices:
        if label is not None:
            print("WARNING: Number of {label} ({global_size}) is smaller than the total number of devices ({total_devices}).")
            print(f"         Increased to: {total_devices}")
        return total_devices
    if global_size % total_devices != 0:
        adjusted_size = ((global_size + total_devices - 1) // total_devices) * total_devices
        if label is not None:
            print(f"WARNING: Number of {label} ({global_size}) is not divisible by the number of devices ({total_devices}).")
            print(f"         Increased to: {adjusted_size}")
        return adjusted_size

    return global_size

def broadcast_split_key(key, n_out_keys: int):
    """
    Split a PRNG key on a single host and broadcast the result to all hosts.

    This function ensures that all JAX processes receive the same set of
    PRNG subkeys in a multi-host setting. Only process 0 performs the key
    splitting, while all other processes allocate a dummy array of the
    correct shape. The resulting array of subkeys is then broadcast from
    process 0 to all other processes.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Base PRNG key used to generate subkeys. The key is only consumed
        on process 0.
    n_out_keys : int
        Number of PRNG subkeys to generate.

    Returns
    -------
    jax.Array
        Array of shape `(n_out_keys, 2)` containing PRNG subkeys, identical
        across all processes and suitable for further splitting or sharding.
    """
    if jax.process_index() == 0:
        # Only process 0 generates the keys
        out_keys = jax.random.split(key, n_out_keys)
    else:
        # Other processes create dummy data with the correct shape
        out_keys = jnp.zeros((n_out_keys, 2), dtype=jnp.uint32)

    # Broadcast from process 0 to all processes
    out_keys = multihost_utils.broadcast_one_to_all(out_keys)

    return out_keys.astype(jnp.uint32)

def create_batches(configs, b):
    append = b * ((configs.shape[0] + b - 1) // b) - configs.shape[0]
    pads = [(0, append), ] + [(0, 0)] * (len(configs.shape) - 1)

    return jnp.pad(configs, pads).reshape((-1, b) + configs.shape[1:])
    
from typing import ParamSpec, TypeVar, Callable

P = ParamSpec('P')
R = TypeVar('R')

class sharded:
    """Decorator to automatically create sharded versions of methods."""
    def __init__(
            self,
            static_argnums=None,
            static_kwarg_names=(),
            use_vmap=True, vmap_in_axes=None, # If None, default to (0,) * num_args
            in_specs=None,                    # If None, default to (DEVICE_SPEC,) * num_args
            out_specs=DEVICE_SPEC
    ):
        self.static_argnums = static_argnums
        self.static_kwarg_names = set(static_kwarg_names + ('batch_size',))
        self.use_vmap = use_vmap
        self.vmap_in_axes = vmap_in_axes
        self.in_specs = in_specs
        self.out_specs = out_specs
        
    def __call__(self, method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            if not hasattr(instance, '_sharded_cache'):
                instance._sharded_cache = {}

            method_name = method.__name__

            if method_name not in instance._sharded_cache:
                if self.in_specs is None:
                    self.in_specs = (DEVICE_SPEC,) * len(args)
                else:
                    if len(self.in_specs) != len(args):
                        raise ValueError(f"in_specs length ({len(self.in_specs)}) must match "
                                         f"number of args ({len(args)})")
                
                if self.vmap_in_axes is None:
                    self.vmap_in_axes = (0,) * len(args)
                else:
                    if len(self.vmap_in_axes) != len(args):
                        raise ValueError(f"vmap_in_axes length ({len(self.vmap_in_axes)}) must match "
                                         f"number of args ({len(args)})")
                    
                if kwargs['batch_size'] % MESH.size != 0:
                    raise ValueError(f"The batch size ({kwargs['batch_size']}) has to be divisible by the number of devices ({MESH.size})")
                
                static_kwargs = {k: v for k, v in kwargs.items() if k in self.static_kwarg_names}
                base_fn = lambda kw, *a: method(instance, *a, **kw, **static_kwargs)
                instance._sharded_cache[method_name] = self._create_sharded_versions(base_fn)

            batch_size = kwargs['batch_size']
            kwargs = {k: v for k, v in kwargs.items() if k not in self.static_kwarg_names}

            return instance._sharded_cache[method_name]['batched'](batch_size, kwargs, *args)
        return wrapper
    
    def _create_sharded_versions(self, base_fn):
        """
        Create all versions of the method.
        """
        vmapd_fn = jax.vmap(base_fn, in_axes=(None,) + self.vmap_in_axes) if self.use_vmap else base_fn
        jsh_fn = jax.jit(shard_map(vmapd_fn, MESH, (REPLICATED_SPEC,) + self.in_specs, self.out_specs), static_argnums=self.static_argnums)
        batched_fn = partial(self._batched_wrapper, jsh_fn=jsh_fn)
        
        return {'single': base_fn, "vmapd": vmapd_fn, 'batched': batched_fn}

    def _batched_wrapper(self, batch_size, kwargs, *args, jsh_fn):
        """Wrapper for batched computation - assumes batch_size is divisible by number of devices."""
        num_samples = args[0].shape[0]
        append = (-num_samples) % batch_size
       
        batched_args = tuple(
            jnp.pad(a, [(0, append),] + [(0, 0)] * (len(a.shape) - 1)).reshape((-1, batch_size) + a.shape[1:]) 
            for a in args
        )

        results = []
        for batch_idx in range(batched_args[0].shape[0]):
            batch_args = tuple(ba[batch_idx] for ba in batched_args)
            results.append(jsh_fn(kwargs, *batch_args))

        def stack_reshape_trim(*xs):
            x = jnp.stack(xs)
            x = x.reshape((-1,) + x.shape[2:])
            return x[:num_samples]

        return jax.tree_util.tree_map(stack_reshape_trim, *results)
    
# class ShardedMethod:
#     """Decorator to automatically create sharded versions of methods."""
#     def __init__(self, use_batch=True, out_specs=DEVICE_SPEC, attr_source=None):
#         """
#         Args:
#             use_batch: Whether to create batched version
#             out_specs: Output sharding spec
#             attr_source: If provided, get parameters/sampleShape/batchSize from kwargs[attr_source] at call time
#         """
#         self.use_batch = use_batch
#         self.vmap_in_axes = (None, 0, None) # parameters, samples, kwargs
#         self.in_specs = (REPLICATED_SPEC, DEVICE_SPEC, REPLICATED_SPEC)
#         self.out_specs = out_specs
#         self.attr_source = attr_source
        
#     def __call__(self, method):
#         @wraps(method)
#         def wrapper(instance, s, **kwargs):
#             if not hasattr(instance, '_sharded_cache'):
#                 instance._sharded_cache = {}
            
#             method_name = method.__name__
#             if method_name not in instance._sharded_cache:
#                 # Call method with s=None and kwargs to get the base lambda
#                 base_fn = method(instance, None, **kwargs)
#                 instance._sharded_cache[method_name] = self._create_sharded_versions(instance, base_fn)
            
#             cache = instance._sharded_cache[method_name]
#             return self._handle_sharding_cases(instance, cache, s, kwargs)        
#         return wrapper
    
#     def _create_sharded_versions(self, instance, base_fn):
#         """Create all sharded versions of the method.
        
#         base_fn has signature: (parameters, sample, **kwargs) -> result
#         """
#         vmapd_fn = jax.vmap(base_fn, in_axes=self.vmap_in_axes)
#         jsh_fn = jax.jit(shard_map(vmapd_fn, MESH, self.in_specs, self.out_specs))
        
#         if self.use_batch:
#             batched_fn = partial(self._batched_wrapper, instance=instance, vmapd_fn=vmapd_fn)
#             batched_jsh_fn = jax.jit(shard_map(batched_fn, MESH, self.in_specs, self.out_specs))

#             # batched_fn = partial(self._batched_wrapper, instance=instance, vmapd_fn=jax.jit(vmapd_fn))
#             # batched_jsh_fn = shard_map(batched_fn, MESH, self.in_specs, self.out_specs)

#             # batched_fn = partial(self._batched_wrapper, instance=instance, vmapd_fn=jax.jit(vmapd_fn))
#             # batched_jsh_fn = batched_fn
#         else:
#             batched_jsh_fn = None
        
#         return {
#             'single': base_fn,
#             'vmap': vmapd_fn,
#             'jsh': jsh_fn,
#             'batched_jsh': batched_jsh_fn
#         }
    
#     def _get_attributes(self, instance, kwargs):
#         """
#         Get parameters, sampleShape, batchSize from instance or attr_source.
        
#         Returns a tuple of (parameters, sampleShape, batchSize, filtered_kwargs)
#         where filtered_kwargs has the attr_source removed if present.
#         """
#         filtered_kwargs = dict(kwargs)
        
#         if self.attr_source and self.attr_source in kwargs:
#             source = kwargs[self.attr_source]
#             # Remove attr_source from kwargs to pass to computation
#             filtered_kwargs.pop(self.attr_source)
#             return source.parameters, source.sampleShape, source.batchSize, filtered_kwargs
        
#         return instance.parameters, instance.sampleShape, instance.batchSize, filtered_kwargs
    
#     def _handle_sharding_cases(self, instance, cache, s, kwargs):
#         """
#         Helper function to handle different sharding cases.
        
#         kwargs may contain attr_source (e.g., 'psi') and runtime params (e.g., 't').
#         """
#         # Get attributes and filter out attr_source from kwargs
#         parameters, sampleShape, batchSize, filtered_kwargs = self._get_attributes(instance, kwargs)
        
#         # Case: single sample
#         if s.shape == sampleShape:
#             return jax.jit(cache['single'])(parameters, s, filtered_kwargs)
        
#         num_devices = MESH.size
#         batch_shape = s.shape[:-len(sampleShape)]
#         s = s.reshape((-1,) + sampleShape)
#         total_samples = s.shape[0]
        
#         # Case: fewer samples than devices -> fall back to vmap
#         if total_samples < num_devices:
#             result = jax.jit(cache['vmap'])(parameters, s, filtered_kwargs)
#         else: 
#             # Case: not divisible by num_devices -> pad
#             original_size = total_samples
#             if total_samples % num_devices != 0:
#                 pad_size = num_devices - (total_samples % num_devices)
#                 s = jnp.concatenate([s, jnp.repeat(s[-1:], pad_size, axis=0)], axis=0)
#                 total_samples = s.shape[0]
            
#             samples_per_device = total_samples // num_devices
#             instance._temp_batchSize = batchSize
            
#             # Case: batch_size is None or samples_per_device <= batch_size
#             if batchSize is None or samples_per_device < batchSize:
#                 result = cache['jsh'](parameters, s, filtered_kwargs)
#             else:
#                 result = cache['batched_jsh'](parameters, s, filtered_kwargs)
            
#             # Clean up temporary attribute
#             if hasattr(instance, '_temp_batchSize'):
#                 delattr(instance, '_temp_batchSize')
            
#             # Trim padding
#             if original_size != total_samples:
#                 result = jax.tree_util.tree_map(lambda x: x[:original_size], result)

#         def restore_batch_size(x):
#             return x.reshape(batch_shape + x.shape[1:])
        
#         return jax.tree_util.tree_map(restore_batch_size, result)
    
#     def _batched_wrapper(self, parameters, s, kwargs, instance, vmapd_fn):
#         """Wrapper for batched computation."""
#         batchSize = instance._temp_batchSize 
#         sb = create_batches(s, batchSize)
        
#         def scan_fun(c, x):
#             batch_result = vmapd_fn(parameters, x, kwargs)
#             return c, batch_result
            
#         result = jax.lax.scan(scan_fun, None, jnp.array(sb))[1]

#         def reshape_and_trim(x):
#             x = x.reshape((-1,) + x.shape[2:])
#             return x[:s.shape[0]]

#         return jax.tree_util.tree_map(reshape_and_trim, result)