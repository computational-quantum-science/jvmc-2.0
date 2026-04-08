import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import PartitionSpec as P
from functools import partial, wraps
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

def is_on_device(args, target_sharding=DEVICE_SHARDING):
    return any(jax.tree_util.tree_map(lambda x: x.sharding == target_sharding, args))

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
            print(f"WARNING: Number of {label} ({global_size}) is smaller than the total number of devices ({total_devices}).")
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
            out_specs=DEVICE_SPEC,
            automatic_sharding=False
    ):
        self.static_argnums = static_argnums
        self.static_kwarg_names = set(static_kwarg_names + ('batch_size',))
        self.use_vmap = use_vmap
        self.vmap_in_axes = vmap_in_axes
        self.in_specs = in_specs
        self.in_sharding = None
        self.out_specs = out_specs
        self.automatic_sharding = automatic_sharding
        
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
                self.in_sharding = tuple(REPLICATED_SHARDING if REPLICATED_SPEC == s else DEVICE_SHARDING for s in self.in_specs)
                
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

        if self.automatic_sharding:
            jsh_fn = jax.jit(vmapd_fn, static_argnums=self.static_argnums)
        else:
            jsh_fn = jax.jit(
                jax.shard_map(
                    vmapd_fn, 
                    mesh=MESH, 
                    in_specs=(REPLICATED_SPEC,) + self.in_specs, 
                    out_specs=self.out_specs
                ),
                static_argnums=self.static_argnums
            )

        batched_fn = partial(self._batched_wrapper, jsh_fn=jsh_fn)
        
        return {'single': base_fn, "vmapd": vmapd_fn, 'batched': batched_fn}

    def _batched_wrapper(self, batch_size, kwargs, *args, jsh_fn):
        """Wrapper for batched computation - assumes batch_size is divisible by number of devices."""
        num_samples = args[0].shape[0]
        append = (-num_samples) % batch_size
        total_sumples = num_samples + append

        if (total_sumples > batch_size) and is_on_device(args):
            args = tuple(jax.device_put(a, REPLICATED_SHARDING) for a in args)
       
        batched_args = tuple(
            jnp.pad(a, [(0, append),] + [(0, 0)] * (len(a.shape) - 1)).reshape((-1, batch_size) + a.shape[1:]) 
            for a in args
        )

        results = []
        num_batches = batched_args[0].shape[0]
        for batch_idx in range(num_batches):
            batch_args = tuple(
                jax.device_put(ba[batch_idx], self.in_sharding[arg_idx]) for arg_idx, ba in enumerate(batched_args)
            )
            results.append(jsh_fn(kwargs, *batch_args))

        def stack_reshape_trim(*xs):
            x = jnp.stack(xs)
            x = x.reshape((-1,) + x.shape[2:])
            return x[:num_samples]

        return jax.tree_util.tree_map(stack_reshape_trim, *results)