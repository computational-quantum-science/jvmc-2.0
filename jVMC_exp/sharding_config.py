import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import PartitionSpec as P
from functools import wraps
from dataclasses import dataclass
from typing import Iterator, ParamSpec, TypeVar, Callable
import math

from jVMC_exp import global_defs

if global_defs.USE_DISTRIBUTED:
    try:
        jax.distributed.initialize()
        num_processes = jax.process_count()
        num_devices = jax.device_count()
        if jax.process_index() == 0:
            print(f"JAX distributed initialized: {num_processes} processes and {num_devices} devices.")
    except Exception as e:
        if jax.process_index() == 0:
            print(f"Failed to initialize JAX distributed: {e}")
else:
    if jax.process_index() == 0:
        print("Running in single-node mode (JVMC_USE_DISTRIBUTED not set)")

global_devices = mesh_utils.create_device_mesh((jax.device_count(),))
MESH = Mesh(global_devices, axis_names=("devices",))
DEVICE_SHARDING = NamedSharding(MESH, P("devices"))
REPLICATED_SHARDING = NamedSharding(MESH, P())
DEVICE_SPEC = P("devices")
REPLICATED_SPEC = P()

def make_2d_mesh(axis_names=("row", "col")):
    num_devices = jax.device_count()

    p_rows = int(math.isqrt(num_devices))
    while num_devices % p_rows != 0:
        p_rows -= 1
    p_cols = num_devices // p_rows
    devices_2d = mesh_utils.create_device_mesh((p_rows, p_cols))
    
    return Mesh(devices_2d, axis_names=axis_names)

MESH_2D = make_2d_mesh(axis_names=("row", "col"))
DEVICE_SPEC_2D = P("row", "col")
DEVICE_SHARDING_2D = NamedSharding(MESH_2D, DEVICE_SPEC_2D)

@dataclass
class SizedIterable:
    reusable_iterable: Callable
    n_iterations: int
    batch_size: int

    def __len__(self):
        return self.n_iterations

    def __iter__(self) -> Iterator:
        return self.reusable_iterable()

P = ParamSpec('P')
R = TypeVar('R')

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
    
class sharded:
    """Decorator to automatically create sharded versions of methods."""
    def __init__(
            self,
            static_argnums=None,
            static_kwarg_names=(),
            use_vmap=True, vmap_in_axes=None, # If None, default to (0,) * num_args
            in_specs=None,                    # If None, default to (DEVICE_SPEC,) * num_args
            out_specs=DEVICE_SPEC,
            automatic_sharding=False,
            donate_argnums=None,
            yield_iter=False,
    ):
        self.static_argnums = static_argnums
        self.static_kwarg_names = set(static_kwarg_names + ('batch_size',))
        self.use_vmap = use_vmap
        self.vmap_in_axes = vmap_in_axes
        self.in_specs = in_specs
        self.in_sharding = None
        self.out_specs = out_specs
        self.automatic_sharding = automatic_sharding
        self.donate_argnums = donate_argnums
        self.yield_iter = yield_iter

    def __call__(self, method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            jsh_fn = self._get_jsh(instance, method, args, kwargs)
            batch_size, kwargs = self._split_batch_size(kwargs)

            num_samples = args[0].shape[0]
            n_batches = 1 if batch_size is None else math.ceil(num_samples / batch_size)

            reusable_iterable = lambda: self._iter_batches(batch_size, kwargs, *args, jsh_fn=jsh_fn)
            if self.yield_iter:
                resolved_batch_size = num_samples if batch_size is None else batch_size
                return SizedIterable(
                    reusable_iterable=reusable_iterable, n_iterations=n_batches, batch_size=resolved_batch_size
                )

            def concat(*xs):
                if len(xs) == 1:
                    return xs[0]
                # NOTE: jnp.concatenate along an already-partitioned axis propagates the
                # existing per-device sharding for free (no gather/replicate). Using
                # jnp.array/jnp.stack on a list of sharded arrays instead loses this and
                # silently replicates everything onto a single device layout.
                return jnp.concatenate(xs, axis=0)

            return jax.tree_util.tree_map(concat, *list(reusable_iterable()))
        
        return wrapper

    def _split_batch_size(self, kwargs):
        batch_size = kwargs['batch_size']
        kwargs = {k: v for k, v in kwargs.items() if k not in self.static_kwarg_names}
        
        return batch_size, kwargs

    def _get_jsh(self, instance, method, args, kwargs):
        if not hasattr(instance, '_sharded_cache'):
            instance._sharded_cache = {}

        method_name = method.__name__

        if method_name not in instance._sharded_cache:
            if self.in_specs is None:
                self.in_specs = (DEVICE_SPEC,) * len(args)
            elif len(self.in_specs) != len(args):
                raise ValueError(f"in_specs length ({len(self.in_specs)}) must match "
                                 f"number of args ({len(args)})")
            self.in_sharding = tuple(
                REPLICATED_SHARDING if REPLICATED_SPEC == s else DEVICE_SHARDING for s in self.in_specs
            )

            if self.vmap_in_axes is None:
                self.vmap_in_axes = (0,) * len(args)
            elif len(self.vmap_in_axes) != len(args):
                raise ValueError(f"vmap_in_axes length ({len(self.vmap_in_axes)}) must match "
                                 f"number of args ({len(args)})")

            if kwargs['batch_size'] is not None and kwargs['batch_size'] % MESH.size != 0:
                raise ValueError(f"The batch size ({kwargs['batch_size']}) "
                                 f"has to be divisible by the number of devices ({MESH.size})")

            static_kwargs = {k: v for k, v in kwargs.items() if k in self.static_kwarg_names}
            base_fn = lambda kw, *a: method(instance, *a, **kw, **static_kwargs)
            instance._sharded_cache[method_name] = self._create_sharded_versions(base_fn)

        return instance._sharded_cache[method_name]['jsh']

    def _create_sharded_versions(self, base_fn):
        vmapd_fn = jax.vmap(
            base_fn, in_axes=(None,) + self.vmap_in_axes
        ) if self.use_vmap else base_fn

        if self.automatic_sharding:
            jsh_fn = jax.jit(
                vmapd_fn, 
                static_argnums=self.static_argnums, 
                donate_argnums=self.donate_argnums
            )
        else:
            jsh_fn = jax.jit(
                jax.shard_map(
                    vmapd_fn,
                    mesh=MESH,
                    in_specs=(REPLICATED_SPEC,) + self.in_specs,
                    out_specs=self.out_specs
                ),
                static_argnums=self.static_argnums,
                donate_argnums=self.donate_argnums
            )

        return {'single': base_fn, 'vmapd': vmapd_fn, 'jsh': jsh_fn}

    def _iter_batches(self, batch_size, kwargs, *args, jsh_fn):
        """
        Generator yielding one (trimmed) chunk result at a time.
        Assumes batch_size is divisible by number of devices.
        """
        num_samples = args[0].shape[0]
        if batch_size is None:
            batch_size = num_samples
            args = tuple(jax.device_put(a, self.in_sharding[i]) for i, a in enumerate(args))

            yield jsh_fn(kwargs, *args)
            return

        append = (-num_samples) % batch_size
        total_samples = num_samples + append

        if (total_samples > batch_size) and is_on_device(args):
            args = tuple(jax.device_put(a, REPLICATED_SHARDING) for a in args)

        batched_args = tuple(
            jnp.pad(
                a, [(0, append),] + [(0, 0)] * (len(a.shape) - 1)
            ).reshape((-1, batch_size) + a.shape[1:]) for a in args
        )

        num_batches = batched_args[0].shape[0]
        for batch_idx in range(num_batches): 
            result = jsh_fn(
                kwargs, 
                *tuple(
                    jax.device_put(
                        ba[batch_idx], self.in_sharding[arg_idx]
                    ) for arg_idx, ba in enumerate(batched_args)
                )
            )

            if batch_idx == num_batches - 1 and append:
                trim = batch_size - append
                result = jax.tree_util.tree_map(lambda x: x[:trim], result)

            yield result