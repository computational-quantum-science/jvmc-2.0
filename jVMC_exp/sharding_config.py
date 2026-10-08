import jax
import jax.numpy as jnp
import numpy as np
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

def pad_to_devices(x):
    pad = (-x.shape[0]) % MESH.size
    if pad == 0:
        return x

    return jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))

def _per_device(x):
    return x.reshape((MESH.size, -1) + x.shape[1:])

@jax.jit(static_argnames=("b", "n_batches"))
def _take_batch(x, i, *, b, n_batches):
    x_dev = _per_device(x)
    pad = n_batches * b - x_dev.shape[1]
    if pad:
        x_dev = jnp.pad(x_dev, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 1))

    return jax.lax.dynamic_slice_in_dim(
        x_dev, i * b, b, axis=1
    ).reshape((MESH.size * b,) + x.shape[1:])

@jax.jit(static_argnames=("rows",))
def _trim_batch(piece, *, rows):
    return _per_device(piece)[:, :rows].reshape((MESH.size * rows,) + piece.shape[1:])

@jax.jit(static_argnames=("b", "rows"), donate_argnums=0)
def _put_batch(out, piece, i, *, b, rows):
    piece_dev = _per_device(piece)[:, :rows]

    return jax.lax.dynamic_update_slice_in_dim(
        _per_device(out), piece_dev, i * b, axis=1
    ).reshape(out.shape)

@dataclass(frozen=True)
class BatchLayout:
    """
    Splits ``num_samples`` samples, sharded across devices along the first axis, into
    batches of ``batch_size`` samples. ``num_samples`` has to be divisible by the number
    of devices, see `pad_to_devices`.

    Batch ``i`` is made of the ``i``-th block of ``batch_size // n_devices`` rows of every
    device's shard, so taking a batch out of an array, or writing the result of a batch
    back into one, never moves data between devices. The last batch holds the remaining
    rows of every device, padded to ``batch_size`` rows when it is taken. Hence, on more
    than one device a batch does not consist of consecutive samples: any full-length
    array that has to line up with the batches must be split with `split` of the same layout.
    """
    num_samples: int
    batch_size: int

    def __post_init__(self):
        to_check = [
            ("number of samples", self.num_samples), ("batch size", self.batch_size)
        ]

        for name, value in to_check:
            if value <= 0 or value % MESH.size != 0:
                raise ValueError(
                    f"The {name} ({value}) has to be a positive "
                    f"multiple of the number of devices ({MESH.size})"
                )

    @property
    def n_batches(self):
        return math.ceil(self._local_size / self._local_batch_size)

    @property
    def _local_size(self):
        return self.num_samples // MESH.size

    @property
    def _local_batch_size(self):
        return self.batch_size // MESH.size

    @property
    def _last_local_size(self):
        return self._local_size - (self.n_batches - 1) * self._local_batch_size

    def take(self, x, i):
        """
        Batch ``i`` of the row-sharded array ``x``, with ``batch_size`` rows.
        The last batch is zero padded if it is not full.
        """
        if self._local_size == self._local_batch_size:
            return x
        return _take_batch(x, i, b=self._local_batch_size, n_batches=self.n_batches)

    def trim(self, piece, i):
        """
        Removes the padding rows from the result of batch ``i``.
        """
        if i == self.n_batches - 1 and self._last_local_size < self._local_batch_size:
            return _trim_batch(piece, rows=self._last_local_size)

        return piece

    def split(self, x):
        """
        Split x in batches so that it matches the layout.
        """
        if x.shape[0] != self.num_samples:
            raise ValueError(
                f"Expected an array with {self.num_samples} rows to split, "
                f"got shape {x.shape}"
            )
        x = jax.device_put(x, DEVICE_SHARDING)

        return [self.trim(self.take(x, i), i) for i in range(self.n_batches)]

    def alloc(self, piece):
        """
        Zeros with ``num_samples`` rows and the trailing shape, dtype and sharding of ``piece``.
        """
        return jnp.zeros(
            (self.num_samples,) + piece.shape[1:], piece.dtype, device=piece.sharding
        )

    def put(self, out, piece, i):
        """
        Writes the result of batch ``i`` (with or without padding) into ``out``, in place:
        ``out`` is donated and must not be used afterwards.
        """
        original_rows = (
            self._last_local_size if i == self.n_batches - 1 
            else self._local_batch_size
        )

        return _put_batch(
            out, piece, i, b=self._local_batch_size, rows=original_rows
        )

    def scaled(self, k):
        """
        Layout of arrays holding ``k`` consecutive rows per sample.
        """
        # TODO: Why are we also scaling the batchsize?
        return BatchLayout(k * self.num_samples, k * self.batch_size)

    def batch_positions(self):
        """
        For every sample, its position in the concatenation of all batches.
        """
        b = self._local_batch_size
        local_rows = np.arange(self.num_samples).reshape(MESH.size, self._local_size)
        order = np.concatenate([local_rows[:, i * b:(i + 1) * b].reshape(-1) for i in range(self.n_batches)])
        positions = np.empty_like(order)
        positions[order] = np.arange(self.num_samples)

        return positions

@dataclass
class SizedIterable:
    reusable_iterable: Callable
    n_iterations: int
    batch_size: int
    layout: BatchLayout | None = None

    def __len__(self):
        return self.n_iterations

    def __iter__(self) -> Iterator:
        return self.reusable_iterable()

P = ParamSpec('P')
R = TypeVar('R')

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

class sharded:
    """
    Decorator to automatically create sharded versions of methods.

    The decorated method is written for a single sample: it is vmapped (if ``use_vmap``)
    and sharded across devices. It is called with a ``batch_size`` keyword argument.
    With a batch size, the samples are evaluated in batches of ``batch_size`` (see
    `BatchLayout`), which bounds how much is computed at once and fixes the input shape
    of the compiled function, whatever the number of samples. With ``batch_size=None``
    all samples are evaluated in a single call.

    A batched call assumes that
        * every positional argument holds one entry per sample along its first axis, and
          is split into batches along that axis;
        * every leaf of the output holds one entry per sample along its first axis.
    Therefore ``in_specs``, ``out_specs`` and ``vmap_in_axes`` can only be customised
    for methods that are called with ``batch_size=None``.
    """
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
        self._customised_specs = [
            name for name, customised in (
                ("in_specs", in_specs is not None),
                ("out_specs", out_specs != DEVICE_SPEC),
                ("vmap_in_axes", vmap_in_axes is not None),
            ) if customised
        ]

    def __call__(self, method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            jsh_fn = self._get_jsh(instance, method, args, kwargs)
            batch_size = kwargs['batch_size']
            kwargs = {k: v for k, v in kwargs.items() if k not in self.static_kwarg_names}
            num_samples = args[0].shape[0]

            if batch_size is None and not self.yield_iter:
                args = tuple(
                    jax.device_put(a, self.in_sharding[i]) for i, a in enumerate(args)
                )

                return jsh_fn(kwargs, *args)

            if self._customised_specs:
                raise ValueError(
                    f"{', '.join(self._customised_specs)} can only be customised for calls "
                    "without a batch size: batched calls split every argument and every "
                    "output along the first axis."
                )

            # Padding to a multiple of the number of devices makes the samples shardable,
            # the last batch of every device is then padded to the batch size by BatchLayout.take
            args = tuple(jax.device_put(pad_to_devices(a), DEVICE_SHARDING) for a in args)
            num_padded = args[0].shape[0]
            layout = BatchLayout(
                num_padded, num_padded if batch_size is None else batch_size
            )

            if self.yield_iter:
                return SizedIterable(
                    reusable_iterable=lambda: self._iter_local_batches(
                        layout, kwargs, args, jsh_fn
                    ),
                    n_iterations=layout.n_batches,
                    batch_size=layout.batch_size,
                    layout=layout
                )

            out = self._call_local(layout, kwargs, args, jsh_fn)
            if num_padded != num_samples:
                # This will remplicate the output, but it's not a problem, 
                # since the input was not divisible by the number of devices anyways.
                out = jax.tree_util.tree_map(lambda x: x[:num_samples], out)

            return out

        return wrapper

    def _get_jsh(self, instance, method, args, kwargs):
        if not hasattr(instance, '_sharded_cache'):
            instance._sharded_cache = {}

        # Static kwargs are baked into the compiled function, so they are part of the key
        static_kwargs = {k: v for k, v in kwargs.items() if k in self.static_kwarg_names}
        cache_key = (method.__name__, tuple(sorted(static_kwargs.items())))

        if cache_key not in instance._sharded_cache:
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

            base_fn = lambda kw, *a: method(instance, *a, **kw, **static_kwargs)
            instance._sharded_cache[cache_key] = self._create_sharded_fn(base_fn)

        return instance._sharded_cache[cache_key]

    def _create_sharded_fn(self, base_fn):
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

        return jsh_fn

    def _iter_local_batches(self, layout: BatchLayout, kwargs, args, jsh_fn):
        """
        Generator yielding the result of one batch of ``layout`` at a time, without padding.
        """
        for i in range(layout.n_batches):
            yield jax.tree_util.tree_map(
                lambda x: layout.trim(x, i), 
                jsh_fn(kwargs, *(layout.take(a, i) for a in args))
            )

    def _call_local(self, layout: BatchLayout, kwargs, args, jsh_fn):
        """
        Evaluates all batches of ``layout`` and writes their results in place into
        preallocated outputs, which keep the order of the input samples.
        """
        if layout.n_batches == 1:
            return next(self._iter_local_batches(layout, kwargs, args, jsh_fn))

        out = None
        for i in range(layout.n_batches):
            result = jsh_fn(kwargs, *(layout.take(a, i) for a in args))
            if out is None:
                # The output is allocated once and every batch result is written into it and
                # then freed, so at most the output and one batch result exist at a time
                out = jax.tree_util.tree_map(layout.alloc, result)
            out = jax.tree_util.tree_map(lambda o, r: layout.put(o, r, i), out, result)

        return jax.block_until_ready(out)