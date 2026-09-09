import jax
import jax.numpy as jnp
import warnings
import numpy as np
from typing import Literal
import math

try:
    import jaxmg
except ImportError:
    jaxmg = None

from jVMC_exp.global_defs import USE_DISTRIBUTED
from jVMC_exp.sharding_config import (
    MESH, MESH_2D,
    DEVICE_SHARDING, DEVICE_SHARDING_2D
)

def _distributed_warning(A, mode):
    if mode.lower() == "distributed":
        if jaxmg is None:
            if jax.process_index() == 0:
                warnings.warn(
                    "mode='distributed' requires the 'jaxmg' package, which is not installed "
                    "(install it via a CUDA extra, e.g. `pip install jVMC_exp[cuda13]`). "
                    "Falling back to mode='device'."
                )
            mode = "device"

        if jax.default_backend() != "gpu":
            if jax.process_index() == 0:
                warnings.warn(
                    "mode='distributed' requires a real GPU backend, "
                    f"got jax.default_backend()={jax.default_backend()}. "
                    "Falling back to mode='device'."
                )
            mode = "device"

        if A.shape[0] % MESH.shape["devices"] != 0:
            if jax.process_index() == 0:
                warnings.warn(
                    "mode=distributed requires the given matrix to "
                    "have shape divisible by the number of devices."
                    f"Got shape={A.shape}, n_devices={MESH.shape['devices']}."
                    "Falling back to mode='device'."
                )
            mode = "device"

        if MESH.shape["devices"] == 1:
            if jax.process_index() == 0:
                warnings.warn(
                    "Got mode=distributed but there is only one device. "
                    "Falling back to mode='device'."
                )
            mode = "device"

        if not USE_DISTRIBUTED:
            if jax.process_index() == 0:
                warnings.warn(
                    "mode='distributed' requires JVMC_USE_DISTRIBUTED=true and "
                    "one process per GPU, since jaxmg's cuSOLVERMp backend "
                    "needs rank-per-GPU execution "
                    "(e.g. launch with `srun --ntasks=<n_gpus> --gpus-per-task=1`). "
                    "Falling back to mode='device'."
                )
            mode = "device"

    return mode

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

def smooth_cutoff_fn(x, c, exp=6):
    return 1 / (1 + (c / x)**exp)

def diagonalize(
        A,
        pad_size: int = 0,
        mode: Literal["device", "distributed", "host"] = "device",
        T_A: int | None = None
    ):
    """
    Diagonalize a Hermitian matrix, optionally sharded across devices.

    Parameters
    ----------
    A : array_like
        Hermitian matrix to diagonalize.
    pad_size : int, default=0
        Number of trailing rows/columns of `A` that are padding rather than
        real data. If nonzero, a large constant is added to the diagonal of
        the padded block before diagonalization, pushing the corresponding
        fake eigenvalues (and their eigenvectors) to the end of the
        spectrum.
    mode : {"device", "distributed", "host"}, default="device"
        Diagonalization backend:

        - "device": `jax.numpy.linalg.eigh` on the default device(s).
        - "distributed": distributed diagonalization via `jaxmg.syevd` on
          `MESH_2D`. Falls back to "device" if there is no GPU backend, if
          `A`'s size isn't divisible by the number of devices, if there is
          only one device, or if `JVMC_USE_DISTRIBUTED` isn't set (jaxmg's
          cuSOLVERMp backend needs rank-per-GPU execution).
        - "host": `numpy.linalg.eigh` on the host.
    T_A : int, optional
        Tile size used for the distributed diagonalization
        (`mode="distributed"`). If not given, it defaults to the greatest
        common divisor of `A.shape[0]` divided by `MESH_2D`'s process rows
        and columns.

    Returns
    -------
    ev : jax.Array
        Eigenvalues of `A` in ascending order, still including the
        `pad_size` fake eigenvalues at the end.
    V : jax.Array
        Eigenvectors of `A`, still including the `pad_size` fake
        rows/columns.

    Notes
    -----
    The padding is deliberately *not* stripped off before returning `ev`
    and `V`. When `V` is sharded across devices, slicing it on the sharded
    axis would force a reshard, gathering it and replicating it onto every
    device. 
    """
    mode = _distributed_warning(A, mode)

    if pad_size !=0:
        # Frobenius norm >= spectral radius for Hermitian A
        # Used here to recognize the fake eigenvalues given by padding
        c = 2 * jnp.linalg.norm(A) + 1
        N = A.shape[0] - pad_size
        pad_idx = jnp.arange(N, A.shape[0])
        A = A.at[pad_idx, pad_idx].add(c)

    if mode.lower() == "device":
        try:
            ev, V = jnp.linalg.eigh(A)
        except ValueError:
            if jax.process_index() == 0:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            ev, V = _eigh_numpy(A)

    elif mode.lower() == "host":
        ev, V = _eigh_numpy(A)

    elif mode.lower() == "distributed":
        A = jax.device_put(A, DEVICE_SHARDING_2D, donate=True)

        if T_A is None:
            p_rows, p_cols = MESH_2D.shape["row"], MESH_2D.shape["col"]
            T_A = math.gcd((A.shape[0]) // p_rows, (A.shape[0]) // p_cols)

        ev, V = jaxmg.syevd(A, T_A, MESH_2D)

        V = jax.device_put(V, DEVICE_SHARDING)

    else:
        raise ValueError(
            f"The available modes are 'device', 'host' or 'distributed' got {mode}."
        )

    return ev, V