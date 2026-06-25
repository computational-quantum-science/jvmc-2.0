from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp

import jVMC_exp.global_defs as global_defs

@dataclass(frozen=True)
class PrecisionDTypes:
    mode: str
    param_dtype: any
    param_complex_dtype: any
    compute_dtype: any
    observable_dtype: any
    complex_dtype: any

def precision_dtypes(mode: str) -> PrecisionDTypes:
    key = str(mode).strip().lower()
    if key == "mixed_fp32":
        return PrecisionDTypes(
            mode=key,
            param_dtype=jnp.float32,
            param_complex_dtype=jnp.complex64,
            compute_dtype=jnp.float32,
            observable_dtype=jnp.float64,
            complex_dtype=jnp.complex128,
        )
    if key == "full_fp64":
        return PrecisionDTypes(
            mode=key,
            param_dtype=jnp.float64,
            param_complex_dtype=jnp.complex128,
            compute_dtype=jnp.float64,
            observable_dtype=jnp.float64,
            complex_dtype=jnp.complex128,
        )
    raise ValueError("precision_mode must be 'mixed_fp32' or 'full_fp64'.")

def set_precision_mode(mode: str) -> PrecisionDTypes:
    dtypes = precision_dtypes(mode)
    global_defs.DT_PARAMS_REAL = dtypes.param_dtype
    global_defs.DT_PARAMS_CPX = dtypes.param_complex_dtype
    global_defs.DT_OPERATORS_REAL = dtypes.observable_dtype
    global_defs.DT_OPERATORS_CPX = dtypes.complex_dtype
    return dtypes
