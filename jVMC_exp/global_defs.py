import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os

allowed_dtypes_real = {
    "f64": jnp.float64,
    "f32": jnp.float32,
    "f16": jnp.float16,
    "bf16": jnp.bfloat16
}
allowed_dtypes_cpx = {
    "c128": jnp.complex128,
    "c64": jnp.complex64
}
allowed_dtypes = {**allowed_dtypes_real, **allowed_dtypes_cpx}

def _read_dtype(dtype_str: str, real: bool | None):
    dtype_str = dtype_str.lower()

    if real is not None:
        dtype_real = allowed_dtypes_real.get(dtype_str)
        dtype_cpx = allowed_dtypes_cpx.get(dtype_str)

        if real:
            if dtype_real is None:
                if dtype_cpx is not None:
                    raise ValueError("Trying to set a real variable with a complex dtype.")
                
                allowed = ", ".join(k for k in allowed_dtypes_real.keys())
                raise ValueError(
                    f"Allowed real dtypes are {allowed}. "
                    f"Got {dtype_str}."
                )
            return jnp.dtype(dtype_real)
                
        if dtype_cpx is None:
            if dtype_real is not None:
                raise ValueError("Trying to set a complex variable with a real dtype.")
            
            allowed = ", ".join(k for k in allowed_dtypes_cpx.keys())
            raise ValueError(
                f"Allowed complex dtypes are {allowed}. "
                f"Got {dtype_str}."
            )
        return jnp.dtype(dtype_cpx)
    
    dtype = allowed_dtypes.get(dtype_str)
    if dtype is None:
        allowed = ", ".join(k for k in allowed_dtypes.keys())
        raise ValueError(
            f"Allowed dtypes are {allowed}. "
            f"Got {dtype_str}."
        )
    
    return jnp.dtype(dtype)

DT_SAMPLES_CONT = jnp.float64
DT_SAMPLES = jnp.int32
DT_PARAMS_REAL = _read_dtype(os.environ.get("JVMC_DT_PARAMS_REAL", "f64"), True)
DT_PARAMS_CPX = _read_dtype(os.environ.get("JVMC_DT_PARAMS_CPX", "c128"), False)
DT_OUT_REAL = _read_dtype(os.environ.get("JVMC_DT_OUT_REAL", "f64"), True)
DT_OUT_CPX = _read_dtype(os.environ.get("JVMC_DT_OUT_CPX", "c128"), False)
DT_OPERATORS_REAL = DT_OUT_REAL
DT_OPERATORS_CPX = DT_OUT_CPX

USE_DISTRIBUTED = os.environ.get('JVMC_USE_DISTRIBUTED', 'false').lower() == 'true'

def update(
    *,
    dt_params_real: None | jnp.dtype = None,
    dt_params_cpx: None | jnp.dtype = None,
    dt_out_real: None | jnp.dtype = None,
    dt_out_cpx: None | jnp.dtype = None,
    use_distributed: None | bool = None,
):
    """
    Update the global configuration.

    Example
    -------
    >>> jVMC_exp.global_defs.update(
    ...     dt_out_real=jnp.float32,
    ...     dt_out_cpx=jnp.complex64,
    ... )
    """
    global DT_PARAMS_REAL
    global DT_PARAMS_CPX
    global DT_OUT_REAL
    global DT_OUT_CPX
    global USE_DISTRIBUTED
    global DT_OPERATORS_REAL
    global DT_OPERATORS_CPX
    
    def check(dtype, real, label):
        if real and jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError(
                f"Got complex dtype ({dtype}) for {label}."
            )
        if not real and not jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError(
                f"Got real dtype ({dtype}) for {label}."
            )
        
        return dtype

    if dt_params_real is not None:
        DT_PARAMS_REAL = check(jnp.dtype(dt_params_real), True, "dt_params_real")

    if dt_params_cpx is not None:
        DT_PARAMS_CPX = check(jnp.dtype(dt_params_cpx), False, "dt_params_cpx")

    if dt_out_real is not None:
        DT_OUT_REAL = check(jnp.dtype(dt_out_real), True, "dt_out_real")
        DT_OPERATORS_REAL = DT_OUT_REAL

    if dt_out_cpx is not None:
        DT_OUT_CPX = check(jnp.dtype(dt_out_cpx), False, "dt_out_cpx")
        DT_OPERATORS_CPX = DT_OUT_CPX

    if use_distributed is not None:
        USE_DISTRIBUTED = bool(use_distributed)