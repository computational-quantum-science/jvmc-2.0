from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial, cached_property

from jVMC_exp import global_defs
from jVMC_exp.sharding_config import DEVICE_SHARDING, DEVICE_SPEC, MESH, REPLICATED_SPEC, SizedIterable, pad_to_devices

@jax.jit
def _get_mean(data, weights):
    return jnp.tensordot(weights, data, axes=(0, 0))

@jax.jit
def _get_var(norm_data):
    return jnp.sum(jnp.abs(norm_data)**2, axis=0)

@jax.jit
def _get_error_of_mean(var, weights):
    return jnp.sqrt(var * jnp.sum(weights ** 2))

@jax.jit
def _center(data, mean):
    return data - mean

def _normalize_fn(data, weights, mean):
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data - mean)

def _inv_normalize_fn(data, inv_weights, mean, mask):
    raw = jnp.einsum("i, i... -> i...", jnp.sqrt(inv_weights), data) + mean
    return jnp.einsum("i, i... -> i...", mask, raw)

_normalize = jax.jit(_normalize_fn)
_inv_normalize = jax.jit(_inv_normalize_fn)
# The same, overwriting ``data`` (donated): no second copy of the observations
_normalize_in_place = jax.jit(_normalize_fn, donate_argnums=0)
_inv_normalize_in_place = jax.jit(_inv_normalize_fn, donate_argnums=0)

def _get_weighted_covar(data, weights, other_normalized):
    """
    Covariance of ``data`` with the normalized observations of another observable, without
    copies of ``data``: the weights and the complex conjugate are applied to the other
    observable, and a real ``data`` is never converted to complex.

    The samples are sharded over the devices; each device sums over chunks of at most
    ``global_defs.MATMUL_CHUNK_BYTES`` of its rows of ``data`` and the partial sums are
    reduced with a ``psum``. A single product over all samples had XLA allocate a copy of
    ``data`` (41 GiB for the Jacobian of 1.1M parameters on 16 GPUs).
    """
    n_samples = data.shape[0]
    n_devices = MESH.size
    if n_samples % n_devices:
        return _weighted_covar_unsharded(data, weights, other_normalized)

    n_local = n_samples // n_devices
    row_bytes = max(1, data.size // max(1, n_samples)) * data.dtype.itemsize
    budget = global_defs.MATMUL_CHUNK_BYTES or n_local * row_bytes
    chunk = max(1, min(n_local, budget // row_bytes))

    return _weighted_covar_chunked(data, weights, other_normalized, chunk=chunk)

def _weighted_covar_operand(data, weights, other_normalized):
    """
    The operand ``w`` of the product ``data^T w`` that :func:`_weighted_covar_finish` turns
    into the covariance, with ``v = sqrt(weights) * other_normalized``: ``conj(v)`` for a
    complex ``data``, ``[Re v, Im v]`` for a real ``data`` and a complex ``v``, else ``v``.
    """
    v = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), other_normalized)
    if jnp.iscomplexobj(data):
        return jnp.conj(v)
    if jnp.iscomplexobj(v):
        return jnp.concatenate([v.real, v.imag], axis=-1)

    return v

def _weighted_covar_finish(product, data, weights, other_normalized):
    """The covariance from ``data^T w``, see :func:`_weighted_covar_operand`."""
    if jnp.iscomplexobj(data):
        return jnp.conj(product).squeeze()
    if jnp.issubdtype(jnp.result_type(weights, other_normalized), jnp.complexfloating):
        k = other_normalized.shape[-1]
        return (product[..., :k] + 1j * product[..., k:]).squeeze()

    return product.squeeze()

@jax.jit
def _weighted_covar_unsharded(data, weights, other_normalized):
    w = _weighted_covar_operand(data, weights, other_normalized)

    return _weighted_covar_finish(jnp.tensordot(data, w, axes=(0, 0)), data, weights, other_normalized)

@partial(jax.jit, static_argnames=("chunk",))
def _weighted_covar_chunked(data, weights, other_normalized, *, chunk):
    def local(data, weights, other):
        w = _weighted_covar_operand(data, weights, other)
        n_local = data.shape[0]

        def add_chunk(k, acc, size=chunk):
            rows, w_rows = (jax.lax.dynamic_slice_in_dim(x, k * chunk, size) for x in (data, w))
            # Materialized: XLA would otherwise fuse the slices
            # into the product and autotune it on whole operands
            rows, w_rows = jax.lax.optimization_barrier((rows, w_rows))
            return acc + jnp.tensordot(rows, w_rows, axes=(0, 0))

        acc = jax.lax.fori_loop(1, n_local // chunk, add_chunk, add_chunk(0, 0))
        if n_local % chunk:
            acc = add_chunk(n_local // chunk, acc, n_local % chunk)

        return jax.lax.psum(acc, "devices")

    product = jax.shard_map(
        local, mesh=MESH, in_specs=(DEVICE_SPEC,) * 3, out_specs=REPLICATED_SPEC
    )(data, weights, other_normalized)

    return _weighted_covar_finish(product, data, weights, other_normalized)

@jax.jit
def _mean_and_weighted_no_center(data, weights):
    mean = jnp.tensordot(weights, data, axes=(0, 0))
    weighted = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data)
    
    return mean, weighted

@jax.jit
def _get_covar(norm_data_1, norm_data_2):
    return jnp.tensordot(jnp.conj(norm_data_1), norm_data_2, axes=(0, 0)).squeeze()

@jax.jit
def _get_self_covar(norm_data):
    return jnp.tensordot(jnp.conj(norm_data), norm_data, axes=(0, 0)).squeeze()

@jax.jit
@jax.vmap
def _get_covar_per_sample(centered_data_1, centered_data_2):
    return jnp.outer(jnp.conj(centered_data_1), centered_data_2)

@jax.jit(static_argnums=(1,))
def _apply_and_project(data, apply_fn, projection):
    return jnp.matmul(projection, apply_fn(data)) 

@jax.jit(static_argnums=(1, 2))
@partial(jax.vmap, in_axes=(0, None, None))
def _get_autocorrelation_time(x, c, dim):
    fft = jnp.abs(jnp.fft.fft(x - jnp.mean(x), n=dim))
    correlation = jnp.fft.ifft(fft ** 2)[:x.size].real
    correlation = correlation / correlation[0]
    tau = 2 * jnp.cumsum(correlation) - 1

    m = jnp.arange(len(tau)) < c * tau
    window = jax.lax.cond(jnp.any(m), lambda: jnp.argmin(m), lambda: len(tau) - 1)

    return tau[window]

class SampledObs():
    def __init__(self, observations, weights=None, donate=False):
        """
        Initializes SampledObs class.

        Args:
            * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)`. \
                The array must have a leading batch dimension.
            * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
            * ``donate``: Whether ``observations`` may be overwritten. The observations are kept in a
                single buffer that is converted between raw, centered and normalized on access; with
                ``donate=True`` the first conversion overwrites the given array (it is deleted)
                instead of making a second copy of it. For arrays nobody else holds, such as a
                freshly computed Jacobian.
        """
        if len(observations.shape) == 1:
            self._num_obs = 1
            observations = observations.reshape((-1, 1))
        else:
            self._num_obs = jnp.prod(jnp.array(observations.shape[1:]))
        self._num_samples = observations.shape[0]
        num_devices = MESH.shape["devices"]

        if weights is None:
            weights = jnp.ones(self._num_samples, dtype=observations.dtype) / self._num_samples
        elif weights.shape != (self._num_samples,):
            raise ValueError(f"Weights must have shape ({self._num_samples},), got {weights.shape}")
        weights /= jnp.sum(weights)

        remainder = self._num_samples % num_devices
        if remainder != 0:
            num_pad = num_devices - remainder
            pad = ((0, num_pad),) + ((0, 0),) * (observations.ndim - 1)
            observations = jnp.pad(observations, pad, mode='constant')
            weights = jnp.pad(weights, (0, num_pad), constant_values=0)
        inv_weights = jnp.where(weights != 0, 1 / weights, weights)
        zero_mask = jnp.where(weights != 0, 1, 0)

        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._inv_weights = jax.device_put(inv_weights, DEVICE_SHARDING)
        self._zero_mask = jax.device_put(zero_mask, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING)

        self._state = "o"
        # Whether the buffer may be held outside of this object, in which case it must not be
        # overwritten: true for an array that was given (unless donated) and for one handed out
        # by `observations`, `_centered_obs` or `_normalized_obs`; a conversion makes a new one
        self._exported = not (donate or remainder != 0)
        self._mean = _get_mean(self._observations, self.weights)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self._num_obs == 1:
            return f"{self.mean.item():.4e} ± {self.error_of_mean.item():.4e} (Var = {self.var.item():.4e})"
        else:
            return f"SampledObs with {self._num_obs} features"

    def _convert(self, state):
        """
        The buffer converted to ``state`` (``"o"``: raw, ``"c"``: centered, ``"n"``: normalized),
        overwriting it unless it may be held outside (`_exported`). Not handed out: the caller
        passes it on to a computation and keeps no reference.
        """
        if state == self._state:
            return self._observations

        if state == "o":
            if self._state == "c":
                fn, args = _normalize_fn, (self._zero_mask, -self.mean)
            else:
                fn, args = _inv_normalize_fn, (self._inv_weights, self.mean, self._zero_mask)
        elif state == "c":
            if self._state == "o":
                fn, args = _normalize_fn, (self._zero_mask, self.mean)
            else:
                fn, args = _normalize_fn, (self._inv_weights, 0)
        else:
            fn, args = _normalize_fn, (self.weights, self.mean if self._state == "o" else 0)

        # In place only if the result has the same type as the buffer (otherwise it cannot be reused)
        in_place = (
            not self._exported
            and jax.eval_shape(fn, self._observations, *args).dtype == self._observations.dtype
        )
        if fn is _normalize_fn:
            convert = _normalize_in_place if in_place else _normalize
        else:
            convert = _inv_normalize_in_place if in_place else _inv_normalize

        self._observations = convert(self._observations, *args)
        self._state = state
        self._exported = False

        return self._observations

    def _export(self, state):
        observations = self._convert(state)
        self._exported = True

        return observations

    @property
    def observations(self):
        return self._export("o")

    @property
    def _centered_obs(self):
        return self._export("c")
    
    @property
    def _normalized_obs(self):
        return self._export("n")
    
    @property
    def weights(self):
        return self._weights

    @property
    def effective_num_samples(self):
        """
        Kish effective sample size :math:`1/\\sum_n w_n^2`.

        Equals the number of samples for uniform weights and is the quantity
        that replaces it in Monte-Carlo error estimates when the samples carry
        importance weights (e.g. from ``CutoffSampler`` or ``mu != 2``).
        """
        return 1.0 / jnp.sum(self.weights ** 2)

    @property
    def mean(self):
        return self._mean
    
    @property
    def var(self):
        return _get_var(self._convert("n"))
    
    @property
    def error_of_mean(self):
        return _get_error_of_mean(self.var, self.weights)

    def get_covar(self, other: SampledObs | None = None):
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            return _get_self_covar(self._convert("n"))

        # `other` first: if it is this object, its conversion would otherwise overwrite `data`
        other_normalized = other._normalized_obs

        return _get_weighted_covar(self._convert("o"), self.weights, other_normalized)

    def get_covar_obs(self, other: SampledObs | None = None) -> SampledObs:
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        other_num_obs = self._num_obs if other is None else other._num_obs
        if other is None:
            centered = self._centered_obs
            covar_per_sample = _get_covar_per_sample(centered, centered)
        else:
            covar_per_sample = _get_covar_per_sample(self._centered_obs, other._centered_obs)
        if other_num_obs == 1:
            covar_per_sample = covar_per_sample[..., 0]

        return SampledObs(covar_per_sample, self.weights)
    
    def get_R_hat(self, n_chains):
        """
        Compute the Gelman-Rubin R-hat convergence diagnostic.

        The samples are split into `n_chains` equal-length chains and the
        between-chain and within-chain variances are compared. If the chains
        have converged to the same distribution, R-hat will be close to 1.
        Convergence is typically assumed when R-hat < 1.01.

        Args:
            n_chains : int
                Number of chains 
        """
        if self._num_obs != 1:
            raise NotImplementedError

        chain_length = self._num_samples // n_chains
        chain_obs = SampledObs(self.observations.reshape((n_chains, chain_length)).T)

        B = jnp.var(chain_obs.mean, ddof=1)
        W = jnp.mean(chain_obs.var * chain_length / (chain_length - 1))
        
        return jnp.sqrt(((chain_length - 1) / chain_length * W + B) / W)
    
    def get_autocorrelation_time(self, n_chains, c=5) -> SampledObs:
        if self._num_obs != 1:
            raise NotImplementedError

        chain_length = self._num_samples // n_chains
        obs = self.observations.reshape((n_chains, chain_length))
        z = obs - jnp.mean(obs, axis=-1)[:, None]
        dim = int(jnp.ceil(jnp.log2(chain_length)))
        
        return SampledObs(_get_autocorrelation_time(z, c, dim))
    
    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> SampledObs:
        """
        Apply a transformation to observations. 
        It modifies the underlying observations to save memory.
            
        The transformation is applied in two stages:
        1. Element-wise function applied to each observation
        2. (Optional) Linear projection via matrix multiplication
        
        Args:
            element_wise_fun: Function applied element-wise to each observation.
            linear_map: Optional linear transformation matrix applied after element_wise_fun.
        """
        observations = self._convert("o")
        if linear_map is not None:
            self._observations = _apply_and_project(observations, element_wise_fn, linear_map)
        else:
            self._observations = jax.jit(element_wise_fn)(observations)
        # A function that returns its input unchanged gives back the same buffer
        self._exported = self._exported and self._observations is observations
        self._mean = _get_mean(self._observations, self.weights)

    def select(self, idx):
        """
        Returns a `SampledObs` for the data selection indicated by the given indices.

        Args:
            * ``idx``: Indices of selected data.
        """
        return SampledObs(self.observations[:, idx], self.weights)
    
    def get_subset(self, start=None, end=None, step=None) -> SampledObs:
        """
        Returns a `SampledObs` for a subset of the observartions.

        Args:
            * ``start``: Start sample index for subset selection
            * ``end``: End sample index for subset selection
            * ``step``: Sample index step for subset selection
        """ 
        sl = slice(start, end, step)
        new_weights = self.weights[sl]

        return SampledObs(self.observations[sl], new_weights / jnp.sum(new_weights))

class LazySampledObs():
    def __init__(self, observations: SizedIterable, weights):
        """
        Args:
            * ``observations``: Batched, lazily (re)computed observations.
                Its ``layout`` (set by `sharded(..., yield_iter=True)`) defines which
                samples make up each batch. It is reused to split `weights`, and any
                other full-length array, into batches that line up exactly with the
                observations'. On more than one device a batch does not consist of
                consecutive samples, see `jVMC_exp.sharding_config.BatchLayout`.
            * ``weights``: Full-length weights array, matching `observations`
                sample-for-sample.
        """
        if weights is None:
            raise ValueError("LazySampledObs require weights to be an array")
        if not isinstance(observations, SizedIterable) or observations.layout is None:
            raise ValueError("Observations must be a SizedIterable with a BatchLayout")
        self._layout = observations.layout
        
        # Like the observations, the weights are padded to a multiple of the number of devices.
        # The padding samples get zero weight and do not contribute to any estimate.
        num_padded = len(pad_to_devices(weights))
        if num_padded != self._layout.num_samples or len(observations) != self._layout.n_batches:
            raise ValueError(
                f"Got {len(weights)} weights and {len(observations)} batches of observations, "
                f"but the layout describes {self._layout.num_samples} samples "
                f"in {self._layout.n_batches} batches."
            )
        weights /= jnp.sum(weights)
        self._num_samples = len(weights)
        self._batch_size = observations.batch_size

        self._full_weights = jax.device_put(pad_to_devices(weights), DEVICE_SHARDING)
        self._weights = self._layout.split(self._full_weights)
        self.observations = observations

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, value):
        if not isinstance(value, SizedIterable):
            raise ValueError(
                "Observations must be an instance jVMC_exp.sharding_config.SizedIterable"
        )

        self._observations = value

        # Clear cached properties
        for name, attr in type(self).__dict__.items():
            if isinstance(attr, cached_property):
                self.__dict__.pop(name, None)
    
    @property
    def layout(self):
        return self._layout

    @property
    def weights(self):
        return self._full_weights
    
    @cached_property
    def mean(self):
        mean = 0
        for batch, weights in zip(self._observations, self._weights):
            mean += _get_mean(batch, weights)
        
        return mean

    @cached_property
    def var(self):
        sq_sum = 0
        mean = 0
        for batch, weights in zip(self._observations, self._weights):
            sq_sum += jnp.tensordot(weights, jnp.abs(batch) ** 2, axes=(0, 0))
            mean += _get_mean(batch, weights)

        return (sq_sum - jnp.abs(mean) ** 2).squeeze()
    
    @property
    def error_of_mean(self):
        return _get_error_of_mean(self.var, self.weights)

    def get_covar(self, other: SampledObs | LazySampledObs | None = None):
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        covar = 0
        if other is None:
            mean = 0
            for batch, weights in zip(self._observations, self._weights):
                batch_mean, batch = _mean_and_weighted_no_center(batch, weights)
                mean += batch_mean
                covar += _get_covar(batch, batch)

            return (covar - jnp.tensordot(jnp.conj(mean), mean, axes=0)).squeeze()

        elif isinstance(other, SampledObs):
            normalized_obs_other = self._layout.split(other._normalized_obs)

            for batch, weights, batch_other in zip(self._observations, self._weights, normalized_obs_other):
                covar += _get_weighted_covar(batch, weights, batch_other)

            return covar.squeeze()

        elif isinstance(other, LazySampledObs):
            if other._layout != self._layout:
                raise ValueError(
                    f"Both LazySampledObs must be batched with the same layout, "
                    f"got {self._layout} and {other._layout}"
                )
            mean_1 = 0
            mean_2 = 0
            for batch_1, weights_1, batch_2, weights_2 in zip(self._observations, self._weights, other._observations, other._weights):
                m1, batch_1 = _mean_and_weighted_no_center(batch_1, weights_1)
                m2, batch_2 = _mean_and_weighted_no_center(batch_2, weights_2)
                mean_1 += m1
                mean_2 += m2
                covar += _get_covar(batch_1, batch_2)

            return (covar - jnp.tensordot(jnp.conj(mean_1), mean_2, axes=0)).squeeze()
        
        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )

    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> LazySampledObs:
        if linear_map is not None:
            raise NotImplementedError(
                "A linear map can't be applied without materializing all the observables"
            )

        jitted_fn = jax.jit(element_wise_fn)
        iterable = self._observations
        self.observations = SizedIterable(
            lambda: map(jitted_fn, iterable), 
            iterable.n_iterations, 
            iterable.batch_size, 
            iterable.layout
        )
