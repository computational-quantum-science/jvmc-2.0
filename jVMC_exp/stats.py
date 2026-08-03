from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial, cached_property

from jVMC_exp.sharding_config import DEVICE_SHARDING, MESH, SizedIterable
from jVMC_exp.sharding_config import SizedIterable

def _reshape_in_batches(data, batch_size: int):
    num_samples = data.shape[0]
    append = (-num_samples) % batch_size
    if append:
        data = jnp.pad(data, ((0, append),) + ((0, 0),) * (data.ndim - 1), constant_values=0)
    n_batches = data.shape[0] // batch_size

    batched_data = []
    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        if i == n_batches - 1 and append:
            batch = batch[:batch_size - append]
        batched_data.append(jax.device_put(batch, DEVICE_SHARDING))

    return batched_data

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

@jax.jit
def _normalize(data, weights, mean):
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data - mean)

@jax.jit
def _normalize_no_center(data, weights):
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data)

@jax.jit(donate_argnums=(0,))
def _normalize_no_copy(data, weights):
    mean = jnp.tensordot(weights, data, axes=(0, 0))
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data - mean)

@jax.jit
def _get_covar(norm_data_1, norm_data_2):
    return jnp.tensordot(jnp.conj(norm_data_1), norm_data_2, axes=(0, 0)).squeeze()

@jax.jit
@jax.vmap
def _get_covar_per_sample(centered_data_1, centered_data_2):
    return jnp.outer(jnp.conj(centered_data_1), centered_data_2)

@jax.jit
@jax.vmap
def _outer_per_sample(data_1, data_2):
    return jnp.outer(data_1, data_2)

@jax.jit
def _get_covar_var_re_im(centered_data_1, centered_data_2, weights):
    """
    Variance of the real and imaginary parts of the covariance
    estimator (plus their cross-covariance), without assuming circular
    symmetry of the underlying complex noise.

    Obtained from the total variance E[|y-Ey|^2] together with
    the pseudo-variance ("relation") E[(y-Ey)^2], where
    y_n = conj(centered_data_1_n) (x) centered_data_2_n is the per-sample
    covariance contribution. Since Re(y)^2+Im(y)^2 = |y|^2 and
    Re(y)^2-Im(y)^2 + 2i Re(y)Im(y) = y^2, the two moments above are enough
    to solve exactly for Var(Re y), Var(Im y) and Cov(Re y, Im y).
    """
    covar_per_sample = _get_covar_per_sample(centered_data_1, centered_data_2)
    covar = jnp.tensordot(weights, covar_per_sample, axes=(0, 0))
    var_total = jnp.tensordot(weights, jnp.abs(covar_per_sample) ** 2, axes=(0, 0)) - jnp.abs(covar) ** 2

    relation_per_sample = _outer_per_sample(jnp.conj(centered_data_1) ** 2, centered_data_2 ** 2)
    relation = jnp.tensordot(weights, relation_per_sample, axes=(0, 0)) - covar ** 2

    var_re = (var_total + jnp.real(relation)) / 2
    var_im = (var_total - jnp.real(relation)) / 2
    cov_re_im = jnp.imag(relation) / 2

    return var_re.squeeze(), var_im.squeeze(), cov_re_im.squeeze()

@jax.jit
def _get_covar_var_re_im_moments(data_1, data_2, weights):
    """
    Raw (uncentered) moments needed to accumulate, across batches, both the
    total variance of the covariance estimator and its pseudo-variance, 
    from which the exact (no isotropy assumption) variance.
    """
    data_1 = data_1.reshape(data_1.shape[0], -1)
    data_2 = data_2.reshape(data_2.shape[0], -1)
    sq_1 = jnp.abs(data_1) ** 2
    sq_2 = jnp.abs(data_2) ** 2
    conj_1_sq = jnp.conj(data_1) ** 2
    data_2_sq = data_2 ** 2

    mean_1 = jnp.tensordot(weights, data_1, axes=(0, 0))
    mean_2 = jnp.tensordot(weights, data_2, axes=(0, 0))
    var_1 = jnp.tensordot(weights, sq_1, axes=(0, 0))
    var_2 = jnp.tensordot(weights, sq_2, axes=(0, 0))

    m_11 = jnp.tensordot(weights, _get_covar_per_sample(data_1, data_2), axes=(0, 0))
    m_20 = jnp.tensordot(weights, _outer_per_sample(data_1, data_2), axes=(0, 0))
    m_sq1 = jnp.tensordot(weights, _outer_per_sample(sq_1, data_2), axes=(0, 0))
    m_sq2 = jnp.tensordot(weights, _outer_per_sample(data_1, sq_2), axes=(0, 0))
    m_sqsq = jnp.tensordot(weights, _outer_per_sample(sq_1, sq_2), axes=(0, 0))

    q_1 = jnp.tensordot(weights, conj_1_sq, axes=(0, 0))
    q_2 = jnp.tensordot(weights, data_2_sq, axes=(0, 0))
    m_21 = jnp.tensordot(weights, _outer_per_sample(conj_1_sq, data_2), axes=(0, 0))
    m_12 = jnp.tensordot(weights, _outer_per_sample(jnp.conj(data_1), data_2_sq), axes=(0, 0))
    m_22 = jnp.tensordot(weights, _outer_per_sample(conj_1_sq, data_2_sq), axes=(0, 0))

    return mean_1, mean_2, var_1, var_2, m_11, m_20, m_sq1, m_sq2, m_sqsq, q_1, q_2, m_21, m_12, m_22

@jax.jit
def _finalize_covar_var_re_im(
    mean_1, mean_2, var_1, var_2, m_11, m_20, m_sq1, m_sq2, m_sqsq, q_1, q_2, m_21, m_12, m_22
):
    covar = m_11 - jnp.outer(jnp.conj(mean_1), mean_2)

    second_moment = (
        m_sqsq
        - 2 * jnp.real(jnp.conj(mean_2)[None, :] * m_sq1)
        - 2 * jnp.real(jnp.conj(mean_1)[:, None] * m_sq2)
        + 2 * jnp.real(jnp.outer(jnp.conj(mean_1), jnp.conj(mean_2)) * m_20)
        + 2 * jnp.real(jnp.outer(mean_1, jnp.conj(mean_2)) * m_11)
        + jnp.outer(var_1, jnp.abs(mean_2) ** 2)
        + jnp.outer(jnp.abs(mean_1) ** 2, var_2)
        - 3 * jnp.outer(jnp.abs(mean_1) ** 2, jnp.abs(mean_2) ** 2)
    )
    var_total = second_moment - jnp.abs(covar) ** 2

    A = jnp.conj(mean_1)
    B = mean_2
    relation = (
        m_22
        - 2 * B[None, :] * m_21
        - 2 * A[:, None] * m_12
        + 4 * jnp.outer(A, B) * m_11
        + jnp.outer(q_1, B ** 2)
        + jnp.outer(A ** 2, q_2)
        - 3 * jnp.outer(A ** 2, B ** 2)
        - covar ** 2
    )

    var_re = (var_total + jnp.real(relation)) / 2
    var_im = (var_total - jnp.real(relation)) / 2
    cov_re_im = jnp.imag(relation) / 2

    return covar.squeeze(), var_re.squeeze(), var_im.squeeze(), cov_re_im.squeeze()

@jax.jit(static_argnums=(1,))
def _apply_and_project(data, apply_fn, projection):
    return jnp.matmul(projection, apply_fn(data)) 

@jax.jit
def _get_tangent_kernel(norm_data):
    return jnp.matmul(norm_data, jnp.conj(jnp.transpose(norm_data)))

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
    def __init__(self, observations, weights=None):
        """
        Initializes SampledObs class.

        Args:
            * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)`. \
                The array must have a leading batch dimension.
            * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
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
        
        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING, donate=True)

        self._consumed = False

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self._num_obs == 1:
            return f"{self.mean.item():.4e} ± {self.error_of_mean.item():.4e} (Var = {self.var.item():.4e})"
        else:
            return f"SampledObs with {self._num_obs} features"

    @property
    def observations(self):
        if self._consumed:
            raise RuntimeError(
                "This SampledObs was consumed by _get_normalized_obs_and_consume() "
                "(its buffer was donated) and can no longer be used."
            )
        return self._observations
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def mean(self):
        return _get_mean(self.observations, self.weights)

    @property
    def _centered_obs(self):
        return _center(self.observations, self.mean)

    @property
    def _normalized_obs(self):
        return _normalize(self.observations, self.weights, self.mean)
    
    @property
    def var(self):
        return _get_var(self._normalized_obs)
    
    @property
    def error_of_mean(self):
        return _get_error_of_mean(self.var, self.weights)
    
    @property
    def tangent_kernel(self):
        return _get_tangent_kernel(self._normalized_obs)

    def get_covar(self, other: SampledObs | None = None):
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        
        return _get_covar(self._normalized_obs, other._normalized_obs)
    
    def get_covar_var(self, other: SampledObs | None = None):
        """
        Returns the variance of the real and imaginary parts of the
        covariance, and their cross-covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.

        Returns:
            Tuple ``(var_re, var_im, cov_re_im)``.
        """
        if other is None:
            other = self

        return _get_covar_var_re_im(self._centered_obs, other._centered_obs, self.weights)

    def get_covar_obs(self, other: SampledObs | None = None) -> SampledObs:
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        covar_per_sample = _get_covar_per_sample(self._centered_obs, other._centered_obs)
        if other._num_obs == 1:
            covar_per_sample = covar_per_sample[..., 0]

        return SampledObs(covar_per_sample, self.weights)
    
    def get_covar_and_covar_var(self, other: SampledObs | None = None):
        """
        Returns the covariance and the variance of its real and
        imaginary parts (and their cross-covariance).

        Returns:
            Tuple ``(covar, var_re, var_im, cov_re_im)``.
        """
        covar = self.get_covar(other)
        var_re, var_im, cov_re_im = self.get_covar_var(other)

        return covar, var_re, var_im, cov_re_im
    
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
        if linear_map is not None:
            self._observations = _apply_and_project(self.observations, element_wise_fn, linear_map)
        else:
            self._observations = jax.jit(element_wise_fn)(self.observations)

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
    
    def _get_normalized_obs_and_consume(self):
        norm_obs = _normalize_no_copy(self.observations, self.weights)
        self._consumed = True

        return norm_obs

class LazySampledObs():
    def __init__(self, observations: SizedIterable, weights):
        """
        Args:
            * ``observations``: Batched, lazily (re)computed observations.
                Its ``batch_size`` (e.g. set by `sharded(..., yield_iter=True)`)
                is reused to split `weights` into batches that line up
                exactly with observations' -- a target batch *count* isn't
                enough for this, since e.g. splitting 20 samples into 4 equal
                batches ([5,5,5,5]) is a different partition than a
                batch_size=6 iterable's ([6,6,6,2]).
            * ``weights``: Full-length weights array, matching `observations`
                sample-for-sample.
        """
        if weights is None:
            raise ValueError("LazySampledObs require weights to be an array")
        weights /= jnp.sum(weights)
        self._num_samples = len(weights)
        self._batch_size = observations.batch_size

        self._weights = _reshape_in_batches(weights, self._batch_size)
        if len(self._weights) != len(observations):
            raise ValueError(
                f"observations.batch_size={self._batch_size} splits {self._num_samples} weights "
                f"into {len(self._weights)} batches, but observations has {len(observations)} "
                "batches -- they must be built with the same batch_size."
            )
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
    def weights(self):
        return jnp.concatenate(self._weights)
    
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
                weighted_data = _normalize_no_center(batch, weights)
                mean += _get_mean(batch, weights)
                covar += _get_covar(weighted_data, weighted_data)

            return (covar - jnp.tensordot(jnp.conj(mean), mean, axes=0)).squeeze()

        elif isinstance(other, SampledObs):
            normalized_obs_other = _reshape_in_batches(other._normalized_obs, self._batch_size)

            for batch, weights, batch_other in zip(self._observations, self._weights, normalized_obs_other):
                weighted_data = _normalize_no_center(batch, weights)
                covar += _get_covar(weighted_data, batch_other)

            return covar.squeeze()
        
        elif isinstance(other, LazySampledObs):
            mean_1 = 0
            mean_2 = 0
            for batch_1, weights_1, batch_2, weights_2 in zip(self._observations, self._weights, other._observations, other._weights):
                weighted_data_1 = _normalize_no_center(batch_1, weights_1)
                weighted_data_2 = _normalize_no_center(batch_2, weights_2)
                mean_1 += _get_mean(batch_1, weights_1)
                mean_2 += _get_mean(batch_2, weights_2)
                covar += _get_covar(weighted_data_1, weighted_data_2)

            return (covar - jnp.tensordot(jnp.conj(mean_1), mean_2, axes=0)).squeeze()
        
        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )
        
    def get_covar_and_covar_var(self, other: SampledObs | LazySampledObs | None = None):
        """
        Returns the covariance and the variance of its real and
        imaginary parts (and their cross-covariance).

        Both quantities are accumulated together over a single pass through
        the underlying iterable(s), so this should be preferred over calling
        `get_covar` and `get_covar_var` separately.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs` or `LazySampledObs`.

        Returns:
            Tuple ``(covar, var_re, var_im, cov_re_im)``.
        """
        def _accumulate(moments, batch_1, batch_2, weights):
            batch_moments = _get_covar_var_re_im_moments(batch_1, batch_2, weights)
            if moments is None:
                return batch_moments
            return tuple(m + b for m, b in zip(moments, batch_moments))

        moments = None

        if other is None:
            for batch, weights in zip(self._observations, self._weights):
                moments = _accumulate(moments, batch, batch, weights)

        elif isinstance(other, SampledObs):
            other_batches = _reshape_in_batches(other.observations, self._batch_size)

            for batch, weights, batch_other in zip(self._observations, self._weights, other_batches):
                moments = _accumulate(moments, batch, batch_other, weights)

        elif isinstance(other, LazySampledObs):
            for batch_1, weights_1, batch_2 in zip(self._observations, self._weights, other._observations):
                moments = _accumulate(moments, batch_1, batch_2, weights_1)

        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )

        return _finalize_covar_var_re_im(*moments)

    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> LazySampledObs:
        if linear_map is not None:
            raise NotImplementedError(
                "A linear map can't be applied withoud materializing all the observables"
            )
        
        jitted_fn = jax.jit(element_wise_fn)
        iterable = self._observations
        transormed_iterable = lambda: (jitted_fn(batch) for batch in iterable)
        self.observations = SizedIterable(transormed_iterable, iterable.n_iterations, iterable.batch_size)