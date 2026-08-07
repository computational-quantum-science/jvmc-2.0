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
def _inv_normalize(data, inv_weights, mean, mask):
    raw = jnp.einsum("i, i... -> i...", jnp.sqrt(inv_weights), data) + mean
    return jnp.einsum("i, i... -> i...", mask, raw)

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
def _covar_partial(centered_data_1, centered_data_2, weights):
    centered_data_1 = centered_data_1.reshape(centered_data_1.shape[0], -1)
    centered_data_2 = centered_data_2.reshape(centered_data_2.shape[0], -1)
    w_c1 = _normalize_no_center(centered_data_1, weights)
    w_c2 = _normalize_no_center(centered_data_2, weights)

    return jnp.tensordot(jnp.conj(w_c1), w_c2, axes=(0, 0))

@jax.jit
def _second_moment_partial(centered_data_1, centered_data_2, weights):
    centered_data_1 = centered_data_1.reshape(centered_data_1.shape[0], -1)
    centered_data_2 = centered_data_2.reshape(centered_data_2.shape[0], -1)
    w_sq1 = _normalize_no_center(jnp.abs(centered_data_1) ** 2, weights)
    w_sq2 = _normalize_no_center(jnp.abs(centered_data_2) ** 2, weights)

    return jnp.tensordot(w_sq1, w_sq2, axes=(0, 0))

@jax.jit
def _relation_partial(centered_data_1, centered_data_2, weights):
    centered_data_1 = centered_data_1.reshape(centered_data_1.shape[0], -1)
    centered_data_2 = centered_data_2.reshape(centered_data_2.shape[0], -1)
    w_conj_c1_sq = _normalize_no_center(jnp.conj(centered_data_1) ** 2, weights)
    w_c2_sq = _normalize_no_center(centered_data_2 ** 2, weights)

    return jnp.tensordot(w_conj_c1_sq, w_c2_sq, axes=(0, 0))

@jax.jit
def _covar_var_re_im_partial_self(centered_data, weights):
    centered_data = centered_data.reshape(centered_data.shape[0], -1)
    w_c = _normalize_no_center(centered_data, weights)
    w_sq = _normalize_no_center(jnp.abs(centered_data) ** 2, weights)
    w_conj_sq = _normalize_no_center(jnp.conj(centered_data) ** 2, weights)

    covar = jnp.tensordot(jnp.conj(w_c), w_c, axes=(0, 0))
    second_moment = jnp.tensordot(w_sq, w_sq, axes=(0, 0))
    relation = jnp.tensordot(w_conj_sq, w_conj_sq, axes=(0, 0))
    return covar, second_moment, relation

def _get_covar_var_re_im_partial(centered_data_1, centered_data_2, weights):
    """
    Per-batch partial sums for the variance of the real and imaginary
    parts of the covariance estimator (plus their cross-covariance).

    Obtained from the total variance E[|y-Ey|^2] together with
    the pseudo-variance ("relation") E[(y-Ey)^2], where
    y_n = conj(centered_data_1_n) (x) centered_data_2_n is the per-sample
    covariance contribution. Since Re(y)^2+Im(y)^2 = |y|^2 and
    Re(y)^2-Im(y)^2 + 2i Re(y)Im(y) = y^2, the two moments above are enough
    to solve exactly for Var(Re y), Var(Im y) and Cov(Re y, Im y).
    """
    if centered_data_1 is centered_data_2:
        return _covar_var_re_im_partial_self(centered_data_1, weights)

    covar = _covar_partial(centered_data_1, centered_data_2, weights)
    second_moment = _second_moment_partial(centered_data_1, centered_data_2, weights)
    relation = _relation_partial(centered_data_1, centered_data_2, weights)
    return covar, second_moment, relation

@jax.jit
def _finalize_covar_var_re_im(covar, second_moment, relation):
    var_total = second_moment - jnp.abs(covar) ** 2
    relation = relation - covar ** 2

    var_re = (var_total + jnp.real(relation)) / 2
    var_im = (var_total - jnp.real(relation)) / 2
    cov_re_im = jnp.imag(relation) / 2

    return covar.squeeze(), var_re.squeeze(), var_im.squeeze(), cov_re_im.squeeze()

def _get_covar_var_re_im(centered_data_1, centered_data_2, weights):
    covar, second_moment, relation = _get_covar_var_re_im_partial(centered_data_1, centered_data_2, weights)
    _, var_re, var_im, cov_re_im = _finalize_covar_var_re_im(covar, second_moment, relation)
    return var_re, var_im, cov_re_im

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
        inv_weights = jnp.where(weights != 0, 1 / weights, weights)
        zero_mask = jnp.where(weights != 0, 1, 0)

        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._inv_weights = jax.device_put(inv_weights, DEVICE_SHARDING)
        self._zero_mask = jax.device_put(zero_mask, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING)

        self._state = "o"
        self._mean = _get_mean(self._observations, self.weights)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self._num_obs == 1:
            return f"{self.mean.item():.4e} ± {self.error_of_mean.item():.4e} (Var = {self.var.item():.4e})"
        else:
            return f"SampledObs with {self._num_obs} features"

    @property
    def observations(self):
        if self._state == "c":
            self._observations = _normalize(self._observations, self._zero_mask, -self.mean)
        elif self._state == "n":
            self._observations = _inv_normalize(
                self._observations, self._inv_weights, self.mean, self._zero_mask
            )
        self._state = "o"

        return self._observations

    @property
    def _centered_obs(self):
        if self._state == "o":
            self._observations = _normalize(self._observations, self._zero_mask, self.mean)
        elif self._state == "n":
            self._observations = _normalize(self._observations, self._inv_weights, 0)
        self._state = "c"

        return self._observations
    
    @property
    def _normalized_obs(self):
        if self._state == "o":
            self._observations = _normalize(self._observations, self.weights, self.mean)
        elif self._state == "c":
            self._observations = _normalize(self._observations, self.weights, 0)
        self._state = "n"

        return self._observations
    
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
        return _get_var(self._normalized_obs)
    
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
            normalized = self._normalized_obs
            return _get_covar(normalized, normalized)

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
            centered = self._centered_obs
            return _get_covar_var_re_im(centered, centered, self.weights)

        return _get_covar_var_re_im(self._centered_obs, other._centered_obs, self.weights)

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
    
    def get_covar_and_covar_var(self, other: SampledObs | None = None):
        """
        Returns the covariance and the variance of its real and
        imaginary parts (and their cross-covariance).
        """
        if other is None:
            centered = self._centered_obs
            partial = _get_covar_var_re_im_partial(centered, centered, self.weights)
        else:
            partial = _get_covar_var_re_im_partial(self._centered_obs, other._centered_obs, self.weights)

        return _finalize_covar_var_re_im(*partial)
    
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

        Args:
            * ``other`` [optional]: Another instance of `SampledObs` or `LazySampledObs`.

        Returns:
            Tuple ``(covar, var_re, var_im, cov_re_im)``.
        """
        mean = self.mean

        if other is None:
            other_mean = mean
        elif isinstance(other, (SampledObs, LazySampledObs)):
            other_mean = other.mean
        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )

        def _accumulate(partial, batch_1, batch_2, weights, self_covar):
            centered_1 = _center(batch_1, mean)
            centered_2 = centered_1 if self_covar else _center(batch_2, other_mean)
            batch_partial = _get_covar_var_re_im_partial(centered_1, centered_2, weights)
            if partial is None:
                return batch_partial
            return tuple(p + b for p, b in zip(partial, batch_partial))

        partial = None

        if other is None:
            for batch, weights in zip(self._observations, self._weights):
                partial = _accumulate(partial, batch, batch, weights, self_covar=True)

        elif isinstance(other, SampledObs):
            other_batches = _reshape_in_batches(other.observations, self._batch_size)

            for batch, weights, batch_other in zip(self._observations, self._weights, other_batches):
                partial = _accumulate(partial, batch, batch_other, weights, self_covar=False)

        else:
            for batch_1, weights_1, batch_2 in zip(self._observations, self._weights, other._observations):
                partial = _accumulate(partial, batch_1, batch_2, weights_1, self_covar=False)

        return _finalize_covar_var_re_im(*partial)

    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> LazySampledObs:
        if linear_map is not None:
            raise NotImplementedError(
                "A linear map can't be applied without materializing all the observables"
            )
        
        jitted_fn = jax.jit(element_wise_fn)
        iterable = self._observations
        transormed_iterable = lambda: (jitted_fn(batch) for batch in iterable)
        self.observations = SizedIterable(transormed_iterable, iterable.n_iterations, iterable.batch_size)