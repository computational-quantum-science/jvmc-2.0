from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from jVMC_exp.sharding_config import (
    DEVICE_SHARDING,
    REPLICATED_SHARDING,
    DEVICE_SPEC,
    REPLICATED_SPEC,
    MESH,
    sharded,
)

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
def _get_covar(norm_data_1, norm_data_2):
    return jnp.tensordot(jnp.conj(norm_data_1), norm_data_2, axes=(0, 0))

@jax.jit
@jax.vmap
def _get_covar_per_sample(centered_data_1, centered_data_2):
    return jnp.outer(jnp.conj(centered_data_1), centered_data_2)

@jax.jit
def _get_covar_var(centered_data_1, centered_data_2, weights):
    covar_per_sample = _get_covar_per_sample(centered_data_1, centered_data_2)
    covar = jnp.tensordot(weights, covar_per_sample, axes=(0, 0))
    covar_sqrd = jnp.tensordot(weights, jnp.abs(covar_per_sample) ** 2, axes=(0, 0))

    return covar_sqrd - jnp.abs(covar) ** 2

@partial(jax.jit, static_argnums=(1,))
def _apply_and_project(data, apply_fn, projection):
    return jnp.matmul(projection, apply_fn(data)) 

@jax.jit
def _get_tangent_kernel(norm_data):
    return jnp.matmul(norm_data, jnp.conj(jnp.transpose(norm_data)))

@jax.jit
def _set_matrix_block(matrix, block, row_start, col_start):
    return jax.lax.dynamic_update_slice(matrix, block, (row_start, col_start))

@jax.jit
def _covar_obs_block(norm_grad, norm_obs):
    return jax.vmap(jnp.outer)(jnp.conj(norm_grad), norm_obs)

@partial(jax.jit, static_argnums=(1, 2))
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
        
        remainder = self._num_samples % num_devices
        if remainder != 0:
            num_pad = num_devices - remainder
            pad = ((0, num_pad),) + ((0, 0),) * (observations.ndim - 1)
            observations = jnp.pad(observations, pad, mode='constant')
            weights = jnp.pad(weights, (0, num_pad), constant_values=0)
        
        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self._num_obs == 1:
            return f"{self.mean.item():.4e} ± {self.error_of_mean.item():.4e} (Var = {self.var.item():.4e})"
        else:
            return f"SampledObs with {self._num_obs} features"

    @property
    def observations(self):
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
        Returns the variance of the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        
        return _get_covar_var(self._centered_obs, other._centered_obs, self.weights)
    
    def get_covar_obs(self, other: SampledObs | None = None) -> SampledObs:
        """
        Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """
        if other is None:
            other = self
        covar_per_sample = _get_covar_per_sample(self._centered_obs, other._centered_obs)

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

        B = jnp.sum((chain_obs.mean.real - self.mean.real)**2) / (n_chains - 1)
        W = jnp.mean(chain_obs.var) 
        
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
        Apply a transformation to observations and return a new SampledObs.
            
        The transformation is applied in two stages:
        1. Element-wise function applied to each observation
        2. (Optional) Linear projection via matrix multiplication
        
        Args:
            element_wise_fun: Function applied element-wise to each observation.
            linear_map: Optional linear transformation matrix applied after element_wise_fun.
        
        Returns:
            SampledObs: New instance with transformed observations.
        """
        if linear_map is not None:
            new_obs = _apply_and_project(self.observations, element_wise_fn, linear_map)
        else:
            new_obs = jax.jit(element_wise_fn)(self.observations)
        
        return SampledObs(new_obs, self.weights)
    
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


class BatchedJacobian:
    """
    Reusable, batched representation of sampled network logarithmic
    derivatives.

    The object stores samples and weights, but not the full sample-parameter
    Jacobian. Jacobian blocks are recomputed from ``psi.gradients`` whenever
    they are requested. Small reductions such as the weighted mean and diagonal
    are cached after first use.
    """
    def __init__(
            self,
            psi,
            samples,
            weights=None,
            batch_size: int | None = None,
            block_transform=lambda x: x,
            real_imag_doubled: bool = False,
        ):
        self.psi = psi
        self._block_transform = block_transform
        self._real_imag_doubled = bool(real_imag_doubled)

        samples = jnp.asarray(samples)
        self._raw_samples = samples
        self._num_samples = int(samples.shape[0])
        self._sample_shape = samples.shape[1:]

        if weights is None:
            weights = jnp.ones(self._num_samples, dtype=psi.out_dtype) / self._num_samples
        elif weights.shape != (self._num_samples,):
            raise ValueError(f"Weights must have shape ({self._num_samples},), got {weights.shape}")
        self._raw_weights = jnp.asarray(weights)

        num_devices = MESH.shape["devices"]
        remainder = self._num_samples % num_devices
        self._num_pad = (num_devices - remainder) if remainder != 0 else 0
        if self._num_pad > 0:
            sample_pad = ((0, self._num_pad),) + ((0, 0),) * (samples.ndim - 1)
            samples = jnp.pad(samples, sample_pad, mode="constant")
            weights = jnp.pad(weights, (0, self._num_pad), constant_values=0)

        if batch_size is None:
            batch_size = psi.batchSize
        if batch_size % num_devices != 0:
            raise ValueError(
                f"The batch size ({batch_size}) has to be divisible by the "
                f"number of devices ({num_devices})"
            )
        self.batch_size = int(batch_size)

        self._samples = jax.device_put(samples, DEVICE_SHARDING)
        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._padded_num_samples = int(samples.shape[0])
        self._slices = tuple(
            slice(start, min(start + self.batch_size, self._padded_num_samples))
            for start in range(0, self._padded_num_samples, self.batch_size)
        )
        self._mean = None
        self._diagonal = None
        self._num_obs = None
        self._dtype = None

    @property
    def samples(self):
        return self._samples

    @property
    def weights(self):
        return self._weights

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._compute_mean()
        return self._mean

    @property
    def sync_target(self):
        return self.mean

    @property
    def num_blocks(self):
        return len(self._slices)

    @property
    def num_effective_samples(self):
        factor = 2 if self._real_imag_doubled else 1
        return factor * self._padded_num_samples

    @property
    def _normalized_obs(self):
        if not self._real_imag_doubled:
            return self.materialize()._normalized_obs
        return jnp.concatenate(tuple(self.iter_normalized_blocks()), axis=0)

    def _slice(self, idx):
        return self._slices[int(idx)]

    def _effective_slice(self, idx):
        sl = self._slice(idx)
        if not self._real_imag_doubled:
            return sl
        return slice(2 * sl.start, 2 * sl.stop)

    def _block_output(self, block):
        if self._num_obs is None:
            self._num_obs = int(block.shape[-1])
            self._dtype = block.dtype
        return block

    def _compute_block(self, samples):
        return self._block_transform(self.psi.gradients(samples))

    def block(self, idx):
        sl = self._slice(idx)
        return self._block_output(self._compute_block(self.samples[sl]))

    def iter_blocks(self):
        for idx in range(self.num_blocks):
            yield self.block(idx)

    def normalized_block(self, idx):
        sl = self._slice(idx)
        block = self.block(idx)
        block = self._center_weighted_block_sh(
            block,
            self.weights[sl],
            mean=self.mean,
            batch_size=int(block.shape[0]),
        )
        if self._real_imag_doubled:
            return jnp.concatenate([jnp.real(block), jnp.imag(block)], axis=0)
        return block

    def iter_normalized_blocks(self):
        for idx in range(self.num_blocks):
            yield self.normalized_block(idx)

    def materialize(self):
        observations = jnp.concatenate(tuple(self.iter_blocks()), axis=0)[:self._num_samples]
        return SampledObs(observations, self._raw_weights)

    def get_subset(self, start=None, end=None, step=None):
        sl = slice(start, end, step)
        weights = self._raw_weights[sl]
        return BatchedJacobian(
            self.psi,
            self._raw_samples[sl],
            weights / jnp.sum(weights),
            batch_size=self.batch_size,
            block_transform=self._block_transform,
            real_imag_doubled=self._real_imag_doubled,
        )

    def transform(self, element_wise_fn=lambda x: x, linear_map=None):
        def block_transform(x):
            x = self._block_transform(x)
            if linear_map is not None:
                return _apply_and_project(x, element_wise_fn, linear_map)
            return jax.jit(element_wise_fn)(x)

        return BatchedJacobian(
            self.psi,
            self._raw_samples,
            self._raw_weights,
            batch_size=self.batch_size,
            block_transform=block_transform,
            real_imag_doubled=self._real_imag_doubled,
        )

    def real_imag_doubled(self):
        out = BatchedJacobian(
            self.psi,
            self._raw_samples,
            self._raw_weights,
            batch_size=self.batch_size,
            block_transform=self._block_transform,
            real_imag_doubled=True,
        )
        out._mean = self._mean
        return out

    def get_covar(self, other=None):
        if other is None:
            out = None
            for block in self.iter_normalized_blocks():
                contribution = jnp.conj(jnp.transpose(block)) @ block
                out = contribution if out is None else out + contribution
            return out

        if isinstance(other, BatchedJacobian):
            out = None
            for left, right in zip(self.iter_normalized_blocks(), other.iter_normalized_blocks()):
                contribution = jnp.conj(jnp.transpose(left)) @ right
                out = contribution if out is None else out + contribution
            return out

        other_norm = other._normalized_obs
        out = None
        for idx, block in enumerate(self.iter_normalized_blocks()):
            sl = self._slice(idx)
            contribution = jnp.conj(jnp.transpose(block)) @ other_norm[sl]
            out = contribution if out is None else out + contribution
        return out

    def get_covar_obs(self, other: SampledObs | None = None) -> SampledObs:
        if other is None:
            return self.materialize().get_covar_obs()

        other_norm = other._normalized_obs
        blocks = []
        for idx, block in enumerate(self.iter_normalized_blocks()):
            sl = self._slice(idx)
            blocks.append(_covar_obs_block(block, other_norm[sl]))

        observations = jnp.concatenate(tuple(blocks), axis=0)[:self._num_samples]
        return SampledObs(observations, self._raw_weights)

    def tangent_kernel(self):
        first_block = jax.block_until_ready(self.normalized_block(0))
        tangent = jax.device_put(
            jnp.zeros((self.num_effective_samples, self.num_effective_samples), dtype=first_block.dtype),
            DEVICE_SHARDING,
        )

        for idx_i in range(self.num_blocks):
            sl_i = self._effective_slice(idx_i)
            left = first_block if idx_i == 0 else jax.block_until_ready(self.normalized_block(idx_i))
            for idx_j in range(idx_i, self.num_blocks):
                sl_j = self._effective_slice(idx_j)
                right = left if idx_i == idx_j else jax.block_until_ready(self.normalized_block(idx_j))
                block = self._tangent_cross_block_sh(
                    left,
                    right=jax.device_put(right, REPLICATED_SHARDING),
                    batch_size=int(left.shape[0]),
                )
                tangent = _set_matrix_block(tangent, block, sl_i.start, sl_j.start)

                if idx_i != idx_j:
                    lower = self._tangent_cross_block_sh(
                        right,
                        right=jax.device_put(left, REPLICATED_SHARDING),
                        batch_size=int(right.shape[0]),
                    )
                    tangent = _set_matrix_block(tangent, lower, sl_j.start, sl_i.start)
                tangent = jax.block_until_ready(tangent)

        return tangent

    def normalized_observable(self, obs: SampledObs):
        obs = obs._normalized_obs.reshape(-1)
        if not self._real_imag_doubled:
            return obs[:self.num_effective_samples]

        blocks = []
        for idx in range(self.num_blocks):
            sl = self._slice(idx)
            block = obs[sl]
            blocks.append(jnp.concatenate([jnp.real(block), jnp.imag(block)], axis=0))
        return jnp.concatenate(tuple(blocks), axis=0)

    def diagonal(self):
        if self._diagonal is None:
            diagonal = None
            for block in self.iter_normalized_blocks():
                contribution = jnp.sum(jnp.abs(block) ** 2, axis=0)
                diagonal = contribution if diagonal is None else diagonal + contribution
            self._diagonal = diagonal
        return self._diagonal

    def matvec(self, v):
        out = None
        for block in self.iter_normalized_blocks():
            contribution = jnp.conj(jnp.transpose(block)) @ (block @ v)
            out = contribution if out is None else out + contribution
        return out

    def conj_transpose_matvec(self, v):
        out = None
        for idx, block in enumerate(self.iter_normalized_blocks()):
            sl = self._effective_slice(idx)
            contribution = self._contract_block_sh(
                block,
                v[sl],
                batch_size=None,
            )
            contribution = contribution[0] if contribution.ndim > 1 else contribution
            out = contribution if out is None else out + contribution
        return out

    def _compute_mean(self):
        mean = None
        for idx in range(self.num_blocks):
            sl = self._slice(idx)
            block = jax.block_until_ready(self.block(idx))
            if mean is None:
                mean = jax.device_put(jnp.zeros((block.shape[-1],), dtype=block.dtype), REPLICATED_SHARDING)
            contribution = self._mean_block_sh(
                block,
                self.weights[sl],
                batch_size=None,
            )
            contribution = contribution[0] if contribution.ndim > 1 else contribution
            mean = jax.block_until_ready(mean + contribution)
        return mean

    @sharded(use_vmap=False, in_specs=(DEVICE_SPEC, DEVICE_SPEC), out_specs=REPLICATED_SPEC)
    def _mean_block_sh(self, block, weights, *, batch_size):
        return jax.lax.psum(jnp.tensordot(weights, block, axes=(0, 0)), "devices")

    @sharded(use_vmap=False, in_specs=(DEVICE_SPEC, DEVICE_SPEC))
    def _center_weighted_block_sh(self, block, weights, *, mean, batch_size):
        return (block - mean[None, :]) * jnp.sqrt(weights)[:, None]

    @sharded(use_vmap=False, in_specs=(DEVICE_SPEC,))
    def _tangent_cross_block_sh(self, left, *, right, batch_size):
        return left @ jnp.conj(jnp.transpose(right))

    @sharded(use_vmap=False, in_specs=(DEVICE_SPEC, DEVICE_SPEC), out_specs=REPLICATED_SPEC)
    def _contract_block_sh(self, block, v, *, batch_size):
        return jax.lax.psum(jnp.conj(jnp.transpose(block)) @ v, "devices")
