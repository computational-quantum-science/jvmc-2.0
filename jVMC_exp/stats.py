from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from jVMC_exp.sharding_config import DEVICE_SHARDING, MESH

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

@jax.jit(static_argnums=(1,))
def _apply_and_project(data, apply_fn, projection):
    return jnp.matmul(projection, apply_fn(data)) 

@jax.jit
def _get_tangent_kernel(norm_data):
    return jnp.matmul(norm_data, jnp.conj(jnp.transpose(norm_data)))

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


