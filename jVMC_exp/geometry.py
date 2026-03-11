import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod

class AbstractGeometry(ABC):
    def __init__(self, n_particles, n_dim, PBC=True, extent=None):
        self._n_particles = n_particles
        self._n_dim = n_dim 

        if not isinstance(PBC, tuple):
            PBC = (PBC, ) * self.n_dim
        self._PBC = PBC

        if extent is None:
            extent = (1,) * self.n_dim
        elif not isinstance(extent, tuple):
            raise ValueError(f'Extent has to be a tuple, got {extent}.')
        self._extent = extent

    def __eq__(self, other):
        if not isinstance(other, AbstractGeometry):
            return NotImplemented
        return (
            self.n_particles == other.n_particles and
            self.n_dim == other.n_dim and
            self.PBC == other.PBC and
            self.extent == other.extent
        )

    @property
    def n_particles(self):
        return self._n_particles
    
    @property
    def n_dim(self):
        return self._n_dim
    
    @property
    def PBC(self):
        return self._PBC
    
    @property
    def extent(self):
        return self._extent
    
    def get_absolute_distance(self, x):
        """
        Given a configuration x, returns a vector with shape (n_particles * (n_particles - 1) / 2,)
        with all pairwise distances between particles, using Minimum Image Convention.
        """
        diff = self.get_distance(x)
        sqrd_dist = jnp.sum(diff**2, axis=-1)

        return jnp.sqrt(sqrd_dist[jnp.triu_indices(self.n_particles, 1)])
    
    @abstractmethod
    def domain(self):
        """
        Return the domain of the geometry.
        """
        pass
    
    @abstractmethod
    def apply_PBC(self, x):
        """
        Apply PBC where needed given the geometry of the system.
        """
        pass

    @abstractmethod
    def uniform_populate(self, key):
        """
        Generate configuration of the particles with positions sampled from a uniform distribution.
        """
        pass

    @abstractmethod
    def get_distance(self, x):
        """
        Given a configuration x, returns a matrix with shape (n_particles, n_particles, n_dim)
        with all pairwise distances between particles, using Minimum Image Convention.
        """

class HyperRectangle(AbstractGeometry):
    @property
    def domain(self):
        ext = jnp.array(self.extent)

        return jnp.stack([-ext / 2, ext / 2], axis=1)
    
    def apply_PBC(self, x):
        extent = jnp.tile(jnp.array(self.extent), self.n_particles)
        PBC = jnp.tile(jnp.array(self.PBC, dtype=jnp.bool_), self.n_particles)

        return jnp.where(PBC, ((x + extent/2) % extent) - extent/2, x)
    
    def uniform_populate(self, key, shape, dtype=jnp.float64):
        low = self.domain[:, 0]
        samples = jax.random.uniform(key, shape, dtype=dtype)

        low_flat = jnp.repeat(low, self.n_particles)
        extent_flat = jnp.repeat(jnp.array(self.extent), self.n_particles)

        return self.apply_PBC(low_flat + samples * extent_flat)
    
    def get_distance(self, x):
        x = x.reshape((self.n_particles, self.n_dim))
        diff = x[:, None, :] - x[None, :, :]
        extent = jnp.array(self.extent)
        PBC = jnp.array(self.PBC, dtype=bool)

        return diff - PBC * extent * jnp.round(diff / extent)