import jax.numpy as jnp
import jax

from jVMC.operator.continuous.base import Operator
from jVMC.geometry import AbstractGeometry

def laplacian(grad_f):
    def lap(x):
        basis_vectors = jnp.eye(len(x))
        def hessian_diag_element(v):
            return jnp.dot(jax.jvp(grad_f, (x,), (v,))[1], v)
        # Vectorize across dimensions of grad_f
        return jax.vmap(hessian_diag_element)(basis_vectors)
    return lap

class TotalKineticOperator(Operator):
    def __init__(self, geometry: AbstractGeometry, mass: float | list = 1):
        super().__init__(geometry, is_diagonal=False)

        if isinstance(mass, complex):
            raise ValueError("The property 'mass' can not be complex.")
        elif isinstance(mass, (int, float)):
            self._mass = jnp.repeat(mass, self.geometry.n_particles)
        elif isinstance(mass, (list, tuple)) or hasattr(mass, '__len__'):
            if len(mass) != geometry.n_particles:
                raise ValueError(f"Got {len(mass)} masses which is not the same as the number of particles ({geometry.n_particles}).")
            self._mass = jnp.asarray(mass)
        else:
            raise ValueError(f"'mass' must be a scalar or array-like, got {type(mass)}.")
            
    @property
    def mass(self):
        return self._mass
    
    @property
    def _inverse_mass(self):
        return 1. / jnp.repeat(self.mass, self.geometry.n_dim)
    
    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        log_psi = lambda x: apply_fun(parameters, x)
        grad_log_psi = jax.grad(log_psi) 
        lap_log_psi = laplacian(grad_log_psi)(s)
        grad_log_psi = grad_log_psi(s)
        laplacian_psi = jnp.real(lap_log_psi + grad_log_psi * grad_log_psi.conj())

        return - 0.5 * jnp.sum(laplacian_psi * self._inverse_mass)