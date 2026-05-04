import jax.numpy as jnp
import jax

from jVMC_exp.operator.continuous.base import Operator
from jVMC_exp.geometry import AbstractGeometry

def grad_real_to_cpx(f, x):
    grad_re = jax.grad(lambda t: jnp.real(f(t)))(x)
    grad_im = jax.grad(lambda t: jnp.imag(f(t)))(x)
    
    return grad_re + 1j * grad_im

def laplacian(grad_f):
    def lap(x):
        jac = jax.jacfwd(grad_f)(x)
        return jnp.diag(jac)
    return lap

class TotalKineticOperator(Operator):
    def __init__(self, geometry: AbstractGeometry, mass: float | list = 1, laplacian_mode='standard'):
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
        
        if laplacian_mode not in ['standard', 'forward']:
            raise ValueError(f"Invalid laplacian_mode '{laplacian_mode}'. Supported modes are 'standard' and 'forward'.")
        self._laplacian_mode = laplacian_mode

    @property
    def mass(self):
        return self._mass

    @property
    def laplacian_mode(self):
        return self._laplacian_mode

    @property
    def _inverse_mass(self):
        return 1. / jnp.repeat(self.mass, self.geometry.n_dim)
    
    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        log_psi = lambda x: apply_fun(parameters, x)
        
        if self.laplacian_mode == 'standard':
            grad_log_psi = lambda x: grad_real_to_cpx(log_psi, x) 
            lap_log_psi = laplacian(grad_log_psi)(s)
            grad_log_psi = grad_log_psi(s)
        
        elif self.laplacian_mode == 'forward': # This no longer works with the latest version of jax
            raise NotImplementedError(
                "The 'forward' laplacian mode is currently not compatible with the latest version of jax. " \
                "Please use 'standard' mode instead."
            )
            from ._frwrd_lap_fix import lap as fwdlap
            _, grad_log_psi, lap_log_psi = fwdlap(
                log_psi,
                (s,),
                (jnp.eye(s.size, dtype=s.dtype),),
                (SymbolicZero(get_aval(s).to_tangent_aval()),)
            )
 
        laplacian_psi = lap_log_psi + grad_log_psi**2

        return - 0.5 * jnp.sum(laplacian_psi * self._inverse_mass)