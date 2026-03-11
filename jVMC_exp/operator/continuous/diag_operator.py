from jVMC_exp.operator.continuous.base import Operator
import jax.numpy as jnp    

class PotentialOperator(Operator):
    def __init__(self, geometry, potential):
        super().__init__(geometry, is_diagonal=True)

        if not callable(potential):
            raise ValueError("The property potential has to be a function.")
        self._potential = potential
    
    @property
    def potential(self):
        return self._potential
        
    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return self.potential(s)
    
class CoulombInteraction(Operator):
    def __init__(self, geometry, charge=None):
        super().__init__(geometry, is_diagonal=True)
        
        if charge is not None:
            if hasattr(charge, '__len__'):
                if len(charge) != self.geometry.n_particles:
                    raise ValueError(f'The number of charges ({len(charge)}) does not match the number of particles ({self.geometry.n_particles}).')
                interaction_charge = charge[None, :] * charge[:, None] 
                self._interaction_charge = interaction_charge[jnp.triu_indices(len(charge), 1)]
            else:
                self._interaction_charge = charge
        else:
            self._interaction_charge = 1

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return jnp.sum(self._interaction_charge / self.geometry.get_absolute_distance(s))