from optax import adam

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.optimizer.stepper import Euler
from jVMC_exp.stats import SampledObs

class Adam(AbstractOptimizer):
    def __init__(
            self, sampler, psi, output_manager = None,
            learning_rate: float=1e-3, b1: float=0.9, b2: float=0.999, eps: float=1e-8
    ):
        stepper = Euler(1.)
        super().__init__(sampler, psi, stepper, output_manager, False)

        self._adam = adam(learning_rate, b1, b2, eps)
        self._opt_state = self._adam.init(self.psi.parameters_flat)

    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        E_grad = gradients.get_covar(Eloc).squeeze()
        update, self._opt_state = self._adam.update(E_grad, self._opt_state, self.psi.parameters_flat)

        return update
    
    def _update_meta_data(self):
        pass

    def cross_validation(self, Eloc, gradients):
        pass