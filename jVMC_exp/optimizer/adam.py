from optax import adam
import tqdm
from typing import Dict

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput
from jVMC_exp.objective_function.base import AbstractObjectiveFunction
from jVMC_exp.util import ObservableEntry
from jVMC_exp.stepper import Euler

class Adam(AbstractOptimizer):
    def __init__(
            self, sampler, psi,
            learning_rate: float=1e-3, b1: float=0.9, b2: float=0.999, eps: float=1e-8
    ):
        super().__init__(sampler, psi, False)

        self._adam = adam(learning_rate, b1, b2, eps)
        self._opt_state = self._adam.init(self.psi.parameters_flat)

    def get_update(self, objective_function_out: ObjectiveFunctionOutput):
        grad = objective_function_out.grad.mean.squeeze()
        update, self._opt_state = self._adam.update(grad, self._opt_state, self.psi.parameters_flat)

        return update
    
    def _update_meta_data(self):
        pass

    def cross_validation(self):
        pass

    def ground_state_search(
            self,
            steps,
            objective_function: AbstractObjectiveFunction,
            observables: Dict[str, ObservableEntry] | None = None,
            save_meta_data: bool = False,
            **objective_function_kwargs
        ):
        stepper = Euler(1)

        pbar = tqdm.tqdm(range(steps))
        for n in pbar: 
            stepper.update_dt(n)
            self.psi.parameters, _ = stepper.step(
                0,
                self,
                self.psi.parameters_flat,
                objective_function=objective_function,
                **objective_function_kwargs
            )
            
            self._measure_and_store(n, observables, save_meta_data)

            pbar.set_postfix(E=f"{self.o_loc}")
  
        self.output_manager.print_timings()
        
        if save_meta_data:
            return self.output_manager.data['observables'], self.output_manager.data['metadata']
        
        return self.output_manager.data["observables"]

    def time_evolution(self):
        raise NotImplementedError