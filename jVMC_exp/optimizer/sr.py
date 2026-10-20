from typing import Dict, List, Callable

from .base import Evolution
from jVMC_exp.solver import PinvSNR
from jVMC_exp.objective_function import AbstractObjectiveFunction
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry
from jVMC_exp.util.output_manager import OutputManager

class SR(Evolution):
    def __init__(
            self, sampler, psi, 
            diagonalShift=0.001, diagonalScale=0., solver=PinvSNR(),
            resample_stepper=True, output_manager: OutputManager | None = None,
            store_qgt: bool = False
        ):
        super().__init__(
            sampler, psi, resample_stepper, True, True, 
            False, diagonalShift, diagonalScale, solver,
            output_manager=output_manager, store_qgt=store_qgt,
            compute_tdvp_error=False
        )

    def time_evolution(
            self,
            t_max,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler,
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **kwargs
        ):
        raise NotImplementedError(
            "The SR optimizer is intended for ground state search only and "
            "cannot be used for time evolution. Use jVMC_exp.optimizer.TDVP instead."
        )

    def cross_validation(self):
            pass