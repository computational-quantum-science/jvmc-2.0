from typing import Dict

from .base import Evolution
from jVMC_exp.solver.pinv_snr import PinvSNR
from jVMC_exp.objective_function import AbstractObjectiveFunction
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry
from jVMC_exp.util.output_manager import OutputManager

class SR(Evolution):
    def __init__(
            self, sampler, psi, 
            use_cross_valiadation=False, diagonalShift=0.001, diagonalScale=0., solver=PinvSNR(),
            output_manager: OutputManager | None = None,
            jacobian_mode: str = "dense",
        ):
        super().__init__(
            sampler, psi, True, True, 
            use_cross_valiadation, diagonalShift, diagonalScale, solver,
            output_manager=output_manager,
            jacobian_mode=jacobian_mode,
        )

    def time_evolution(
            self,
            t_max,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler,
            observables: Dict[str, ObservableEntry] | None = None,
            save_meta_data: bool = False,
            **kwargs
        ):
        raise NotImplementedError(
            "The SR optimizer is intended for ground state search only and "
            "cannot be used for time evolution. Use jVMC_exp.optimizer.TDVP instead."
        )
