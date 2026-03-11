from jVMC_exp.optimizer.base import Evolution

class SR(Evolution):
    def __init__(
            self, sampler, psi, stepper, 
            output_manager=None, use_cross_valiadation=False, diagonalizeOnDevice=True, 
            snrTol=2, pinvTol=1e-14, pinvCutoff=1e-8, diagonalShift=0.001
        ):
        super().__init__(
            sampler, psi, stepper, True, True, 
            output_manager, use_cross_valiadation, diagonalizeOnDevice, 
            snrTol, pinvTol, pinvCutoff, diagonalShift
        )