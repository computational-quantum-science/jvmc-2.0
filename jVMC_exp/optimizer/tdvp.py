from jVMC_exp.optimizer.base import Evolution

class TDVP(Evolution):
    def __init__(
            self, sampler, psi, stepper, make_real: bool,
            output_manager=None, use_cross_valiadation=False, diagonalizeOnDevice=True, 
            snrTol=2, pinvTol=1e-14, pinvCutoff=1e-8, diagonalShift=0.001
        ):
        super().__init__(
            sampler, psi, stepper, False, make_real, 
            output_manager, use_cross_valiadation, diagonalizeOnDevice, 
            snrTol, pinvTol, pinvCutoff, diagonalShift
        )