from .base import Evolution
from jVMC_exp.solver.pinv_snr import PinvSNR

class TDVP(Evolution):
    def __init__(
            self, sampler, psi, make_real: bool,
            use_cross_valiadation=False, diagonalShift=0.001, solver=PinvSNR()
        ):
        super().__init__(
            sampler, psi, False, make_real, 
            use_cross_valiadation, diagonalShift, solver
        )