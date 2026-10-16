from typing import Dict, List, Callable
import jax.numpy as jnp

from .base import Evolution
from jVMC_exp.solver import PinvSNR
from jVMC_exp.objective_function import AbstractObjectiveFunction
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry
from jVMC_exp.util.output_manager import OutputManager

class TDVP(Evolution):
    def __init__(
            self, sampler, psi, make_real: bool, resample_stepper=True,
            use_cross_valiadation=False, diagonalShift=0., diagonalScale=0.,
            solver=PinvSNR(), store_qgt: bool = True, compute_tdvp_error: bool = True,
            output_manager: OutputManager | None = None
        ):
        if use_cross_valiadation:
            if not store_qgt:
                raise ValueError(
                    "Cross validation requires the QGT to be stored, "
                    "but store_qgt is set to False."
                )
            if not compute_tdvp_error:
                raise ValueError(
                    "Cross validation requires the TDVP error to be computed, "
                    "but compute_tdvp_error is set to False."
                )

        super().__init__(
            sampler, psi, resample_stepper, False, make_real, 
            use_cross_valiadation, diagonalShift, diagonalScale, 
            solver, output_manager=output_manager,
            store_qgt=store_qgt, compute_tdvp_error=compute_tdvp_error
        )

    def ground_state_search(
            self,
            steps,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler(),
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **kwargs
        ):
        raise NotImplementedError(
            "The SR optimizer is intended for ground state search only and "
            "cannot be used for time evolution. Use jVMC_exp.optimizer.TDVP instead."
        )

    def cross_validation(self, objective_function: AbstractObjectiveFunction, **objective_function_kwargs):
        if "residual" not in self.meta_data:
            raise ValueError(
                f"Cross validation normalizes against the residual of the solve, but solver "
                f"'{type(self.solver).__name__}' does not report a 'residual' in the info "
                "dictionary it returns. Either use a solver that does (e.g. Pinv or PinvSNR), "
                "or construct the optimizer with use_cross_valiadation=False."
            )

        residual = self.meta_data["residual"]
        tvp_error = self.meta_data["tdvp_error"]

        full_samples, full_logPsi, full_weights = (
            self.sampler.samples, 
            self.sampler.logPsi, 
            self.sampler.weights
        )

        def _value_and_grad_on_subset(start):
            self.sampler._samples = full_samples[start::2]
            self.sampler._logPsi = full_logPsi[start::2]
            w = full_weights[start::2]
            self.sampler._weights = w / jnp.sum(w)

            return objective_function.value_and_grad(
                self.sampler, compute_grad=True, **objective_function_kwargs
            )

        try:
            objective_fn_out_1 = _value_and_grad_on_subset(0)
            objective_fn_out_2 = _value_and_grad_on_subset(1)
        finally:
            self.sampler._samples, self.sampler._logPsi, self.sampler._weights = (
                full_samples, 
                full_logPsi, 
                full_weights
            )

        update_1 = self.get_update(objective_fn_out_1)
        validation_tdvp_err = self._get_tdvp_error(
            update_1, objective_fn_out_1.grad, objective_fn_out_1.o_loc.var
        )
        if self.psi.holomorphic:
            update_1 = self._make_cmplx_fn(update_1)
            objective_fn_out_2.grad_log_psi.transform(self._remove_double_trans)
            objective_fn_out_2.grad = self._remove_double_fn(objective_fn_out_2.grad)
        F2 = self._get_rhs(objective_fn_out_2.grad)
        S2 = self._get_lhs(objective_fn_out_2.grad_log_psi)
        if callable(S2):
            Sv = S2(update_1)
        else:
            Sv = S2.dot(jnp.pad(update_1, (0, self._params_pad_size)))
            Sv = Sv[:-self._params_pad_size] if self._params_pad_size else Sv
        validation_residual = (jnp.linalg.norm(Sv - F2) / jnp.linalg.norm(F2)) / residual

        crossValidationFactor_residual = validation_residual
        crossValidationFactor_tdvpErr = validation_tdvp_err / tvp_error
        self.meta_data["tdvp_residual_cross_validation_ratio"] = crossValidationFactor_residual
        self.meta_data["tdvp_error_cross_validation_ratio"] = crossValidationFactor_tdvpErr