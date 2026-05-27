import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import tqdm
from typing import Dict

from jVMC_exp.stats import SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler, ExactSampler
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry, measure
from jVMC_exp.solver.base import AbstractSolver, SolverState
from jVMC_exp.solver.pinv_snr import PinvSNR
from jVMC_exp.objective_function.base import AbstractObjectiveFunction, ObjectiveFunctionOutput

@jax.jit 
def real_fn(x):
    return jnp.real(x)

@jax.jit
def imag_fn(x):
    return 1j * jnp.imag(x)

class AbstractOptimizer(ABC):
    def __init__(self, sampler:AbstractMCSampler, psi: NQS, use_cross_valiadation: bool=False):
        self._sampler = sampler
        self._psi = psi
        self.use_cross_valiadation = use_cross_valiadation
        self.meta_data = {}
        self._output_manager = OutputManager()
        self.o_loc = None
        self._elapsed = 0

    @property
    def output_manager(self):
        return self._output_manager

    @property
    def sampler(self):
        return self._sampler
    
    @property
    def psi(self):
        return self._psi
    
    def __call__(
            self, parameters, t, *, numSamples=None, intStep=None,
            objective_function: AbstractObjectiveFunction, **objective_function_kwargs
    ):
        """ 
        Arguments:
            * ``parameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.
        """
        tmp_parameters = self.psi.parameters
        self.psi.parameters = parameters
        self._elapsed = 0
        
        def stop_timing(name, wait_for=None):
            if wait_for is not None:
                jax.block_until_ready(wait_for)
            return self.output_manager.stop_timing(name)

        # Get sample
        self.output_manager.start_timing("sampling")
        samples, _, _ = self.sampler.sample(numSamples=numSamples)
        self._elapsed += stop_timing("sampling", wait_for=samples)

        # Evaluate local observables and their gradient 
        self.output_manager.start_timing("compute objective function and gradient")
        objective_fn_out = objective_function.value_and_grad(self.sampler, t=t, **objective_function_kwargs)
        self._elapsed += stop_timing("compute objective function and gradient", wait_for=objective_fn_out.grad_log_psi.observations)

        # Obtain the update from the gradients
        self.output_manager.start_timing("solve")
        update = self.get_update(objective_fn_out)
        self._elapsed += stop_timing("solve")

        self.psi.parameters = tmp_parameters

        if intStep is not None:
            if intStep == 0:
                self.o_loc = objective_fn_out.o_loc
                self._update_meta_data()

                if self.use_cross_valiadation:
                    self.output_manager.start_timing("cross_validation")
                    self.cross_validation(objective_fn_out)
                    self._elapsed += stop_timing("cross_validation")

        return update
    
    def ground_state_search(
            self,
            steps,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler(),
            observables: Dict[str, ObservableEntry] | None = None,
            save_meta_data: bool = False,
            **objective_function_kwargs
        ):
        if not hasattr(stepper, "update_dt"):
            raise ValueError("For ground state search the stepper must " \
                            f"implement a mehod called 'update_dt'")

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

    def time_evolution(
            self,
            t_max,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler,
            observables: Dict[str, ObservableEntry] | None = None,
            save_meta_data: bool = False,
            **objective_function_kwargs
        ):

        pbar = tqdm.tqdm(total=t_max)
        t = 0
        while t < t_max:
            self.psi.parameters, dt = stepper.step(
                t,
                self,
                self.psi.parameters_flat,
                objective_function=objective_function,
                **objective_function_kwargs
            )

            dt = float(dt)
            if t + dt >= t_max:
                dt = t_max - t
            t += dt

            self.meta_data['dt'] = dt
            self._measure_and_store(t, observables, save_meta_data)

            pbar.update(1)
            pbar.set_postfix({
                "t": f"{t:.4f}/{t_max}",
                "dt": f"{dt:.2e}",
                "ETA": f"{(t_max - t) / dt * self._elapsed:.1f}s",
                'Progress': f'{int(t / t_max * 100)}%',
                'E': f"{self.o_loc}",
            })

        pbar.close()
        self.output_manager.print_timings()

        if save_meta_data:
            return self.output_manager.data['observables'], self.output_manager.data['metadata']
        
        return self.output_manager.data["observables"]

    @abstractmethod
    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        '''
        Return the update and write self.update and self._additional_info
        '''
        self.update = None
        self._additional_info = None

    @abstractmethod
    def cross_validation(self, objective_function_output: ObjectiveFunctionOutput):
        pass

    @abstractmethod
    def _update_meta_data(self):
        '''
        Update the dictionary self.meta_data
        '''
        self.meta_data = {}

    def _measure_and_store(self, t, observables, save_meta_data):
        measures = {}
        energy = dict(
            mean = jnp.real(self.o_loc.mean).item(),
            variance = jnp.real(self.o_loc.var).item(),
            MC_error = jnp.real(self.o_loc.error_of_mean).item()
        )

        if observables is not None:
            measures = measure(observables, self.sampler)
        self.output_manager.write_observables(t, energy=energy, **measures)

        if save_meta_data:
            self.output_manager.write_metadata(t, **self.meta_data)

class Evolution(AbstractOptimizer):
    def __init__(
            self, sampler, psi, 
            imag_time: bool, make_real: bool, use_cross_valiadation: bool=False, 
            diagonal_shift: float=1e-3, solver: AbstractSolver=PinvSNR()
        ):
        self.rhsPrefactor = 1 if imag_time else 1j
        self._lhs_trans_fn = real_fn if make_real else imag_fn
        self._rhs_trans_fn = lambda x: self._lhs_trans_fn((- self.rhsPrefactor) * x)
        self.diagonal_shift = diagonal_shift
        self._solver = solver
        self._get_lhs = self._get_lhs_dense if solver._needs_dense_matrix else self._get_lhs_lazy
       
        super().__init__(sampler, psi, use_cross_valiadation)

        self._solver_state = SolverState(
            covar_grad_o_loc=lambda: self._covar_grad_o_loc,
            rhs_trans_fn=self._rhs_trans_fn,
            exact_sampler=isinstance(self.sampler, ExactSampler)
        )

        self._F0 = None
        self._S0 = None

    @property
    def solver(self):
        return self._solver
    
    @property
    def solver_state(self):
        return self._solver_state
    
    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        grad = objective_function_output.grad
        grad_log_psi = objective_function_output.grad_log_psi
        self._covar_grad_o_loc = grad

        S = self._get_lhs(grad_log_psi)
        F = self._get_rhs(grad)   
        self.update, self._additional_info = self.solver(S, F, self.solver_state)

        return self.update
    
    # TODO: Test if this actually works (tdvp error has to be handled in a different way)
    def cross_validation(self, objective_function_output: ObjectiveFunctionOutput):
        o_loc_1 = objective_function_output.o_loc.get_subset(start=0, step=2)
        grad_1 = objective_function_output.grad.get_subset(start=0, step=2)
        grad_log_psi_1 = objective_function_output.grad_log_psi.get_subset(start=0, step=2)
        grad_2 = objective_function_output.grad.get_subset(start=1, step=2)
        grad_log_psi_2 = objective_function_output.grad_log_psi.get_subset(start=1, step=2)

        update_1 = self.get_update(o_loc_1, grad_1, grad_log_psi_1)
        F2 = self._get_rhs(grad_2)
        S2 = self._get_lhs(grad_log_psi_2)

        validation_tdvp_err = self._get_tdvp_error(update_1)
        _ = self.get_update(objective_function_output)
        Sv = S2(update_1) if callable(S2) else S2.dot(update_1)
        validation_residual = (jnp.linalg.norm(Sv - F2) / jnp.linalg.norm(F2)) / self._additional_info["residual"]

        crossValidationFactor_residual = validation_residual
        crossValidationFactor_tdvpErr = validation_tdvp_err / self.meta_data["tdvp_error"]
        self.meta_data["tdvp_residual_cross_validation_ratio"] = crossValidationFactor_residual
        self.meta_data["tdvp_error_cross_validation_ratio"] = crossValidationFactor_tdvpErr

    def _get_tdvp_error(self, update):
        Sv = self._S0(update) if callable(self._S0) else self._S0.dot(update)

        return jnp.abs(1. + jnp.real(update.dot(Sv) - 2. * jnp.real(update.dot(self._F0))) / (self.o_loc.var + 1e-10))
    
    def _get_lhs_dense(self, grad_log_psi: SampledObs):
        '''
        Returns left hand side of the TDVP equation
        '''
        self._S0 = grad_log_psi.get_covar()
        S = self._lhs_trans_fn(self._S0)

        if self.diagonal_shift > 1e-15:
            S = S + jnp.diag(self.diagonal_shift * jnp.diag(S))

        return S
    
    def _get_lhs_lazy(self, grad_log_psi: SampledObs):
        '''
        Returns a function that computes the matrix vector product with the left hand side of the TDVP equation
        '''
        O = grad_log_psi._normalized_obs

        def raw_matvec(v):
            return (O.conj().T @ (O @ v))
        self._S0 = raw_matvec

        def matvec(v):
            Sv = self._lhs_trans_fn(raw_matvec(v))
            if self.diagonal_shift > 1e-15:
                diag = jnp.sum(jnp.abs(O) ** 2, axis=0)
                Sv = Sv + self.diagonal_shift * diag * v

            return Sv

        return matvec 
    
    def _get_rhs(self, grad: SampledObs):
        self._F0 = - self.rhsPrefactor * grad.mean.ravel()
        F = self._lhs_trans_fn(self._F0)
        F.block_until_ready()

        return F

    def _update_meta_data(self):
        self.meta_data = dict(
            tdvp_error=self._get_tdvp_error(self.update).item(), 
            **self._additional_info
        )