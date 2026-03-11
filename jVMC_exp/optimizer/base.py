import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
import warnings
import tqdm
from typing import Dict
import os

from jVMC_exp.stats import SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler, ExactSampler
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.optimizer.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry, measure

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))
    return jnp.array(e), jnp.array(V)

@jax.jit 
def real_fn(x):
    return jnp.real(x)

@jax.jit
def imag_fn(x):
    return 1j * jnp.imag(x)

class AbstractOptimizer(ABC):
    def __init__(
            self, sampler:AbstractMCSampler, psi: NQS, stepper: AbstractStepper, 
            output_manager: None | OutputManager=None, use_cross_valiadation: bool=False
        ):
        self._sampler = sampler
        self._psi = psi
        self._stepper = stepper
        self._output_manager = output_manager
        self.use_cross_valiadation = use_cross_valiadation
        self.meta_data = {}

    @property
    def output_manager(self):
        return self._output_manager
    
    @property
    def sampler(self):
        return self._sampler
    
    @property
    def psi(self):
        return self._psi
    
    def __call__(self, parameters, t, *, hamiltonian: AbstractOperator, numSamples=None, intStep=None):
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

        def start_timing(name):
            if self.output_manager is not None:
                self.output_manager.start_timing(name)

        def stop_timing(name, waitFor=None):
            if self.output_manager is not None:
                if waitFor is not None:
                    waitFor.block_until_ready()
                self.output_manager.stop_timing(name)

        # Get sample
        start_timing("sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        stop_timing("sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing("compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, self.psi, LogPsiS=sampleLogPsi, t=t)
        stop_timing("compute Eloc", waitFor=Eloc)
        self.Eloc = SampledObs(Eloc, p)

        # Evaluate gradients
        start_timing("compute gradients")
        sampleGradients = self.psi.gradients(sampleConfigs)
        stop_timing("compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs(sampleGradients, p)

        start_timing("solve")
        update = self.solve(self.Eloc, sampleGradients)
        stop_timing("solve")

        self.psi.parameters = tmp_parameters

        if intStep is not None:
            if intStep == 0:
                self.energy = self.Eloc
                self._update_meta_data()

                if self.use_cross_valiadation:
                    self.cross_validation()

        return update
    
    def ground_state_search(
            self, 
            steps, 
            hamiltonian: AbstractOperator,
            observables: Dict[str, ObservableEntry] | None = None
        ):
        if not hasattr(self._stepper, "update_dt"):
            raise ValueError("For ground state search the stepper must " \
                            f"implement a mehod called 'update_dt'")

        output_manager = self.output_manager or OutputManager("_tmp.h5")
        measures = {}

        pbar = tqdm.tqdm(range(steps))
        for n in pbar: 
            self._stepper.update_dt(n)
            self.psi.parameters, _ = self.step(hamiltonian)
            
            energy = dict(
                mean = jnp.real(self.energy.mean).item(),
                variance = jnp.real(self.energy.var).item(),
                MC_error = jnp.real(self.energy.error_of_mean).item()
            )
            if observables is not None:
                measures = measure(observables, self.psi, self.sampler)
            output_manager.write_observables(n, energy=energy, **measures)

            pbar.set_postfix(E=f"{self.energy}")
  
        results = output_manager.to_dict()
        output_manager.print_timings()
        if self.output_manager is None:
            os.remove("_tmp.h5")

        return results['observables']
        
    def time_evolution(
            self,
            t_max,
            hamiltonian: AbstractOperator,
            observables: Dict[str, ObservableEntry] | None = None
        ):
        output_manager = self.output_manager or OutputManager("_tmp.h5")
        measures = {}

        pbar = tqdm.tqdm(total=t_max)
        t = 0
        while t < t_max:
            self.psi.parameters, dt = self.step(hamiltonian)
            t += dt
            energy = dict(
                mean = jnp.real(self.energy.mean).item(),
                variance = jnp.real(self.energy.var).item(),
                MC_error = jnp.real(self.energy.error_of_mean).item()
            )
            if observables is not None:
                measures = measure(observables, self.psi, self.sampler)
            output_manager.write_observables(t, energy=energy, **measures)

            pbar.update(float(dt))

            # TODO: Decide what to print
            # pbar.set_postfix(E=f"{self.energy}")

        results = output_manager.to_dict()
        output_manager.print_timings()
        if self.output_manager is None:
            os.remove("_tmp.h5")

        return results['observables']
        
    def step(self, hamiltonian: AbstractOperator):
        """
        Returns new parameters and new time step (if changed).
        """
        return self._stepper.step(0, self, self.psi.parameters_flat, hamiltonian=hamiltonian)

    @abstractmethod
    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        '''
        Return the update.
        '''
        pass

    @abstractmethod
    def _update_meta_data(self):
        '''
        Update the dictionary self.meta_data
        '''
        pass

    @abstractmethod
    def cross_validation(self, Eloc: SampledObs, gradients: SampledObs):
        pass

class Evolution(AbstractOptimizer):
    def __init__(
            self, sampler, psi, stepper, imag_time: bool, make_real: bool,
            output_manager=None, use_cross_valiadation=False, diagonalizeOnDevice=True,
            snrTol=2, pinvTol=1e-14, pinvCutoff=1e-8, diagonalShift=1e-3
        ):
        self.snrTol = snrTol
        self.pinvTol = pinvTol
        self.pinvCutoff = pinvCutoff
        self.diagonalShift = diagonalShift
        self.rhsPrefactor = 1 if imag_time else 1j
        self.diagonalizeOnDevice = diagonalizeOnDevice
        self._lhs_trans_fn = real_fn if make_real else imag_fn
        self._rhs_trans_fn = lambda x: self._lhs_trans_fn((- self.rhsPrefactor) * x)

        super().__init__(sampler, psi, stepper, output_manager, use_cross_valiadation)

    def _get_tdvp_error(self, update):
        return jnp.abs(1. + jnp.real(update.dot(self._S0.dot(update)) - 2. * jnp.real(update.dot(self._F0))) / (self.energy.var + 1e-10))
    
    def get_tdvp_equation(self, Eloc: SampledObs, gradients: SampledObs):
        '''
        Returns left and right hand side of the TDVP equation
        '''
        self._covar_grad_eloc = gradients.get_covar_obs(Eloc)
        self._F0 = - self.rhsPrefactor * self._covar_grad_eloc.mean.ravel()
        F = self._lhs_trans_fn(self._F0)

        self._S0 = gradients.get_covar()
        S = self._lhs_trans_fn(self._S0)

        if self.diagonalShift > 1e-10:
            S = S + jnp.diag(self.diagonalShift * jnp.diag(S))

        return S, F
    
    def _transform_to_eigenbasis(self, S, F):
        if self.diagonalizeOnDevice:
            try:
                self._ev, self._V = jnp.linalg.eigh(S)
            except ValueError:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            
                self._ev, self._V = _eigh_numpy(S)
        else:
            self._ev, self._V = _eigh_numpy(S)

        self._VtF = jnp.dot(jnp.transpose(jnp.conj(self._V)), F)

    def _get_snr(self):
        rho = self._covar_grad_eloc.transform(self._rhs_trans_fn, jnp.transpose(self._V))
        self._rho_var = rho.var.ravel()

        return jnp.sqrt(jnp.abs(rho._num_samples * (jnp.conj(self._VtF) * self._VtF) / (self._rho_var + 1e-10))).ravel()
    
    def solve(self, Eloc: SampledObs, gradients: SampledObs):
        # Get TDVP equation from MC data
        self._S, F = self.get_tdvp_equation(Eloc, gradients)
        F.block_until_ready()

        # Transform TDVP equation to eigenbasis and compute Signal to Noise Ratio
        self._transform_to_eigenbasis(self._S, F) 
        self._snr = self._get_snr()

        # Discard eigenvalues below numerical precision
        self._invEv = jnp.where(jnp.abs(self._ev / self._ev[-1]) > 1e-14, 1. / self._ev, 0.)

        residual = 1.0
        cutoff = 1e-2
        F_norm = jnp.linalg.norm(F)
        while residual > self.pinvTol and cutoff > self.pinvCutoff:
            cutoff *= 0.8
            # Set regularizer for singular value cutoff
            regularizer = 1. / (1. + (max(cutoff, self.pinvCutoff) / jnp.abs(self._ev / self._ev[-1]))**6)

            if not isinstance(self.sampler, ExactSampler):
                # Construct a soft cutoff based on the SNR
                regularizer *= 1. / (1. + (self.snrTol / self._snr)**6)

            pinvEv = self._invEv * regularizer

            self._residual = jnp.linalg.norm((pinvEv * self._ev - jnp.ones_like(pinvEv)) * self._VtF) / F_norm
            self._update = jnp.real(jnp.dot(self._V, (pinvEv * self._VtF)))
            self._pinv_cutoff = max(cutoff, self.pinvCutoff)

        return self._update
    
    def cross_validation(self, Eloc, gradients):
        Eloc1 = Eloc.get_subset(start=0, step=2)
        sampleGradients1 = gradients.get_subset(start=0, step=2)
        Eloc2 = Eloc.get_subset(start=1, step=2)
        sampleGradients2 = gradients.get_subset(start=1, step=2)
        update_1, _, _ = self.solve(Eloc1, sampleGradients1)
        S2, F2 = self.get_tdvp_equation(Eloc2, sampleGradients2)

        validation_tdvpErr = self._get_tdvp_error(update_1)
        _, solverResidual, _ = self.solve(Eloc, gradients)
        validation_residual = (jnp.linalg.norm(S2.dot(update_1) - F2) / jnp.linalg.norm(F2)) / solverResidual

        self.crossValidationFactor_residual = validation_residual
        self.crossValidationFactor_tdvpErr = validation_tdvpErr / self.meta_data["tdvp_error"]
        self.meta_data["tdvp_residual_cross_validation_ratio"] = self.crossValidationFactor_residual
        self.meta_data["tdvp_error_cross_validation_ratio"] = self.crossValidationFactor_tdvpErr

        self.S, _ = self.get_tdvp_equation(self.Eloc, gradients)
    
    def _update_meta_data(self):
        self.meta_data = {
            "tdvp_error": self._get_tdvp_error(self._update),
            "tdvp_residual": self._residual,
            "pinv_cutoff": self._pinv_cutoff,
            "SNR": self._snr, 
            "spectrum": self._ev,
        }