import jax.numpy as jnp

from jVMC_exp.stats import SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.operator.base import AbstractOperator

class MinSR:
    """ 
    This class provides functionality for energy minimization via MinSR.

    See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``pinvTol``: Regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``diagonalSchift``: Regularization parameter :math:`\\lambda`, see below.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """

    def __init__(self, sampler: AbstractMCSampler, pinvTol=1e-14, diagonalShift=0., diagonalizeOnDevice=True):
        self.sampler = sampler
        self.pinvTol = pinvTol
        self.diagonalShift = diagonalShift
        self.diagonalizeOnDevice = diagonalizeOnDevice
        self.metaData = None
        self.Eloc0 = None

    @property
    def energy(self) -> SampledObs:
        return self.Eloc0

    def solve(self, Eloc: SampledObs, gradients: SampledObs, holomorphic):
        """
        Uses the techique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """
        if holomorphic:
            T = gradients.tangent_kernel
            T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)
            return - gradients._normalized_obs.conj().T @ T_inv @ Eloc._normalized_obs.squeeze()

        gradients_all = jnp.concatenate([jnp.real(gradients._normalized_obs), jnp.imag(gradients._normalized_obs)])
        Eloc_all = jnp.concatenate([jnp.real(Eloc._normalized_obs), jnp.imag(Eloc._normalized_obs)]).squeeze()

        T = gradients_all @ gradients_all.T
        T = T + self.diagonalShift * jnp.eye(T.shape[-1])
        T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)

        return - gradients_all.T @ T_inv @ Eloc_all

    def __call__(self, netParameters, t, *, psi: NQS, hamiltonian: AbstractOperator,
                 numSamples=None, outp: None | OutputManager = None, intStep=None):
        """ 
        For given network parameters computes an update step using the MinSR method.

        This function returns :math:`\\dot\\theta=\\bar O^\\dagger (\\bar O\\bar O^\\dagger + \\lambda\\mathbb{I})^{-1}\\bar E_{loc}`
        (see `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details). 
        Thereby an instance of the ``MinSR`` class is a suited callable for the right hand side of an ODE to be 
        used in combination with the integration schemes implemented in ``jVMC.stepper``. 
        Alternatively, the interface matches the scipy ODE solvers as well.

        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.

        Returns:
            The solution of the MinSR equation, :math:`\\dot\\theta=\\bar O^\\dagger (\\bar O\\bar O^\\dagger)^{-1}\\bar E_{loc}`.
        """
        tmpParameters = psi.parameters
        psi.parameters = netParameters

        def start_timing(name):
            if outp is not None:
                outp.start_timing(name)

        def stop_timing(name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if outp is not None:
                outp.stop_timing(name)

        # Get sample
        start_timing("sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        stop_timing("sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing("compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, LogPsiS=sampleLogPsi, t=t)
        stop_timing("compute Eloc", waitFor=Eloc)
        self.Eloc = SampledObs(Eloc, p)

        # Evaluate gradients
        start_timing("compute gradients")
        sampleGradients = psi.gradients(sampleConfigs)
        stop_timing("compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs(sampleGradients, p)

        start_timing("solve TDVP eqn.")
        update = self.solve(self.Eloc, sampleGradients, holomorphic=psi.holomorphic)
        stop_timing("solve TDVP eqn.")

        psi.parameters = tmpParameters

        if intStep is not None:
            if intStep == 0:
                self.Eloc0 = self.Eloc
                self.metaData = {}

        return update
