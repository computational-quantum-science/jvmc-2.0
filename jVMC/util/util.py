import jax.numpy as jnp
import collections
from typing import Dict, Union, Any
import numpy as np

import jVMC
import jVMC.nets.activation_functions as act_funs
import jVMC.util.symmetries as sym
from jVMC.vqs import NQS
from jVMC.sampler import AbstractMCSampler
from jVMC.stats import SampledObs
from jVMC.operator.base import AbstractOperator

OperatorWithKwargs = tuple[AbstractOperator, dict[str, Any]]
ObservableEntry = Union[AbstractOperator, OperatorWithKwargs]

def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)

def init_net(descr, dims, seed=0):
    def get_activation_functions(actFuns):
        if type(actFuns) is list:
            return tuple([act_funs.activationFunctions[fn] for fn in actFuns])

        return act_funs.activationFunctions[actFuns]

    netTypes = {
        "RBM": jVMC.nets.RBM,
        "FFN": jVMC.nets.FFN,
        "CNN": jVMC.nets.CNN,
        "RNN": jVMC.nets.RNN1DGeneral,
        "RNN2D": jVMC.nets.RNN2DGeneral,
        "CpxRBM": jVMC.nets.CpxRBM,
        "CpxCNN": jVMC.nets.CpxCNN
    }

    def get_net(descr, dims):
        return netTypes[descr["type"]](**descr["parameters"])

    if "actFun" in descr["net1"]["parameters"]:
        descr["net1"]["parameters"]["actFun"] = get_activation_functions(descr["net1"]["parameters"]["actFun"])

    symms = tuple()
    for key in ["translation", "rotation", "reflection", "spinflip"]:
        if key in descr:
            symms = symms + (key,)
    symm_factors = {}
    for key in symms:
        fac_key = key + "_factor"
        symm_factors[fac_key] = descr[fac_key] if fac_key in descr else 1

    if len(dims) == 2:
        orbit = sym.get_orbit_2D_square(dims[0], *symms, **symm_factors)
    if len(dims) == 1:
        orbit = sym.get_orbit_1D(dims[0], *symms, **symm_factors)

    if not "net2" in descr:
        model = get_net(descr["net1"], dims)
        isGenerator = "sample" in dir(model)

    else:
        if "actFun" in descr["net2"]["parameters"]:
            descr["net2"]["parameters"]["actFun"] = get_activation_functions(descr["net2"]["parameters"]["actFun"])

        model1 = get_net(descr["net1"], dims)
        model2 = get_net(descr["net2"], dims)
        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=model1, net2=model2)
        isGenerator = "sample" in dir(model1)

    avgFun = jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep if isGenerator else jVMC.nets.sym_wrapper.avgFun_Coefficients_Exp
    model = jVMC.nets.sym_wrapper.SymNet(orbit=orbit, net=model, avgFun=avgFun)
    psi = NQS(model, sampleShape=dims, batchSize=descr["batch_size"], seed=seed) # TODO: is dims = sampleShape?
    psi(jnp.zeros((1,) + dims, dtype=np.int32))

    return psi

def measure(
    observables: Dict[str, ObservableEntry], 
    psi: NQS, 
    sampler: AbstractMCSampler, 
    numSamples: int = None
):
    ''' 
    Measure expectation values of operators.

    Args:
        observables: Dictionary with operator names as keys. Values can be:
            - A single operator
            - A single (operator, kwargs_dict) tuple  
            - A list of operators and/or (operator, kwargs_dict) tuples
            
            Examples:
            
            .. code-block:: python

                {
                    "energy": hamiltonian,
                    "magnetization": (mag_op, {"direction": "z", "t": 0.1})
                }

        psi: Variational wave function
        sampler: Monte Carlo sampler  
        numSamples: Number of samples (optional)

    Returns:
        Dictionary with "mean", "variance", "MC_error" for each observable.
    '''
    samples, logPsiS, p = sampler.sample(numSamples=numSamples)

    result = {}
    for name, op in observables.items():
        kwargs = {}
        if isinstance(op, tuple):
            if len(op) != 2:
                raise ValueError("If an operator takes kwargs it has to be passed as tuple: (operator, dict_of_kwargs)")
            kwargs = op[1]
            op = op[0]

        Oloc = SampledObs(op.get_O_loc(samples, psi, logPsiS=logPsiS, **kwargs), p)
        result[name] = {}
        result[name]["mean"] = jnp.real(Oloc.mean).item()
        result[name]["variance"] = jnp.real(Oloc.var).item()
        result[name]["MC_error"] = jnp.real(Oloc.error_of_mean).item()

    return result


# def ground_state_search(psi: NQS, ham, tdvpEquation, sampler, numSteps=200, varianceTol=1e-10, 
#                         stepSize=1e-2, observables=None, outp: OutputManager | None =None):
#     ''' 
#     This function performs a ground state search by Stochastic Reconfiguration.

#     Arguments:
#         * ``psi``: Variational wave function (``jVMC.vqs.NQS``)
#         * ``ham``: Hamiltonian operator
#         * ``tdvpEquation``: An instance of ``jVMC.util.TDVP`` or ``jVMC.util.MinSR``
#         * ``numSteps``: Maximal number of steps
#         * ``varianceTol``: Stopping criterion
#         * ``stepSize``: Update step size (learning rate)
#         * ``observables``: Observables to be measured during ground state search
#         * ``outp``: ``None`` or instance of ``jVMC.util.OutputManager``.

#     '''
#     if "diagonalShift" in dir(tdvpEquation):
#         delta = tdvpEquation.diagonalShift

#     stepper = jVMCstepper.Euler(timeStep=stepSize)

#     n = 0
#     if outp is not None:
#         if observables is not None:
#             obs = measure(observables, psi, sampler)
#             outp.write_observables(n, **obs)

#     varE = 1.0

#     while n < numSteps and varE > varianceTol:

#         tic = time.perf_counter()

#         dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=ham, psi=psi, numSamples=None, outp=outp)
#         psi.set_parameters(dp)
#         n += 1

#         varE = tdvpEquation.get_energy_variance()

#         if outp is not None:
#             if observables is not None:
#                 obs = measure(observables, psi, sampler)
#                 outp.write_observables(n, **obs)

#         if "set_diagonal_shift" in dir(tdvpEquation):
#             delta = 0.95 * delta
#             tdvpEquation.set_diagonal_shift(delta)

#         if outp is not None:
#             outp.print(" STEP %d" % (n))
#             outp.print("   Energy mean: %f" % (tdvpEquation.get_energy_mean()))
#             outp.print("   Energy variance: %f" % (varE))
#             outp.print_timings(indent="   ")
#             outp.print("   == Time for step: %fs" % (time.perf_counter() - tic))
