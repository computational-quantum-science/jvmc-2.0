import jax.numpy as jnp
import collections
from typing import Dict, Union, Any
import numpy as np

import jVMC_exp
import jVMC_exp.nets.activation_functions as act_funs
import jVMC_exp.util.symmetries as sym
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler
from jVMC_exp.stats import SampledObs
from jVMC_exp.operator.base import AbstractOperator

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
        "RBM": jVMC_exp.nets.RBM,
        "FFN": jVMC_exp.nets.FFN,
        "CNN": jVMC_exp.nets.CNN,
        "RNN": jVMC_exp.nets.RNN1DGeneral,
        "RNN2D": jVMC_exp.nets.RNN2DGeneral,
        "CpxRBM": jVMC_exp.nets.CpxRBM,
        "CpxCNN": jVMC_exp.nets.CpxCNN
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
        model = jVMC_exp.nets.two_nets_wrapper.TwoNets(net1=model1, net2=model2)
        isGenerator = "sample" in dir(model1)

    avgFun = jVMC_exp.nets.sym_wrapper.avgFun_Coefficients_Sep if isGenerator else jVMC_exp.nets.sym_wrapper.avgFun_Coefficients_Exp
    model = jVMC_exp.nets.sym_wrapper.SymNet(orbit=orbit, net=model, avgFun=avgFun)
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