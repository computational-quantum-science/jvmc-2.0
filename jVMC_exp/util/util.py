import jax.numpy as jnp
import collections
from typing import Dict, Union, Any
import numpy as np

import jVMC_exp
from jVMC_exp.global_defs import DT_OPERATORS_CPX
import jVMC_exp.nets.activation_functions as act_funs
import jVMC_exp.util.symmetries as sym
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import AbstractMCSampler
from jVMC_exp.operator.base import AbstractOperator
import jVMC_exp.operator.discrete.branch_free as op

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
    sampler: AbstractMCSampler, 
    num_samples: int = None
):
    ''' 
    Measure expectation values of operators.

    Args:
        observables: Dictionary with operator names as keys. Values can be:
            - A single operator or a callable
            - A single (operator or a callable, kwargs_dict) tuple  
            - A list of operators and/or (operator, kwargs_dict) tuples

            The callable has to return a jVMC.stats.SampledObs
            
            Examples:
            
            .. code-block:: python

                {
                    "energy": hamiltonian,
                    "magnetization": (mag_op, {"direction": "z", "t": 0.1})
                }
        sampler: Monte Carlo sampler  
        num_samples: Number of samples (optional)

    Returns:
        Dictionary with "mean", "variance", "MC_error" for each observable.
    '''
    result = {}
    for name, op in observables.items():
        kwargs = {}
        if isinstance(op, tuple):
            if len(op) != 2:
                raise ValueError("If an operator takes kwargs it has to be passed as tuple: (operator, dict_of_kwargs)")
            kwargs = op[1]
            op = op[0]

        if isinstance(op, AbstractOperator):
            Oloc = sampler(op, num_samples=num_samples, **kwargs)
        else:
            Oloc = op(**kwargs)
        
        result[name] = {}
        result[name]["mean"] = jnp.real(Oloc.mean).item()
        result[name]["variance"] = jnp.real(Oloc.var).item()
        result[name]["MC_error"] = jnp.real(Oloc.error_of_mean).item()

    return result

def matrix_to_jvmc_operator(
    matrix: np.ndarray,
    sites,
    *,
    return_conjugate: bool = False,
):
    r"""Convert a dense one- or two-site matrix into a ``jVMC`` discrete operator.

    The matrix is expanded in the local operator basis
    :math:`\{I, \sigma_x, \sigma_y, \sigma_z\}` and mapped onto the
    corresponding ``jVMC_exp.operator.discrete`` operators. For two-site
    matrices, tensor products of the same basis are used on the specified sites.

    Args:
        matrix: Dense operator matrix. Supported shapes are ``(2, 2)`` for a
            single-site operator and ``(4, 4)`` for a two-site operator.
        sites: Target lattice site or pair of lattice sites. Pass either a
            single integer or an iterable of length two.
        return_conjugate: If ``True``, also return the operator constructed from
            the conjugate transpose ``matrix.conj().T``.

    Returns:
        The discrete operator corresponding to ``matrix`` on ``sites``. If
        ``return_conjugate=True``, returns ``(operator, conjugate_operator)``.

    Raises:
        ValueError: If ``sites`` does not specify one or two sites, if
            ``matrix`` is not square, or if its shape is incompatible with the
            number of sites.
    """
    local_basis = [
        np.eye(2, dtype=DT_OPERATORS_CPX),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=DT_OPERATORS_CPX),
        np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=DT_OPERATORS_CPX),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=DT_OPERATORS_CPX),
    ]
    jvmc_basis = [
        lambda i: op.IdentityOperator(2, i),
        lambda i: op.SigmaX(i),
        lambda i: op.SigmaY(i),
        lambda i: op.SigmaZ(i),
    ]

    if isinstance(sites, (int, np.integer)):
        sites = (int(sites),)
    else:
        sites = tuple(sites)

    if len(sites) not in (1, 2):
        raise ValueError("sites must be an int or a sequence of length 2.")

    matrix = np.asarray(matrix, dtype=DT_OPERATORS_CPX)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    if len(sites) == 1 and matrix.shape != (2, 2):
        raise ValueError("Single-site operators require a 2x2 matrix.")
    if len(sites) == 2 and matrix.shape != (4, 4):
        raise ValueError("Two-site operators require a 4x4 matrix.")

    def build_from(mat):
        if len(sites) == 1:
            jvmc_op = 0
            for idx, basis in enumerate(local_basis):
                coeff = 0.5 * complex(np.trace(basis.conj().T @ mat))
                jvmc_op += coeff * jvmc_basis[idx](sites[0])
            return jvmc_op

        jvmc_op = 0
        for i, left_basis in enumerate(local_basis):
            for j, right_basis in enumerate(local_basis):
                basis = np.kron(left_basis, right_basis)
                coeff = 0.25 * complex(np.trace(basis.conj().T @ mat))
                jvmc_op += coeff * (jvmc_basis[i](sites[0]) * jvmc_basis[j](sites[1]))
        return jvmc_op

    if return_conjugate:
        return build_from(matrix), build_from(matrix.conj().T)
    return build_from(matrix)