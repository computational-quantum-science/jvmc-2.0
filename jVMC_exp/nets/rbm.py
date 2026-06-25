import jax
import flax.linen as nn
import jax.numpy as jnp

import jVMC_exp.global_defs as global_defs
import jVMC_exp.nets.activation_functions as act_funs
from jVMC_exp.nets.initializers import init_fn_args

import jVMC_exp.nets.initializers


class CpxRBM(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC_exp.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.DT_PARAMS_CPX)
                         )

        return jnp.sum(act_funs.log_cosh(layer(2 * s.ravel() - 1)))

class CpxRBM_Nospinflip(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC_exp.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.DT_PARAMS_CPX)
                         )

        return jnp.sum(act_funs.log_cosh(layer(s.ravel())))


class RBM(nn.Module):
    """Restricted Boltzmann machine with real parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.DT_PARAMS_REAL),
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.DT_PARAMS_REAL)
                        )

        return jnp.sum(jnp.log(jnp.cosh(layer(2 * s - 1))))

class CpxRBM_ratio(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC_exp.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.DT_PARAMS_CPX)
                         )

        return jnp.sum(act_funs.log_cosh(layer(2 * s.ravel() - 1)))

    def eval_ratio(self, s, sp):
        return jnp.exp(self(sp) - self(s))
