import jax
import flax
import jVMC_exp

# This class defines the network structure of a complex RBM


class MyNet(flax.linen.Module):
    numHidden: int = 2

    @flax.linen.compact
    def __call__(self, s):

        s = 2 * s - 1  # Go from 0/1 representation to 1/-1

        h = flax.linen.Dense(features=self.numHidden,
                             dtype=jVMC_exp.global_defs.DT_PARAMS_CPX)(s)

        h = jax.numpy.log(jax.numpy.cosh(h))

        return jax.numpy.sum(h)


L = 4  # system size

# Initialize custom net
net = MyNet(numHidden=7)

# Create the variational quantum state
psi = jVMC_exp.vqs.NQS(net, seed=1234)

# Create a set of 13 random input configurations
configs = jax.random.bernoulli(jax.random.PRNGKey(4321), shape=(1, 13, L))

# Evaluate the net on the input
coeffs = psi(configs)

# Check output shape
print("coeffs.shape:", coeffs.shape)
