import jax
from jax import grad
from jax.tree_util import tree_flatten, tree_map
import jax.numpy as jnp

def flat_gradient_real(fun, params, arg):
    """
    Grad for fun: Real -> Real
    """
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]

    return jnp.concatenate(g)

def flat_gradient(fun, params, arg):
    """
    Grad for fun: Real -> Complex 
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_nonholo(fun, params, arg):
    """
    Grad for fun: Complex -> Complex (not holomorphic) 
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_holo(fun, params, arg):
    """
    Grad for fun: Complex -> Complex (holomorphic) 
    """
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j * x.ravel()], g))[0]

    return jnp.concatenate(g)

def dict_gradient(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_map(lambda x: x.ravel(), gr)
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_map(lambda x: x.ravel(), gi)

    return tree_map(lambda x,y: x + 1.j * y, gr, gi)

def dict_gradient_real(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_map(lambda x: x.ravel(), g)

    return g

def pick_gradient(fn, params, single_input):
    dtypes = [a.dtype for a in tree_flatten(params)[0]]
    if not all(d == dtypes[0] for d in dtypes):
        raise Exception("Network uses different parameter data types. This is not supported.")

    real_params = dtypes[0] in [jnp.single, jnp.double]

    def make_flat(t):
        return jnp.concatenate([p.ravel() for p in tree_flatten(t)[0]])
    
    grads_r = make_flat(jax.grad(lambda a, b: jnp.real(fn(a, b)))(params, single_input)["params"])
    grads_i = make_flat(jax.grad(lambda a, b: jnp.imag(fn(a, b)))(params, single_input)["params"])
    holomorphic = jnp.isclose(jnp.linalg.norm(grads_r - 1.j * grads_i) / grads_r.shape[0], 0.0, atol=1e-14)

    if holomorphic:
       return real_params, holomorphic, flat_gradient_holo, dict_gradient_real
    
    if real_params:
        return real_params, holomorphic, flat_gradient, dict_gradient
    else:
        return real_params, holomorphic, flat_gradient_nonholo, dict_gradient_real