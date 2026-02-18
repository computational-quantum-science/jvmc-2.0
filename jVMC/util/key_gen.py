import jax
import numpy as np

def format_key(key):
    if isinstance(key, jax.Array):
        return key
    elif key is None:
        return jax.random.PRNGKey(generate_seed())
    else:
        return jax.random.PRNGKey(int(key))
    
def generate_seed():
    return np.random.SeedSequence().entropy % int(2 ** 32)