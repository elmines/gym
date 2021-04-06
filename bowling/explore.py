import numpy as np

def make_uniform_noise_func(num_choices):
    arr = np.array(num_choices * [1./num_choices])
    return lambda: arr

def make_random_baseline(seed = 0):
    rng = np.random.default_rng()
    def f(s):
        s = np.squeeze(np.array(s))
        return rng.uniform(-25, 25, len(s))
    return f
