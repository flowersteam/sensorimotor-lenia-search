import torch
import numbers
import numpy as np



def sample_value(rnd=None, config=None):
    '''Samples scalar values depending on the provided properties.'''

    if rnd is None:
        rnd = np.random.RandomState()

    val = None

    if isinstance(config, numbers.Number): # works also for booleans
        val = config

    elif config is None:
        val = rnd.rand()

    elif isinstance(config, tuple):

        if config[0] == 'continuous' or config[0] == 'continous':
            val = config[1] + (rnd.rand() * (config[2] - config[1]))

        elif config[0] == 'discrete':
            val = rnd.randint(config[1], config[2] + 1)

        elif config[0] == 'function' or config[0] is 'func':
            val = config[1](rnd, *config[2:])    # call function and give the other elements in the tupel as paramters

        elif len(config) == 2:
            val = config[0] + (rnd.rand() * (config[1] - config[0]))

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config[0])

    elif isinstance(config, list):
        val = config[rnd.randint(len(config))] # do not use choice, because it does not work with tuples

    elif isinstance(config, dict):
        if config['type'] == 'discrete':
            val = rnd.randint(config['min'], config['max'] + 1)

        elif config['type'] == 'continuous':
            val = config['min'] + (rnd.rand() * (config['max'] - config['min']))

        elif config['type'] == 'boolean':
            val = bool(rnd.randint(0,1))

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config['type'])

    return val
    

def roll_n(X, axis, n):
    """ Rolls a tensor with a shift n on the specified axis"""
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
