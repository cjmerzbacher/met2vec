from numpy.random import RandomState
import random

def get_random_state(seed, index=0):
    index %= 2**32
        
    if seed == None:
        return RandomState(random.randint(0, 2**32))
    return RandomState(seed + index)