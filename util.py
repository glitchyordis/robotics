import numpy as np

def print_array(array, suppress_scientific=False, precision=8):
    """
    Prints a numpy array with a specified precision and optionally suppresses scientific notation.
    """
    if suppress_scientific:
        print(np.array2string(array, 
                              suppress_small=True, precision=precision))
    else:
        print(array)