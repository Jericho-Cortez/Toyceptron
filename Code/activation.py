def identity(x):
    return x

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def sigmoid(x):
    from math import exp
    return 1 / (1 + exp(-x))

def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0
