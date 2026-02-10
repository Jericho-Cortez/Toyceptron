# activations.py

def identity(x):
    """Fonction identité : f(x) = x"""
    return x

def heaviside(x):
    """Fonction seuil (Heaviside) : 0 si x < 0, sinon 1"""
    return 1 if x >= 0 else 0

def sigmoid(x):
    """Fonction sigmoïde : f(x) = 1 / (1 + e^(-x))"""
    import math
    return 1 / (1 + math.exp(-x))

def relu(x):
    """Fonction ReLU : f(x) = max(0, x)"""
    return max(0, x)
