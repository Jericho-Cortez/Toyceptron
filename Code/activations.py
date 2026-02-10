"""
Fonctions d'activation pour les neurones
Toyceptron - Projet pédagogique Python pur
"""

def identity(x):
    """Fonction identité : retourne x tel quel"""
    return x

def heaviside(x):
    """Fonction seuil (Heaviside) : 1 si x >= 0, sinon 0"""
    return 1 if x >= 0 else 0

def sigmoid(x):
    """Fonction sigmoïde : 1 / (1 + e^-x)"""
    import math
    return 1 / (1 + math.exp(-x))

def relu(x):
    """ReLU (Rectified Linear Unit) : max(0, x)"""
    return max(0, x)
