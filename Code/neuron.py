import random
from activation import identity


class Neuron:
    def __init__(self, weights=None, bias=0.0, num_inputs=None, activation=None):
        if weights is not None:
            self.weights = weights
        elif num_inputs is not None:
            self.weights = []
            for i in range(num_inputs):
                poids_aleatoire = random.uniform(-1, 1)
                self.weights.append(poids_aleatoire)
        else:
            raise ValueError("Il faut donner soit 'weights', soit 'num_inputs'")
        
        self.bias = bias
        
        if activation is None:
            self.activation = identity
        else:
            self.activation = activation
    
    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError(f"Erreur : {len(inputs)} entrées mais {len(self.weights)} poids")
        
        somme = 0.0
        for i in range(len(inputs)):
            somme += inputs[i] * self.weights[i]
        
        z = somme + self.bias
        sortie = self.activation(z)
        
        return sortie
