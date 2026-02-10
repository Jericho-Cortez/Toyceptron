# layer.py - VERSION CORRIGÉE
import random
from neuron import Neuron


class Layer:
    def __init__(self, num_neurons, num_inputs, activation):
        """
        Initialise une couche avec plusieurs neurones.
        
        Args:
            num_neurons (int): Nombre de neurones dans la couche
            num_inputs (int): Nombre d'entrées pour chaque neurone
            activation (function): Fonction d'activation commune
        """
        self.neurons = []
        
        for _ in range(num_neurons):
            # Générer les poids aléatoires pour chaque neurone
            weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
            
            # Créer le neurone avec poids pré-générés
            neuron = Neuron(
                weights=weights,
                bias=random.uniform(-1, 1),
                activation=activation
            )
            self.neurons.append(neuron)
    
    def forward(self, inputs):
        """
        Propagation avant de toute la couche.
        Chaque neurone reçoit les MÊMES inputs.
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return outputs
