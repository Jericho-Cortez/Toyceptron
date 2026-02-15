from neuron import Neuron
from activation import identity


class Layer:
    def __init__(self, num_neurons=None, num_inputs=None, activation=None,
                 weights_list=None, biases_list=None):
        self.neurons = []
        
        if num_neurons is not None and num_inputs is not None:
            if activation is None:
                activation = identity
            
            for i in range(num_neurons):
                neurone = Neuron(num_inputs=num_inputs, activation=activation)
                self.neurons.append(neurone)
        
        elif weights_list is not None and biases_list is not None:
            if len(weights_list) != len(biases_list):
                raise ValueError("Il faut autant de biais que de neurones (poids)")
            
            for i in range(len(weights_list)):
                neurone = Neuron(
                    weights=weights_list[i],
                    bias=biases_list[i],
                    activation=activation
                )
                self.neurons.append(neurone)
        
        else:
            raise ValueError("Donner soit (num_neurons + num_inputs), soit (weights_list + biases_list)")
    
    def forward(self, inputs):
        sorties = []
        for neurone in self.neurons:
            sortie = neurone.forward(inputs)
            sorties.append(sortie)
        return sorties
