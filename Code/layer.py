from neuron import Neuron
from activations import identity

class Layer:
    """
    Une couche = plusieurs neurones qui reçoivent les mêmes inputs.
    """
    def __init__(self, num_neurons=None, num_inputs=None, weights_list=None, biases_list=None, activation=identity):
        """
        Deux modes d'initialisation :
        
        MODE 1 (automatique) :
            num_neurons (int) : Nombre de neurones
            num_inputs (int) : Nombre d'entrées par neurone
            activation (function) : Fonction d'activation
            → Les poids/biais sont générés aléatoirement
        
        MODE 2 (manuel) :
            weights_list : [[w11, w12, ...], [w21, w22, ...], ...]
            biases_list : [b1, b2, ...]
            activation (function) : Fonction d'activation
            → Les poids/biais sont fournis explicitement
        """
        self.neurons = []
        
        # MODE 1 : Initialisation automatique
        if weights_list is None and num_neurons is not None and num_inputs is not None:
            for _ in range(num_neurons):
                neuron = Neuron(num_inputs=num_inputs, activation=activation)
                self.neurons.append(neuron)
        
        # MODE 2 : Initialisation manuelle
        elif weights_list is not None and biases_list is not None:
            for weights, bias in zip(weights_list, biases_list):
                neuron = Neuron(weights=weights, bias=bias, activation=activation)
                self.neurons.append(neuron)
        
        else:
            raise ValueError("Fournir soit (num_neurons, num_inputs) soit (weights_list, biases_list)")
    
    def forward(self, inputs):
        """
        Propage les inputs dans tous les neurones.
        
        Args:
            inputs : liste [x1, x2, ...]
        Returns:
            liste des sorties [z1, z2, ...]
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return outputs


# Test
if __name__ == "__main__":
    # Test MODE 2 (manuel)
    layer_manual = Layer(
        weights_list=[
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1],
        ],
        biases_list=[0.0, 0.1],
        activation=identity
    )
    result = layer_manual.forward([1.0, 2.0, 4.0])
    print(f"✅ Sorties couche (manuel) : {result}")
    
    # Test MODE 1 (automatique)
    layer_auto = Layer(num_neurons=3, num_inputs=2, activation=identity)
    result_auto = layer_auto.forward([1.0, 2.0])
    print(f"✅ Sorties couche (auto) : {result_auto}")
