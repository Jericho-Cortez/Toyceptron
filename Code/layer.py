from neuron import Neuron


class Layer:
    """
    Une couche = plusieurs neurones qui reçoivent les mêmes inputs.
    Retourne les sorties brutes (pas activées).
    """
    def __init__(self, weights_list, biases_list):
        """
        weights_list : liste de listes [[w11, w12, ...], [w21, w22, ...], ...]
                       Chaque sous-liste = poids d'un neurone
        biases_list : liste des biais [b1, b2, ...]
        """
        self.neurons = []
        
        # Créer un neurone pour chaque ligne de poids
        for weights, bias in zip(weights_list, biases_list):
            neuron = Neuron(weights=weights, bias=bias)
            self.neurons.append(neuron)
    
    def forward(self, inputs):
        """
        Propage les inputs dans tous les neurones.
        inputs : liste [x1, x2, ...]
        Retourne : liste des sorties brutes [z1, z2, ...]
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        
        return outputs


# Test
if __name__ == "__main__":
    layer = Layer(
        weights_list=[
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1],
        ],
        biases_list=[0.0, 0.1]
    )
    result = layer.forward([1.0, 2.0, 4.0])
    print(f"Sorties couche : {result}")  # Doit afficher [1.6, 0.7]
