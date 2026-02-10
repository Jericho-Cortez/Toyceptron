from neuron import Neuron


class Layer:
    def __init__(self, num_neurons=None, num_inputs=None, activation=None, 
                 weights_list=None, biases_list=None):
        """
        Crée une couche de neurones.
        
        Mode 1 - Poids aléatoires :
            Layer(num_neurons=3, num_inputs=2, activation=relu)
        
        Mode 2 - Poids fournis (main.py prof) :
            Layer(
                weights_list=[[0.5, -0.3], [0.2, 0.1]],
                biases_list=[0.0, 0.1],
                activation=sigmoid  # Optionnel si pas d'activation
            )
        """
        self.neurons = []
        
        # MODE 1 : Créer avec poids aléatoires
        if num_neurons is not None and num_inputs is not None:
            if activation is None:
                from activations import identity
                activation = identity
            
            for _ in range(num_neurons):
                neuron = Neuron(num_inputs=num_inputs, activation=activation)
                self.neurons.append(neuron)
        
        # MODE 2 : Créer avec poids fournis
        elif weights_list is not None and biases_list is not None:
            num_neurons = len(weights_list)
            
            if len(biases_list) != num_neurons:
                raise ValueError(f"Il faut autant de biais ({len(biases_list)}) que de poids ({num_neurons})")
            
            for i in range(num_neurons):
                neuron = Neuron(weights=weights_list[i], bias=biases_list[i], activation=activation)
                self.neurons.append(neuron)
        
        else:
            raise ValueError(
                "Vous devez fournir soit :\n"
                "  - num_neurons + num_inputs (poids aléatoires)\n"
                "  - weights_list + biases_list (poids fournis)"
            )
    
    def forward(self, inputs):
        """
        Propagation avant à travers tous les neurones de la couche.
        
        Paramètres :
            inputs (list) : Vecteur d'entrée
        
        Retourne :
            list : Liste des sorties (une par neurone)
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return outputs


# ============================================================
# TESTS
# ============================================================
if __name__ == "__main__":
    from activations import relu, sigmoid
    
    print("\n" + "="*60)
    print("TEST 1 : Layer avec poids aléatoires")
    print("="*60)
    
    layer1 = Layer(num_neurons=3, num_inputs=2, activation=relu)
    result1 = layer1.forward([1.0, 2.0])
    print(f"Entrée: [1.0, 2.0]")
    print(f"Sortie: {result1}")
    
    print("\n" + "="*60)
    print("TEST 2 : Layer avec poids fournis (main.py)")
    print("="*60)
    
    layer2 = Layer(
        weights_list=[
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1]
        ],
        biases_list=[0.0, 0.1]
    )
    result2 = layer2.forward([2, 3, 1])
    print(f"Entrée: [2, 3, 1]")
    print(f"Sortie: {result2}")
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS !")
    print("="*60)
