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
                from activation import identity
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
