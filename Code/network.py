from layer import Layer

class Network:
    def __init__(self, layer_sizes, activations):
        """
        Construit un réseau de neurones multi-couches.
        
        Args:
            layer_sizes: liste [nb_inputs, nb_hidden1, ..., nb_outputs]
            activations: liste de fonctions d'activation (une par couche)
        """
        self.layers = []
        
        # Créer chaque couche
        for i in range(len(layer_sizes) - 1):
            num_inputs = layer_sizes[i]
            num_neurons = layer_sizes[i + 1]
            activation = activations[i]
            
            layer = Layer(num_neurons, num_inputs, activation)
            self.layers.append(layer)
    
    def forward(self, inputs):
        """
        Propagation avant : fait passer les inputs à travers toutes les couches.
        
        Args:
            inputs: liste de valeurs d'entrée
        
        Returns:
            Liste des sorties du réseau
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current


# Test rapide
if __name__ == "__main__":
    from activations import relu, sigmoid
    
    # Réseau 2 → 3 → 1
    net = Network(layer_sizes=[2, 3, 1], activations=[relu, sigmoid])
    result = net.forward([1.0, 2.0])
    print(f"Sortie du réseau : {result}")
    
    # Réseau plus complexe 4 → 5 → 3 → 1
    net2 = Network(layer_sizes=[4, 5, 3, 1], activations=[relu, relu, sigmoid])
    result2 = net2.forward([1.0, 2.0, 3.0, 4.0])
    print(f"Sortie réseau complexe : {result2}")
