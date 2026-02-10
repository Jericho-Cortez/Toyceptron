from layer import Layer

class Network:
    """
    Un réseau = plusieurs couches enchaînées.
    Les sorties d'une couche = inputs de la suivante.
    """
    def __init__(self, layer_sizes, activations):
        """
        Construit le réseau à partir des hyperparamètres.
        
        Args:
            layer_sizes (list) : [nb_inputs, nb_hidden1, nb_hidden2, ..., nb_outputs]
                                 Ex: [2, 3, 1] → 2 inputs, 1 couche cachée de 3 neurones, 1 output
            activations (list) : Liste des fonctions d'activation par couche
                                 Ex: [relu, sigmoid] → relu pour couche cachée, sigmoid pour output
        
        Exemple :
            net = Network([2, 3, 1], [relu, sigmoid])
            → Input : 2 valeurs
            → Hidden layer : 3 neurones avec ReLU
            → Output layer : 1 neurone avec sigmoid
        """
        self.layers = []
        
        # Créer les couches successives
        for i in range(len(layer_sizes) - 1):
            num_inputs = layer_sizes[i]      # Taille de la couche précédente
            num_neurons = layer_sizes[i + 1] # Taille de la couche actuelle
            activation = activations[i]       # Fonction d'activation de cette couche
            
            layer = Layer(
                num_neurons=num_neurons,
                num_inputs=num_inputs,
                activation=activation
            )
            self.layers.append(layer)
    
    def forward(self, inputs):
        """
        Propage les inputs à travers toutes les couches.
        
        Args:
            inputs (list) : Vecteur d'entrée initial
        
        Returns:
            list : Sortie finale du réseau (dernière couche)
        
        Exemple :
            net = Network([2, 3, 1], [relu, sigmoid])
            output = net.forward([1.0, 2.0])
            → output = [0.73] (1 valeur car 1 neurone en sortie)
        """
        current = inputs
        
        # Propager à travers chaque couche
        for layer in self.layers:
            current = layer.forward(current)
        
        return current


# Test
if __name__ == "__main__":
    from activations import identity, relu, sigmoid
    
    # Test 1 : Réseau simple avec activations identité (pour debug)
    net_simple = Network([2, 3, 1], [identity, identity])
    result = net_simple.forward([1.0, 1.0])
    print(f"✅ Test réseau simple : {result}")
    print(f"   (Attendu : 1 valeur)")
    
    # Test 2 : Réseau avec activations réelles
    net_real = Network([3, 5, 2], [relu, sigmoid])
    result_real = net_real.forward([0.5, -0.2, 1.0])
    print(f"✅ Test réseau réel : {result_real}")
    print(f"   (Attendu : 2 valeurs entre 0 et 1)")
