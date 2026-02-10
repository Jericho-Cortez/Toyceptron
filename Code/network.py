from layer import Layer


class Network:
    """
    Un réseau = plusieurs couches enchaînées.
    L'activation est appliquée APRÈS chaque couche.
    """
    def __init__(self, input_size, activation):
        """
        input_size : nombre d'entrées du réseau (ex: 3)
        activation : fonction d'activation (ex: act_sigmoid)
        """
        self.input_size = input_size
        self.activation = activation
        self.layers = []
    
    def add(self, weights, biases):
        """
        Ajoute une couche au réseau.
        weights : liste de listes [[w11, ...], [w21, ...], ...]
        biases : liste [b1, b2, ...]
        """
        layer = Layer(weights_list=weights, biases_list=biases)
        self.layers.append(layer)
    
    def feedforward(self, inputs):
        """
        Propage les inputs à travers toutes les couches.
        Applique l'activation après chaque couche.
        inputs : liste [x1, x2, ...]
        Retourne : liste des sorties finales activées
        """
        current = inputs
        
        for layer in self.layers:
            # 1. Propagation brute
            raw_outputs = layer.forward(current)
            
            # 2. Application de l'activation
            activated_outputs = [self.activation(z) for z in raw_outputs]
            
            # 3. Les sorties activées deviennent les entrées de la couche suivante
            current = activated_outputs
        
        return current


# Test
if __name__ == "__main__":
    from math import exp
    
    def act_sigmoid(x):
        return 1 / (1 + exp(-x))
    
    net = Network(input_size=3, activation=act_sigmoid)
    
    net.add(
        weights=[
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1],
        ],
        biases=[0.0, 0.1]
    )
    
    result = net.feedforward([1.0, 2.0, 4.0])
    print(f"Sortie réseau : {result}")
    # Doit afficher environ [0.832..., 0.668...]
