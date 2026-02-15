from layer import Layer


class Network:
    def __init__(self, input_size=None, activation=None):
        self.layers = []
        self.activation_default = activation
    
    def add(self, weights, biases, activation=None):
        if activation is None:
            activation = self.activation_default
        
        couche = Layer(
            weights_list=weights,
            biases_list=biases,
            activation=activation
        )
        self.layers.append(couche)
    
    def forward(self, inputs):
        donnees = inputs
        for couche in self.layers:
            donnees = couche.forward(donnees)
        return donnees
    
    feedforward = forward
    
    def summary(self):
        print("=" * 60)
        print("ARCHITECTURE DU RÉSEAU")
        print("=" * 60)
        
        if len(self.layers) == 0:
            print("Réseau vide")
        else:
            for i, couche in enumerate(self.layers):
                premier_neurone = couche.neurons[0]
                nb_entrees = len(premier_neurone.weights)
                nb_neurones = len(couche.neurons)
                nom_activation = premier_neurone.activation.__name__
                
                print(f"Couche {i+1} : {nb_entrees} entrées → {nb_neurones} neurones | {nom_activation}")
        
        print("=" * 60)
