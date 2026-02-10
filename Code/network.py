from layer import Layer


class Network:
    def __init__(self, layer_sizes=None, activations=None, 
                 input_size=None, hidden_layers=None, output_size=None, activation=None):
        """
        Constructeur flexible pour Network.
        
        Mode 1 - Architecture complète :
            Network(layer_sizes=[2, 4, 1], activations=[relu, sigmoid])
            → Crée les couches immédiatement
        
        Mode 2 - Construction progressive (main.py prof) :
            net = Network(input_size=3, activation=sigmoid)
            net.add(weights=[...], biases=[...])
            → Ne crée RIEN au départ, juste métadonnées
        
        Mode 3 - Architecture auto avec couches cachées :
            Network(input_size=3, hidden_layers=[5,3], output_size=1, activation=relu)
            → Crée les couches automatiquement
        """
        self.layers = []
        self.input_size = input_size
        self.activation_default = activation
        
        # MODE 1 : Architecture explicite avec layer_sizes
        if layer_sizes is not None:
            if activations is None:
                raise ValueError("Si 'layer_sizes' est fourni, 'activations' est obligatoire")
            
            if len(activations) != len(layer_sizes) - 1:
                raise ValueError(f"Il faut {len(activations)} activations")
            
            for i in range(len(layer_sizes) - 1):
                num_inputs = layer_sizes[i]
                num_neurons = layer_sizes[i + 1]
                activation_func = activations[i]
                
                layer = Layer(
                    num_neurons=num_neurons,
                    num_inputs=num_inputs,
                    activation=activation_func
                )
                self.layers.append(layer)
        
        # MODE 2 : Architecture automatique UNIQUEMENT si hidden_layers OU output_size fournis
        elif input_size is not None and (hidden_layers is not None or output_size is not None):
            if activation is None:
                raise ValueError("Si 'input_size' + 'hidden_layers'/'output_size', 'activation' obligatoire")
            
            if hidden_layers is None:
                hidden_layers = []
            if output_size is None:
                output_size = 1
            
            all_sizes = [input_size] + hidden_layers + [output_size]
            
            for i in range(len(all_sizes) - 1):
                layer = Layer(
                    num_neurons=all_sizes[i + 1],
                    num_inputs=all_sizes[i],
                    activation=activation
                )
                self.layers.append(layer)
        
        # MODE 3 : Si juste input_size + activation → réseau VIDE (pour .add() après)
        # On ne fait rien, self.layers reste vide []
    
    def add(self, layer=None, weights=None, biases=None, activation=None):
        """
        Ajoute une couche au réseau.
        
        Mode 1 - Ajouter un objet Layer :
            net.add(Layer(num_neurons=5, num_inputs=3, activation=relu))
        
        Mode 2 - Créer avec poids fournis (main.py prof) :
            net.add(
                weights=[[0.5, -0.3], [0.2, 0.1]],
                biases=[0.0, 0.1],
                activation=sigmoid  # Optionnel, utilise self.activation_default si None
            )
        """
        # MODE 1 : Ajouter un objet Layer directement
        if layer is not None:
            if not isinstance(layer, Layer):
                raise TypeError("Le paramètre 'layer' doit être un objet de type Layer")
            self.layers.append(layer)
            return
        
        # MODE 2 : Créer une couche à partir de weights + biases
        if weights is not None and biases is not None:
            # Si pas d'activation fournie, utiliser celle par défaut du réseau
            if activation is None:
                activation = self.activation_default
            
            new_layer = Layer(
                weights_list=weights,
                biases_list=biases,
                activation=activation
            )
            self.layers.append(new_layer)
            return
        
        # Paramètres invalides
        raise ValueError(
            "Usage invalide de add(). Utilisez :\n"
            "  - add(layer=Layer(...)) pour ajouter une couche existante\n"
            "  - add(weights=[...], biases=[...], activation=...) pour créer une couche"
        )
    
    def forward(self, inputs):
        """
        Propagation avant à travers toutes les couches.
        """
        if len(self.layers) == 0:
            raise ValueError("Le réseau est vide ! Ajoutez des couches avec .add()")
        
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        
        return current
    
    def feedforward(self, inputs):
        """
        Alias de forward() pour compatibilité avec main.py.
        """
        return self.forward(inputs)
    
    def summary(self):
        """
        Affiche l'architecture du réseau.
        """
        print("=" * 60)
        print("ARCHITECTURE DU RÉSEAU")
        print("=" * 60)
        
        if len(self.layers) == 0:
            print("⚠️  Réseau vide - Aucune couche ajoutée")
        else:
            for i, layer in enumerate(self.layers):
                num_neurons = len(layer.neurons)
                
                if num_neurons > 0:
                    activation_func = layer.neurons[0].activation
                    if activation_func is not None:
                        activation_name = activation_func.__name__
                    else:
                        activation_name = "identity"
                    
                    num_inputs = len(layer.neurons[0].weights)
                    print(f"Layer {i+1}: {num_inputs} inputs → {num_neurons} neurones | Activation: {activation_name}")
                else:
                    print(f"Layer {i+1}: Vide")
        
        print("=" * 60)


# ============================================================
# TESTS
# ============================================================
if __name__ == "__main__":
    from activations import sigmoid, relu
    from math import exp
    
    def act_sigmoid(x):
        return 1 / (1 + exp(-x))
    
    print("\n" + "="*60)
    print("TEST : Main.py prof (construction progressive)")
    print("="*60)
    
    x = [1.0, 2.0, 4.0]
    
    # Réseau VIDE au départ
    net = Network(input_size=3, activation=act_sigmoid)
    print(f"Nombre de couches après __init__: {len(net.layers)}")  # Doit être 0
    
    # Ajouter les couches
    net.add(
        weights=[
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1],
        ],
        biases=[0.0, 0.1],
    )
    net.add(
        weights=[
            [0.5, -0.2],
            [-0.3, 0.4],
            [0.1, 0.2],
        ],
        biases=[0.0, 0.1, -0.1],
    )
    net.add(
        weights=[
            [0.3, -0.1, 0.2],
            [-0.5, 0.4, 0.1],
        ],
        biases=[-0.1, 0.0],
    )
    
    net.summary()
    y = net.feedforward(x)
    print(f"\nSorties activées: {y}")
    
    print("\n" + "="*60)
    print("✅ TEST PASSÉ !")
    print("="*60)
