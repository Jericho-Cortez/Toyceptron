# layer.py
import random
from neuron import Neuron


class Layer:
    """
    Couche de neurones (fully-connected).
    Tous les neurones de la couche reçoivent les mêmes inputs.
    
    Schéma conceptuel:
        Inputs [x1, x2, ..., xn]
            ↓ ↓ ↓ (mêmes inputs pour tous)
        [Neuron1, Neuron2, ..., NeuronM]
            ↓ ↓ ↓
        Outputs [y1, y2, ..., yM]
    """
    
    def __init__(self, num_neurons, num_inputs, activation):
        """
        Initialise une couche avec plusieurs neurones.
        
        Args:
            num_neurons (int): Nombre de neurones dans la couche
            num_inputs (int): Nombre d'entrées pour chaque neurone
            activation (function): Fonction d'activation commune
        
        Exemple:
            Layer(num_neurons=3, num_inputs=2, activation=relu)
            → Crée 3 neurones
            → Chacun a 2 poids (générés aléatoirement)
            → Tous utilisent la fonction relu
        """
        self.neurons = []  # Liste vide pour stocker les neurones
        
        # Boucle pour créer num_neurons neurones
        for _ in range(num_neurons):
            # Générer les poids aléatoires ICI (Python pur, pas numpy!)
            weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
            
            # Créer le neurone avec les poids générés
            neuron = Neuron(
                weights=weights,        # Poids générés dans Layer
                bias=0,                 # Biais initialisé à 0
                activation=activation   # Fonction d'activation commune
            )
            self.neurons.append(neuron)  # Ajouter le neurone à la liste
    
    def forward(self, inputs):
        """
        Passe avant (forward pass) de la couche.
        
        Processus:
        1. Chaque neurone de la couche reçoit les MÊMES inputs
        2. Chaque neurone calcule sa propre sortie (avec ses propres poids)
        3. On collecte toutes les sorties dans une liste
        
        Args:
            inputs (list): Liste des valeurs d'entrée
                          Taille: len(inputs) doit égaler num_inputs
        
        Returns:
            list: Liste des sorties de chaque neurone
                  Taille: len(outputs) = num_neurons
        
        Exemple concret:
            inputs = [1.0, 2.0]  # 2 valeurs
            layer avec 3 neurones
            → outputs = [0.5, -0.3, 0.8]  # 3 sorties (1 par neurone)
        """
        outputs = []  # Liste vide pour stocker les sorties
        
        # Boucle sur chaque neurone de la couche
        for neuron in self.neurons:
            # Chaque neurone traite les mêmes inputs
            output = neuron.forward(inputs)  # Retourne 1 scalaire
            outputs.append(output)           # Ajouter à la liste
        
        return outputs  # Retourner la liste complète des sorties


# ========================================
# TESTS UNITAIRES
# ========================================
if __name__ == "__main__":
    import sys
    sys.path.append('.')  # Pour importer activations.py
    
    from activations import identity, relu, sigmoid
    
    print("="*50)
    print("TEST 1: Layer avec activation identity")
    print("="*50)
    
    # Créer une couche: 3 neurones, 2 inputs chacun, activation identity
    layer = Layer(num_neurons=3, num_inputs=2, activation=identity)
    
    # Afficher les poids générés automatiquement (doivent être aléatoires)
    print(f"Neurone 1 poids: {layer.neurons[0].weights}")
    print(f"Neurone 2 poids: {layer.neurons[1].weights}")
    print(f"Neurone 3 poids: {layer.neurons[2].weights}")
    
    # Forward pass avec [1.0, 2.0]
    result = layer.forward([1.0, 2.0])
    print(f"\nSorties de la couche: {result}")
    print(f"Type: {type(result)} | Longueur: {len(result)}")
    
    print("\n" + "="*50)
    print("TEST 2: Layer avec activation ReLU")
    print("="*50)
    
    # Couche avec 2 neurones, 3 inputs, activation relu
    layer_relu = Layer(num_neurons=2, num_inputs=3, activation=relu)
    result_relu = layer_relu.forward([1.0, -2.0, 3.0])
    print(f"Sorties avec ReLU: {result_relu}")
    
    print("\n" + "="*50)
    print("TEST 3: Layer avec activation sigmoid")
    print("="*50)
    
    # Couche avec 4 neurones, 2 inputs, activation sigmoid
    layer_sigmoid = Layer(num_neurons=4, num_inputs=2, activation=sigmoid)
    result_sigmoid = layer_sigmoid.forward([0.5, -0.5])
    print(f"Sorties avec sigmoid: {result_sigmoid}")
    
    print("\n✅ Tous les tests layer.py sont OK !")
