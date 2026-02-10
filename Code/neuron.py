# neuron.py (VERSION CORRIGÉE)
import random
from activations import identity

class Neuron:
    """
    Un neurone effectue un produit scalaire + biais + activation.
    z = w·x + b
    output = f(z)
    """
    def __init__(self, weights=None, num_inputs=None, bias=0.0, activation=identity):
        """
        Deux modes d'initialisation :
        
        MODE 1 (automatique) :
            num_inputs (int) : Nombre d'entrées
            → Les poids sont générés aléatoirement
        
        MODE 2 (manuel) :
            weights (list) : Liste des poids [w1, w2, ..., wn]
            → Les poids sont fournis explicitement
        
        Args:
            weights (list, optional) : Poids manuels
            num_inputs (int, optional) : Nombre d'entrées pour génération auto
            bias (float) : Biais du neurone
            activation (function) : Fonction d'activation
        """
        # Gestion des poids
        if weights is None:
            if num_inputs is None:
                raise ValueError("Fournir soit 'weights' soit 'num_inputs'")
            # Initialisation aléatoire entre -1 et 1
            self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        else:
            self.weights = weights
        
        self.bias = bias
        self.activation = activation
    
    def forward(self, inputs):
        """
        Calcule la sortie du neurone.
        
        Étapes :
        1. Produit scalaire : z = Σ(wi * xi)
        2. Ajout du biais : z = z + b
        3. Activation : output = f(z)
        
        Args:
            inputs (list) : Vecteur d'entrée [x1, x2, ..., xn]
        Returns:
            float : Sortie activée du neurone
        """
        # Produit scalaire manuel
        z = 0.0
        for w, x in zip(self.weights, inputs):
            z += w * x
        
        # Ajout du biais
        z += self.bias
        
        # Application de l'activation
        return self.activation(z)


# Test
if __name__ == "__main__":
    # Test MODE 2 (manuel)
    n_manual = Neuron(weights=[0.2, -0.1, 0.4], bias=0.0, activation=identity)
    result = n_manual.forward([1.0, 2.0, 4.0])
    print(f"✅ Résultat neurone (manuel) : {result}")  # 1.6
    
    # Test MODE 1 (automatique)
    n_auto = Neuron(num_inputs=3, bias=0.5, activation=identity)
    result_auto = n_auto.forward([1.0, 2.0, 4.0])
    print(f"✅ Résultat neurone (auto) : {result_auto}")
    print(f"   Poids générés : {n_auto.weights}")
