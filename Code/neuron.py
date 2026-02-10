"""
Classe Neuron - Unité de base du réseau
Toyceptron - Projet pédagogique Python pur
"""

from activations import identity

class Neuron:
    def __init__(self, weights, bias=0, activation=identity):
        """
        Initialise un neurone
        - weights : liste des poids [w1, w2, ..., wn]
        - bias : biais (scalaire)
        - activation : fonction d'activation
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def forward(self, inputs):
        """
        Propagation avant (forward pass)
        1. Calcule le produit scalaire inputs · weights
        2. Ajoute le biais
        3. Applique la fonction d'activation
        
        - inputs : liste de valeurs [x1, x2, ..., xn]
        - retourne : sortie scalaire après activation
        """
        # Étape 1 : Produit scalaire
        z = 0
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        
        # Étape 2 : Ajout du biais
        z += self.bias
        
        # Étape 3 : Activation
        output = self.activation(z)
        
        return output


# Tests unitaires
if __name__ == "__main__":
    from activations import relu, sigmoid, heaviside
    
    print("=" * 50)
    print("TEST NEURON")
    print("=" * 50)
    
    # Test 1 : Neurone avec activation identité
    print("\n[Test 1] Neurone identité")
    n1 = Neuron(weights=[0.5, -0.3], bias=0.1, activation=identity)
    result1 = n1.forward([1, 2])
    print(f"Inputs: [1, 2]")
    print(f"Poids: [0.5, -0.3], Biais: 0.1")
    print(f"Résultat: {result1}")
    print(f"Attendu: 0.0 (1*0.5 + 2*(-0.3) + 0.1 = 0.0)")
    
    # Test 2 : Neurone avec ReLU
    print("\n[Test 2] Neurone ReLU")
    n2 = Neuron(weights=[0.5, -0.3], bias=0.1, activation=relu)
    result2 = n2.forward([1, 2])
    print(f"Résultat: {result2}")
    print(f"Attendu: 0.0 (max(0, 0.0) = 0.0)")
    
    # Test 3 : Neurone avec sigmoid
    print("\n[Test 3] Neurone sigmoid")
    n3 = Neuron(weights=[0.5, -0.3], bias=0.1, activation=sigmoid)
    result3 = n3.forward([1, 2])
    print(f"Résultat: {result3:.4f}")
    print(f"Attendu: ~0.5 (sigmoid(0.0) = 0.5)")
    
    print("\n" + "=" * 50)
    print("✅ Tous les tests neuron.py terminés")
    print("=" * 50)
