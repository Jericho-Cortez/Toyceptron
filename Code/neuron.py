import random


class Neuron:
    def __init__(self, weights=None, bias=0.0, num_inputs=None, activation=None):
        """
        Crée un neurone.
        
        Paramètres :
            weights (list) : Liste des poids (si fournis)
            bias (float) : Biais du neurone
            num_inputs (int) : Nombre d'entrées (si poids aléatoires)
            activation (function) : Fonction d'activation
        """
        # Gestion des poids
        if weights is not None:
            self.weights = weights
        elif num_inputs is not None:
            # Génération aléatoire
            self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        else:
            raise ValueError("Vous devez fournir soit 'weights' soit 'num_inputs'")
        
        self.bias = bias
        
        # CORRECTION : Activation par défaut = identity
        if activation is None:
            from activation import identity
            self.activation = identity
        else:
            self.activation = activation
    
    def forward(self, inputs):
        """
        Calcule la sortie du neurone.
        
        Paramètres :
            inputs (list) : Vecteur d'entrée
        
        Retourne :
            float : Sortie activée du neurone
        """
        # Vérification des dimensions
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Le nombre d'inputs ({len(inputs)}) ne correspond pas "
                f"au nombre de poids ({len(self.weights)})"
            )
        
        # 1. Produit scalaire (dot product)
        z = 0
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        
        # 2. Ajouter le biais
        z += self.bias
        
        # 3. Appliquer l'activation
        return self.activation(z)

