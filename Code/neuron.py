class Neuron:
    """
    Un neurone effectue un produit scalaire + biais.
    Pas d'activation ici, juste le calcul brut z = w·x + b
    """
    def __init__(self, weights, bias=0.0):
        """
        weights : liste des poids [w1, w2, ..., wn]
        bias : biais (scalaire)
        """
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        """
        Calcule z = w1*x1 + w2*x2 + ... + wn*xn + b
        inputs : liste de valeurs [x1, x2, ..., xn]
        Retourne : scalaire (float)
        """
        # Produit scalaire manuel
        z = 0.0
        for w, x in zip(self.weights, inputs):
            z += w * x
        
        # Ajout du biais
        z += self.bias
        
        return z


# Test
if __name__ == "__main__":
    n = Neuron(weights=[0.2, -0.1, 0.4], bias=0.0)
    result = n.forward([1.0, 2.0, 4.0])
    print(f"Résultat neurone : {result}")  # Doit afficher 1.6
