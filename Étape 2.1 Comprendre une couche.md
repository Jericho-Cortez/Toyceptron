# JOUR 2 - Étape 2.1 : Comprendre une couche 

## Récapitulatif complet de ce qu'on a fait


***

## Structure actuelle du projet

```
Perceptron/Code/
├── activations.py     Créé aujourd'hui
├── neuron.py          Fait JOUR 1
└── layer.py           Créé aujourd'hui
```


***

#  Fichier 1 : `activations.py`

```python
"""
Fonctions d'activation pour les neurones
Chaque fonction prend un nombre et retourne un nombre transformé
"""

def identity(x):
    """
    Fonction identité : retourne x tel quel
    Utilisée quand on ne veut pas modifier la sortie
    Exemple : identity(5) → 5
    """
    return x


def heaviside(x):
    """
    Fonction seuil (Heaviside) : perceptron classique
    Retourne 1 si x >= 0, sinon 0
    Utilisée pour les classifications binaires (oui/non)
    Exemple : heaviside(0.5) → 1, heaviside(-0.2) → 0
    """
    return 1 if x >= 0 else 0


def sigmoid(x):
    """
    Fonction sigmoïde : courbe en S
    Formule : 1 / (1 + e^-x)
    Transforme n'importe quelle valeur entre 0 et 1
    Utilisée pour les probabilités
    Exemple : sigmoid(0) → 0.5, sigmoid(5) → 0.993
    """
    import math
    return 1 / (1 + math.exp(-x))


def relu(x):
    """
    ReLU (Rectified Linear Unit) : la plus populaire
    Formule : max(0, x)
    Si x positif → garde x, si x négatif → met à 0
    Très utilisée dans les réseaux modernes (simple et efficace)
    Exemple : relu(3) → 3, relu(-2) → 0
    """
    return max(0, x)
```


###  Pourquoi ces 4 fonctions ?

| Fonction | Comportement | Usage typique |
| :-- | :-- | :-- |
| **identity** | Ne change rien | Couche de sortie (régression) |
| **heaviside** | Tout ou rien (0 ou 1) | Perceptron classique |
| **sigmoid** | Écrase entre 0 et 1 | Probabilités, classification binaire |
| **relu** | Garde le positif, tue le négatif | Couches cachées (standard moderne) |

**Lien avec TensorFlow Playground :** Quand tu changes l'activation sur le playground, tu bascules entre ces fonctions.

***

#  Fichier 2 : `neuron.py` (rappel JOUR 1)

```python
"""
Classe Neurone : l'unité de base d'un réseau de neurones
Un neurone fait 3 choses :
1. Produit scalaire (multiplie inputs par poids)
2. Ajoute le biais
3. Applique l'activation
"""

from activations import identity


class Neuron:
    def __init__(self, weights, bias=0, activation=identity):
        """
        Initialise un neurone
        
        Paramètres :
        - weights : liste des poids [w1, w2, w3, ...]
                    Un poids par input attendu
        - bias : le biais (décalage), par défaut 0
        - activation : fonction d'activation, par défaut identity
        
        Exemple :
        n = Neuron(weights=[0.5, -0.3], bias=0.1, activation=relu)
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def forward(self, inputs):
        """
        Calcule la sortie du neurone (forward pass)
        
        Étapes mathématiques :
        1. z = w1*x1 + w2*x2 + ... + wn*xn  (produit scalaire)
        2. z = z + bias
        3. output = activation(z)
        
        Paramètres :
        - inputs : liste de valeurs [x1, x2, x3, ...]
                   Doit avoir la même longueur que weights
        
        Retourne : un nombre (la sortie du neurone)
        
        Exemple :
        inputs = [1.0, 2.0]
        weights = [0.5, -0.3]
        bias = 0.1
        z = 1.0*0.5 + 2.0*(-0.3) + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
        Si activation = relu : output = max(0, 0.0) = 0.0
        """
        # Étape 1 : Produit scalaire
        z = 0
        for i in range(len(self.weights)):
            z += self.weights[i] * inputs[i]
        
        # Étape 2 : Ajouter le biais
        z += self.bias
        
        # Étape 3 : Appliquer l'activation
        output = self.activation(z)
        
        return output


# Test du neurone
if __name__ == "__main__":
    from activations import relu, sigmoid
    
    print("=" * 50)
    print("TEST NEURONE")
    print("=" * 50)
    
    # Test 1 : Neurone avec activation identity
    n1 = Neuron(weights=[0.5, -0.3], bias=0.1, activation=identity)
    result1 = n1.forward([1.0, 2.0])
    print(f"Test 1 (identity) : {result1}")  # Attendu : 0.0
    
    # Test 2 : Neurone avec activation ReLU
    n2 = Neuron(weights=[0.5, -0.3], bias=0.1, activation=relu)
    result2 = n2.forward([1.0, 2.0])
    print(f"Test 2 (ReLU) : {result2}")  # Attendu : 0.0
    
    # Test 3 : Neurone avec activation sigmoid
    n3 = Neuron(weights=[0.5, -0.3], bias=0.1, activation=sigmoid)
    result3 = n3.forward([1.0, 2.0])
    print(f"Test 3 (sigmoid) : {result3}")  # Attendu : 0.5
    
    print("=" * 50)
```


###  Logique du Neuron

**Analogie :** Imagine une balance avec plusieurs plateaux :

- Chaque **input** est un poids sur un plateau
- Chaque **weight** est un multiplicateur (importance)
- Le **bias** est un poids fixe toujours présent
- L'**activation** décide comment interpréter le résultat final

**Sur TensorFlow Playground :** Un cercle = un Neuron. Les lignes colorées = les poids. L'épaisseur = l'importance du poids.

***

#  Fichier 3 : `layer.py` (nouveau - JOUR 2)

```python
"""
Classe Layer (Couche) : une collection de neurones
Tous les neurones d'une couche :
- Reçoivent les MÊMES inputs
- Ont leur propres poids/biais
- Produisent chacun une sortie
- Utilisent la même fonction d'activation
"""

from neuron import Neuron


class Layer:
    def __init__(self, num_neurons, num_inputs, activation):
        """
        Initialise une couche de neurones
        
        Paramètres :
        - num_neurons : combien de neurones dans cette couche (ex: 3)
        - num_inputs : combien d'inputs chaque neurone reçoit (ex: 2)
        - activation : fonction d'activation commune à tous les neurones
        
        Logique :
        - Crée num_neurons neurones
        - Chaque neurone a num_inputs poids
        - Pour l'instant : poids fixés à 0.5 (on améliorera à l'étape 2.3)
        
        Exemple :
        layer = Layer(num_neurons=3, num_inputs=2, activation=relu)
        → Crée 3 neurones, chacun attend 2 inputs
        """
        self.neurons = []  # Liste vide pour stocker les neurones
        
        # Boucle : créer num_neurons neurones
        for _ in range(num_neurons):
            # Chaque neurone a num_inputs poids
            # [0.5] * num_inputs → [0.5, 0.5, 0.5, ...] (répète 0.5)
            weights = [0.5] * num_inputs
            
            # Créer le neurone et l'ajouter à la liste
            neuron = Neuron(weights=weights, bias=0.1, activation=activation)
            self.neurons.append(neuron)
    
    def forward(self, inputs):
        """
        Propage les inputs à travers TOUS les neurones de la couche
        
        Paramètres :
        - inputs : liste de valeurs [x1, x2, ...]
                   Sera passée à CHAQUE neurone
        
        Retourne : liste des sorties [out1, out2, out3, ...]
                   Une sortie par neurone
        
        Logique :
        1. Prends les inputs
        2. Passe-les au neurone 1 → obtiens out1
        3. Passe-les au neurone 2 → obtiens out2
        4. ... pour tous les neurones
        5. Retourne [out1, out2, out3, ...]
        
        Exemple :
        inputs = [1.0, 2.0]
        3 neurones dans la couche
        → Chaque neurone calcule sa sortie
        → Résultat : [0.8, 1.2, 0.5] (3 valeurs)
        """
        outputs = []  # Liste vide pour collecter les sorties
        
        # Pour chaque neurone de la couche
        for neuron in self.neurons:
            # Calcule la sortie de ce neurone avec les inputs
            output = neuron.forward(inputs)
            
            # Ajoute cette sortie à la liste
            outputs.append(output)
        
        # Retourne toutes les sorties
        return outputs


# Test de la couche
if __name__ == "__main__":
    from activations import relu, sigmoid
    
    print("=" * 50)
    print("TEST LAYER (COUCHE)")
    print("=" * 50)
    
    # Test 1 : Couche avec 3 neurones, 2 inputs, activation ReLU
    print("\n--- Test 1 : 3 neurones, 2 inputs, ReLU ---")
    layer1 = Layer(num_neurons=3, num_inputs=2, activation=relu)
    result1 = layer1.forward([1.0, 2.0])
    print(f"Inputs : [1.0, 2.0]")
    print(f"Sorties : {result1}")
    print(f"Nombre de sorties : {len(result1)}")
    print(f" On a bien {len(result1)} sorties (une par neurone)")
    
    # Test 2 : Couche avec 5 neurones, 3 inputs, activation sigmoid
    print("\n--- Test 2 : 5 neurones, 3 inputs, sigmoid ---")
    layer2 = Layer(num_neurons=5, num_inputs=3, activation=sigmoid)
    result2 = layer2.forward([0.5, 1.5, -0.5])
    print(f"Inputs : [0.5, 1.5, -0.5]")
    print(f"Sorties : {result2}")
    print(f"Nombre de sorties : {len(result2)}")
    print(f" On a bien {len(result2)} sorties (une par neurone)")
    
    # Test 3 : Vérifier que chaque neurone reçoit les mêmes inputs
    print("\n--- Test 3 : Vérification inputs identiques ---")
    print("Les 3 neurones reçoivent TOUS [1.0, 2.0]")
    print("Mais comme ils ont les mêmes poids (0.5, 0.5), ils donnent le même résultat")
    print("À l'étape 2.3, on aura des poids aléatoires → sorties différentes")
    
    print("=" * 50)
```


###  Logique de la Layer

**Analogie :** Une couche = une équipe de spécialistes qui regardent tous les **mêmes** données, mais chacun donne son propre avis.

```
Inputs : [1.0, 2.0]
        ↓       ↓
   ┌────────────────┐
   │  Neuron 1      │ → 0.8  (avis du spécialiste 1)
   │  Neuron 2      │ → 1.2  (avis du spécialiste 2)
   │  Neuron 3      │ → 0.5  (avis du spécialiste 3)
   └────────────────┘
Outputs : [0.8, 1.2, 0.5]
```

**Sur TensorFlow Playground :** Une colonne verticale de cercles = une Layer. Tous les cercles d'une colonne reçoivent les mêmes lignes qui arrivent.

***

##  Ce qu'on a accompli à l'étape 2.1

| Élément          | Statut | Rôle                                                          |
| :--------------- | :----- | :------------------------------------------------------------ |
| `activations.py` | OK     | 4 fonctions d'activation (identity, heaviside, sigmoid, relu) |
| `neuron.py`      | OK     | Classe Neuron avec forward pass complet                       |
| `layer.py`       | OK     | Classe Layer qui regroupe plusieurs neurones                  |


***

##  Validation finale

Lance chaque fichier pour tester :

```bash
python activations.py  # (pas de test, mais aucune erreur)
python neuron.py       # Affiche les 3 tests de neurones
python layer.py        # Affiche les 3 tests de couches
```

**Résultat attendu pour `layer.py` :**

```
==================================================
TEST LAYER (COUCHE)
==================================================

--- Test 1 : 3 neurones, 2 inputs, ReLU ---
Inputs : [1.0, 2.0]
Sorties : [1.6, 1.6, 1.6]
Nombre de sorties : 3
On a bien 3 sorties (une par neurone)

--- Test 2 : 5 neurones, 3 inputs, sigmoid ---
Inputs : [0.5, 1.5, -0.5]
Sorties : [0.7310585786300049, 0.7310585786300049, ...]
Nombre de sorties : 5
On a bien 5 sorties (une par neurone)
```


***

## Liens théoriques

**Ce que tu as codé correspond exactement à :**

1. **Produit scalaire** : `z = w1*x1 + w2*x2 + ...` (dans `neuron.py`)
2. **Biais** : `z = z + bias` (dans `neuron.py`)
3. **Activation** : `output = f(z)` (dans `neuron.py`)
4. **Parallélisation** : tous les neurones d'une couche calculent en même temps (dans `layer.py`)

**Sur TensorFlow Playground :** Active la visualisation et observe comment les valeurs se propagent colonne par colonne (layer par layer).

