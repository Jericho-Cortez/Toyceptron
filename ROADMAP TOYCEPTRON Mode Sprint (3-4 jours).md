## Philosophie de la roadmap

- **Code d'abord, comprend en faisant**
    
- **Validation à chaque étape** (tests rapides)
    
- **Pas de perfectionnisme** : on vise le fonctionnel avant l'élégant
    
- **Révision POO intégrée** dans chaque étape
    

---

## JOUR 1 : Setup + Neurone (3-4h)

## Étape 1.1 : Remise à niveau POO Express (30 min)

**Objectif** : Rafraîchir les bases Python et POO nécessaires au projet[playground.tensorflow+1](https://playground.tensorflow.org/)

**Actions concrètes :**

- Ouvre [https://learnxinyminutes.com/python/](https://learnxinyminutes.com/python/) et relis rapidement :
    
    - Les listes (`[]`, append, boucles `for`)
        
    - Les classes (`class`, `__init__`, `self`)
        
    - Les méthodes (`def method(self, param):`)
        
- **Test rapide** : Crée un fichier `test_poo.py` et code :
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self):
        return (self.x**2 + self.y**2)**0.5

p = Point(3, 4)
print(p.distance())  # Doit afficher 5.0
```

**Validation** : Si ton code fonctionne, tu es prêt pour la suite !

---

## Étape 1.2 : Comprendre le Perceptron (1h)

**Objectif** : Saisir la logique mathématique d'un neurone

**Actions concrètes :**

1. **Va sur [https://playground.tensorflow.org**](https://playground.tensorflow.org%2A%2A/)[playground.tensorflow+1](https://playground.tensorflow.org/)
    
    - Clique sur "Play" et regarde le réseau s'entraîner
        
    - Observe les **poids** (lignes bleues/oranges) et les **neurones** (cercles)
        
    - Change le nombre de neurones/couches et observe
        
2. **Note ces 3 formules clés** (c'est TOUTE la magie du perceptron) :
    
    - **Produit scalaire** : z=w1⋅x1+w2⋅x2+...+wn⋅xnz = w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_nz=w1⋅x1+w2⋅x2+...+wn⋅xn
        
    - **Ajout du biais** : z=z+bz = z + bz=z+b
        
    - **Activation** : sortie=f(z)sortie = f(z)sortie=f(z) où fff peut être sigmoid, ReLU, etc.[[tensorflow](https://www.tensorflow.org/guide/core/mlp_core)]​
        
3. **Exemple concret à la main** :
    
    - Inputs : `[1, 2]`
        
    - Poids : `[0.5, -0.3]`
        
    - Biais : `0.1`
        
    - Calcul : z=1×0.5+2×(−0.3)+0.1=0.5−0.6+0.1=0.0z = 1 \times 0.5 + 2 \times (-0.3) + 0.1 = 0.5 - 0.6 + 0.1 = 0.0z=1×0.5+2×(−0.3)+0.1=0.5−0.6+0.1=0.0
        
    - Activation ReLU : max(0,0.0)=0.0max(0, 0.0) = 0.0max(0,0.0)=0.0
        

**Validation** : Tu dois pouvoir expliquer "un neurone = produit scalaire + biais + activation"

---

## Étape 1.3 : Coder `neuron.py` (1h30)

**Objectif** : Implémenter ta première classe `Neuron`

**Architecture attendue :**

```python
class Neuron:
    def __init__(self, weights, bias):
        # Stocker poids et biais
        pass
    
    def forward(self, inputs):
        # 1. Produit scalaire inputs × weights
        # 2. Ajouter le biais
        # 3. Retourner le résultat (sans activation pour l'instant)
        pass
```

**Actions concrètes :**

1. Crée le fichier `neuron.py` dans VS Code
    
2. **Code le `__init__`** :
    
    - Stocke `self.weights` (liste)
        
    - Stocke `self.bias` (nombre)
        
3. **Code la méthode `forward`** :
    
    - Fais une boucle `for` pour calculer le produit scalaire
        
    - Ajoute le biais
        
    - Retourne la somme
        
4. **Test immédiat** :
```python
# À la fin de neuron.py
if __name__ == "__main__":
    n = Neuron(weights=[0.5, -0.3], bias=0.1)
    result = n.forward([1, 2])
    print(f"Résultat: {result}")  # Doit afficher 0.0
```

**Validation** : Ton test affiche `0.0` ? Bravo, ton neurone calcule correctement !

---

## Étape 1.4 : Ajouter les activations (1h)

**Objectif** : Rendre ton neurone "intelligent" avec des fonctions d'activation

**Fonctions à coder (dans `neuron.py` ou un fichier séparé `activations.py`) :**

```python
def identity(x):
    return x

def heaviside(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + (2.718281828 ** (-x)))  # e ≈ 2.718...

def relu(x):
    return max(0, x)
```

**Modifier la classe `Neuron` :**

```python
class Neuron:
    def __init__(self, weights, bias, activation=identity):
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def forward(self, inputs):
        z = # ... ton calcul précédent
        return self.activation(z)  # <-- AJOUT ICI
```

**Test avec différentes activations :**

```python
n_sigmoid = Neuron([0.5, -0.3], 0.1, activation=sigmoid)
n_relu = Neuron([0.5, -0.3], 0.1, activation=relu)
print(n_sigmoid.forward([1, 2]))  # ~0.5
print(n_relu.forward([1, 2]))     # 0.0
```

**Validation** : Tu peux changer l'activation et voir des résultats différents !

---

## JOUR 2 : Layer + Tests (3-4h)

## Étape 2.1 : Comprendre une couche (30 min)

**Concept clé :** Une couche (`Layer`) = plusieurs neurones qui reçoivent les MÊMES inputs[[tensorflow](https://www.tensorflow.org/guide/core/mlp_core)]​

**Schéma mental :**

```text
Inputs: [x1, x2, x3]
         ↓   ↓   ↓
       ┌─────┴─────┬─────┐
       ↓           ↓     ↓
    Neuron1    Neuron2  Neuron3  ← 3 neurones dans la couche
       ↓           ↓     ↓
    [out1]      [out2] [out3]    ← Sortie = liste de 3 valeurs
```

**Sur le playground TensorFlow :**

- Une colonne de cercles = 1 layer
    
- Chaque cercle = 1 neurone
    
- Tous les cercles d'une colonne reçoivent les mêmes inputs[playground.tensorflow+1](https://playground.tensorflow.org/)
    

---

## Étape 2.2 : Coder `layer.py` (1h30)

**Architecture attendue :**

```python
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs, activation):
        # Créer une liste de neurones
        # Tous avec le même nombre d'inputs
        pass
    
    def forward(self, inputs):
        # Appeler forward() sur chaque neurone
        # Retourner une liste des sorties
        pass
```

**Actions concrètes :**

1. Crée `layer.py`
    
2. **Dans `__init__` :**
    
    - Crée une liste vide `self.neurons = []`
        
    - Boucle `for` pour créer `num_neurons` neurones
        
    - Chaque neurone a `num_inputs` poids (initialise avec des 0.5 pour tester)
        
3. **Dans `forward` :**
    
    - Crée une liste vide `outputs = []`
        
    - Pour chaque neurone, appelle `neuron.forward(inputs)`
        
    - Ajoute le résultat à `outputs`
        
    - Retourne `outputs`
        

**Test immédiat :**

```python
if __name__ == "__main__":
    from activations import relu
    layer = Layer(num_neurons=3, num_inputs=2, activation=relu)
    result = layer.forward([1.0, 2.0])
    print(f"Sorties de la couche: {result}")  # Liste de 3 valeurs

```

**Validation** : Tu dois voir `[valeur1, valeur2, valeur3]`

---

## Étape 2.3 : Initialisation intelligente des poids (1h)

**Problème :** Tous les poids à 0.5, c'est pas terrible

**Solutions à implémenter :**

1. **Poids aléatoires :**

```python
import random
random.seed(42)  # Pour reproductibilité
weight = random.uniform(-1, 1)  # Entre -1 et 1

```
   
2. **Modifier `Neuron.__init__` pour accepter `None` :**
   
```python
def __init__(self, weights=None, num_inputs=None, bias=0, activation=identity):
    if weights is None:
        # Générer des poids aléatoires
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
    else:
        self.weights = weights
    # ...

```

**Test :**

python

`n = Neuron(num_inputs=3)  # Poids auto-générés print(n.weights)  # Doit afficher 3 valeurs aléatoires`

**Validation** : Chaque création de neurone donne des poids différents

---

## Étape 2.4 : Tests unitaires basiques (1h)

**Objectif :** Vérifier que ton code marche avec des cas simples

**Crée `test_manual.py` :**

```python
from neuron import Neuron
from layer import Layer
from activations import identity, relu

# Test 1: Neurone avec poids fixés
n = Neuron(weights=[1, 1], bias=0, activation=identity)
assert n.forward([2, 3]) == 5, "Erreur calcul neurone"
print("Test neurone OK")

# Test 2: Couche avec 2 neurones
# ...

```

**Validation** : Tous tes `print("...")` s'affichent sans erreur

---

##  JOUR 3 : Network + Integration (4-5h)

##  Étape 3.1 : Comprendre le réseau multi-couches (30 min)

**Concept clé :** Les sorties d'une couche = inputs de la couche suivante[[tensorflow](https://www.tensorflow.org/guide/core/mlp_core)]​

**Schéma :**

```text
Input Layer    Hidden Layer    Output Layer
[x1, x2, x3] → [h1, h2, h3, h4] → [y1]
              ↑                   ↑
          sorties Layer1 =   sorties Layer2 =
          inputs Layer2      sortie finale

```

**Sur le playground :**

- Observe comment les valeurs "coulent" de gauche à droite
    
- C'est la **forward pass** (passage avant)[[youtube](https://www.youtube.com/watch?v=wmqTVC17KV4)]​
    

---

## Étape 3.2 : Coder `network.py` - Version simple (2h)

**Architecture attendue :**

```python
from layer import Layer

class Network:
    def __init__(self, layer_sizes, activations):
        # layer_sizes = [3, 4, 1] → 3 inputs, 4 hidden, 1 output
        # activations = [relu, sigmoid]
        self.layers = []
        # Créer les couches
        pass
    
    def forward(self, inputs):
        # Passer inputs dans chaque couche successivement
        pass

```

**Actions concrètes :**

1. **Dans `__init__` :**

```python
for i in range(len(layer_sizes) - 1):
    num_inputs = layer_sizes[i]
    num_neurons = layer_sizes[i + 1]
    activation = activations[i]
    layer = Layer(num_neurons, num_inputs, activation)
    self.layers.append(layer)

```

2. **Dans `forward` :**

```python
current = inputs
for layer in self.layers:
    current = layer.forward(current)
return current

```

**Test :**

```python
if __name__ == "__main__":
    from activations import relu, sigmoid
    net = Network(
        layer_sizes=[2, 3, 1],
        activations=[relu, sigmoid]
    )
    result = net.forward([1.0, 2.0])
    print(f"Sortie du réseau: {result}")  # Liste avec 1 valeur

```

**Validation** : Tu obtiens une sortie (même si elle est bizarre pour l'instant)

---

## Étape 3.3 : Vérifications et ajustements (1h)

**Checklist finale :**

-  Le réseau accepte n'importe quelle architecture (modifie `layer_sizes` dans `main.py`)
    
-  Les activations sont bien appliquées
    
-  Les poids sont différents à chaque exécution (sauf si `random.seed()`)
    
-  Le code est commenté (au moins les méthodes principales)
    
-  Chaque fichier a un `if __name__ == "__main__":` avec un test
    

**Validation** : Ton code est propre et fonctionne

---

## JOUR 4 : Finitions + Documentation (2-3h)

## Étape 4.1 : Cas d'usage classiques (1h)

**Implémente 2 exemples :**

1. **Perceptron AND (porte logique) :**

```python
# Poids fixés pour résoudre AND
n = Neuron(weights=[1, 1], bias=-1.5, activation=heaviside)
print(n.forward([0, 0]))  # 0
print(n.forward([1, 1]))  # 1
```

2. **Impossible XOR avec 1 neurone :**
    
    - Montre qu'un seul neurone ne peut pas résoudre XOR
        
    - Explique pourquoi il faut une couche cachée
        

---

## Étape 4.2 : Méthode `summary()` (optionnel, 30 min)

**Afficher l'architecture du réseau :**

```python
def summary(self):
    print("=" * 50)
    print("ARCHITECTURE DU RÉSEAU")
    print("=" * 50)
    for i, layer in enumerate(self.layers):
        print(f"Layer {i+1}: {len(layer.neurons)} neurones")
    print("=" * 50)
```

---

## Étape 4.3 : README.md (30 min)

**Rédige un fichier README simple :**

```text
# Toyceptron - Perceptron Multi-Couches

## Description
Implémentation d'un réseau de neurones en Python pur (sans librairies).

## Utilisation
```bash
python main.py
```

## Architecture

- `neuron.py` : Classe Neuron
    
- `layer.py` : Classe Layer
    
- `network.py` : Classe Network
    
- `activations.py` : Fonctions d'activation
    

## Auteur

[Ton nom] - [Date]


***

### Étape 4.4 : Relecture et test final (1h)
**Actions :**
1. Relis ton code du début à la fin
2. Teste tous les cas d'usage
3. Vérifie que le `main.py` fonctionne
4. Prépare ta présentation (si nécessaire)

**Validation finale** : Ton projet est prêt à être rendu ! 

***

## RÉSUMÉ ULTRA-SYNTHÉTIQUE

| Jour | Focus | Livrable |
|------|-------|----------|
| **J1** | Neuron + Activations | `neuron.py` fonctionnel |
| **J2** | Layer + Tests | `layer.py` fonctionnel |
| **J3** | Network + Integration | `network.py` + `main.py` OK |
| **J4** | Finitions + Doc | Projet complet |

***

## PIÈGES À ÉVITER

1. **Dimensions incompatibles** : Vérifie toujours que `len(inputs) == len(weights)`
2. **Oublier `self`** : En POO, n'oublie jamais `self.` devant les attributs
3. **Boucles infinies** : Teste avec de petites architectures d'abord
4. **Confusion liste/scalaire** : `forward()` d'un neurone retourne un nombre, pas une liste

***

## ASTUCES DE DERNIÈRE MINUTE

- **Bloqué sur un bug ?** → Utilise `print()` partout pour voir les valeurs
- **Code qui plante ?** → Vérifie les dimensions avec `print(len(...))` 
- **Résultat bizarre ?** → C'est normal sans entraînement, les poids sont aléatoires
- **Manque de temps ?** → Concentre-toi sur Neuron → Layer → Network, le reste est bonus

***

## RESSOURCES RAPIDES

- **Refresh Python POO** : https://learnxinyminutes.com/python/
- **Visualiser** : https://playground.tensorflow.org
- **Référence technique** : PDF dans l'espace

***
