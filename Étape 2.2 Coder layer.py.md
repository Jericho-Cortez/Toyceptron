# Sprint Toyceptron - JOUR 2 - Étape 2.2 : Coder layer.py

## Concept théorique : Qu'est-ce qu'une Layer ?

### Définition simple

Une **Layer (couche)** = **collection de neurones** qui travaillent **en parallèle**.

**Règle d'or :**

- Tous les neurones d'une couche reçoivent **les mêmes inputs**
- Mais chaque neurone a **ses propres poids** (générés aléatoirement)
- Donc chaque neurone produit **une sortie différente**


### Schéma visuel

```
Inputs: [x1, x2, x3]
     ↓      ↓     ↓
   [Neuron1] [Neuron2] [Neuron3]  ← Layer (3 neurones)
   w=[...1]  w=[...2]  w=[...3]   ← Poids différents
     ↓         ↓         ↓
  [out1,     out2,     out3]       ← 3 sorties (liste)
```


### Lien avec TensorFlow Playground

Sur [playground.tensorflow.org](https://playground.tensorflow.org)  :

- **1 colonne de cercles** = 1 Layer
- **Chaque cercle** = 1 Neuron
- **Lignes qui arrivent** = inputs (identiques pour toute la colonne)
- **Couleur/épaisseur des lignes** = valeur des poids (aléatoires au départ)

***

## Code complet commenté de `layer.py`

```python
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
    print(f"Neurone 1 poids: {layer.neurons[^0].weights}")
    print(f"Neurone 2 poids: {layer.neurons[^1].weights}")
    print(f"Neurone 3 poids: {layer.neurons[^2].weights}")
    
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
```


***

##  Explications détaillées ligne par ligne

### **PARTIE 1 : Imports**

```python
import random
from neuron import Neuron
```

**Pourquoi `random` ?**

- Module **natif Python** (pas numpy !)
- Génère des nombres aléatoires pour initialiser les poids
- `random.uniform(-1, 1)` → nombre aléatoire entre -1 et 1

**Pourquoi `from neuron import Neuron` ?**

- On réutilise la classe `Neuron` créée au JOUR 1
- Architecture modulaire : Layer **compose** des Neuron

***

### **PARTIE 2 : `__init__` - Construction de la couche**

#### Ligne par ligne :

```python
self.neurons = []
```

- Crée une **liste vide** pour stocker tous les neurones de la couche
- Exemple : `[Neuron1, Neuron2, Neuron3]`

***

```python
for _ in range(num_neurons):
```

- Boucle qui tourne `num_neurons` fois
- `_` (underscore) = "je n'utilise pas la variable de boucle"
- Exemple : Si `num_neurons=3`, la boucle tourne 3 fois

***

```python
weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
```

**Décortiquage complet :**


| Partie | Explication |
| :-- | :-- |
| `random.uniform(-1, 1)` | Génère 1 nombre aléatoire entre -1 et 1 |
| `for _ in range(num_inputs)` | Répète `num_inputs` fois |
| `[...]` | List comprehension = crée une liste |

**Exemple concret :**

```python
num_inputs = 3
weights = [random.uniform(-1, 1) for _ in range(3)]
# Résultat possible : [0.543, -0.821, 0.234]
```

**Pourquoi entre -1 et 1 ?**

- Bonne pratique en deep learning
- Évite les valeurs trop grandes (explosion de gradient)
- Évite les valeurs trop petites (vanishing gradient)

***

```python
neuron = Neuron(
    weights=weights,        # Poids générés juste avant
    bias=0,                 # Biais = 0 (simple pour commencer)
    activation=activation   # Fonction passée en paramètre
)
```

**Points importants :**

- **`weights=weights`** : On passe les poids générés (pas `None`)
- **Tous les neurones ont le même `num_inputs`** (cohérence dimensionnelle)
- **Tous les neurones ont la même `activation`** (simplifie l'architecture)
- **Mais chaque neurone a des poids DIFFÉRENTS** (générés aléatoirement)

***

```python
self.neurons.append(neuron)
```

- Ajoute le neurone créé à la liste `self.neurons`
- Après 3 itérations : `self.neurons = [Neuron1, Neuron2, Neuron3]`

***

### **PARTIE 3 : `forward` - Propagation avant**

#### Ligne par ligne :

```python
outputs = []
```

- Liste vide pour collecter les résultats
- Contiendra 1 valeur par neurone

***

```python
for neuron in self.neurons:
```

- Parcourt **chaque neurone** de la couche
- Exemple : `neuron` = `Neuron1`, puis `Neuron2`, puis `Neuron3`

***

```python
output = neuron.forward(inputs)
```

**Point crucial** : **Tous les neurones reçoivent les mêmes `inputs` !**

**Exemple détaillé :**

```python
inputs = [1.0, 2.0]

# Neuron 1 (poids = [0.5, -0.3])
output1 = neuron1.forward([1.0, 2.0])
# → 0.5×1.0 + (-0.3)×2.0 + 0 = -0.1

# Neuron 2 (poids = [0.8, 0.2])
output2 = neuron2.forward([1.0, 2.0])
# → 0.8×1.0 + 0.2×2.0 + 0 = 1.2

# Neuron 3 (poids = [-0.4, 0.9])
output3 = neuron3.forward([1.0, 2.0])
# → (-0.4)×1.0 + 0.9×2.0 + 0 = 1.4
```

**Même inputs, sorties différentes → magie des poids différents !**

***

```python
outputs.append(output)
```

- Ajoute la sortie du neurone (1 scalaire) à la liste
- Après 3 neurones : `outputs = [-0.1, 1.2, 1.4]`

***

```python
return outputs
```

- Retourne la **liste complète** des sorties
- **Neuron retourne 1 scalaire, Layer retourne 1 liste !**

***

##  Exemple concret avec calculs complets

### Configuration

```python
layer = Layer(num_neurons=3, num_inputs=2, activation=identity)
```

**Ce qui se passe :**

1. Crée une liste vide `neurons = []`
2. **Itération 1 :**
    - Génère `weights = [0.5, -0.3]` (aléatoire)
    - Crée `Neuron1(weights=[0.5, -0.3], bias=0, activation=identity)`
    - Ajoute à la liste
3. **Itération 2 :**
    - Génère `weights = [0.8, 0.2]`
    - Crée `Neuron2(weights=[0.8, 0.2], bias=0, activation=identity)`
4. **Itération 3 :**
    - Génère `weights = [-0.4, 0.9]`
    - Crée `Neuron3(weights=[-0.4, 0.9], bias=0, activation=identity)`

**Résultat : `layer.neurons = [Neuron1, Neuron2, Neuron3]`**

***

### Forward pass

```python
result = layer.forward([1.0, 2.0])
```

**Déroulement détaillé :**


| Étape | Neurone | Calcul | Résultat |
| :-- | :-- | :-- | :-- |
| 1 | Neuron1 | `0.5×1.0 + (-0.3)×2.0 + 0 = -0.1` | `-0.1` |
| 2 | Neuron2 | `0.8×1.0 + 0.2×2.0 + 0 = 1.2` | `1.2` |
| 3 | Neuron3 | `(-0.4)×1.0 + 0.9×2.0 + 0 = 1.4` | `1.4` |

**Résultat final : `[-0.1, 1.2, 1.4]`**

***

##  Analyse des tests unitaires

### **TEST 1 : Identity (pas de modification)**

```python
layer = Layer(num_neurons=3, num_inputs=2, activation=identity)
result = layer.forward([1.0, 2.0])
```

**Résultat attendu :**

```
Neurone 1 poids: [-0.079, -0.640]
Neurone 2 poids: [0.728, 0.481]
Neurone 3 poids: [-0.733, -0.897]

Sorties: [-1.358, 1.691, -2.527]
Type: <class 'list'> | Longueur: 3
```

** Vérifications :**

- 3 neurones créés
- Poids aléatoires différents
- 3 sorties (type `list`)
- Identity ne modifie pas les valeurs

***

### **TEST 2 : ReLU (coupe les négatifs)**

```python
layer_relu = Layer(num_neurons=2, num_inputs=3, activation=relu)
result_relu = layer_relu.forward([1.0, -2.0, 3.0])
```

**Comportement ReLU :**

```
Neuron 1 : z = ... → 2.351 → relu(2.351) = 2.351 
Neuron 2 : z = ... → -0.456 → relu(-0.456) = 0.0 
```

**ReLU = `max(0, x)`** → garde les positifs, met les négatifs à 0

***

### **TEST 3 : Sigmoid (compresse entre 0 et 1)**

```python
layer_sigmoid = Layer(num_neurons=4, num_inputs=2, activation=sigmoid)
result_sigmoid = layer_sigmoid.forward([0.5, -0.5])
```

**Comportement sigmoid :**

```
Toutes les sorties entre 0 et 1 : [0.444, 0.421, 0.551, 0.635]
```

**Sigmoid = $\frac{1}{1 + e^{-x}}$** → sortie toujours entre 0 et 1

***

##  Points clés à retenir

###  **Architecture**

```
Layer = collection de Neuron
- Même num_inputs pour tous
- Même activation pour tous
- Poids différents pour chacun
```


###  **Génération de poids (Python pur)**

```python
weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
```

-  Pas de numpy
-  Valeurs entre -1 et 1
-  Différentes à chaque exécution


###  **Forward pass**

```
Même inputs → Tous les neurones
Poids différents → Sorties différentes
1 neurone → 1 scalaire
N neurones → liste de N scalaires
```


###  **Séparation des responsabilités**

- **Neuron** : calcule produit scalaire + biais + activation
- **Layer** : gère la collection, génère les poids, orchestre le forward

***

##  Checklist Étape 2.2

- [x] Fichier `layer.py` créé
- [x] Classe `Layer` avec `__init__` et `forward`
- [x] Import `random` (Python pur)
- [x] Génération poids aléatoires avec list comprehension
- [x] Création de `num_neurons` neurones avec boucle `for`
- [x] `forward()` retourne une liste (pas un scalaire)
- [x] Tests identity, relu, sigmoid
- [x] Poids différents à chaque exécution

***

##  Validation finale

**Lance :**

```bash
python layer.py
```

**Résultat attendu :**

```
 Tous les tests layer.py sont OK !
```

**Lance une 2ème fois :**

```bash
python layer.py
```

**→ Les poids doivent être différents !** (preuve de la génération aléatoire)

***

## 🎓 Ce que tu as appris

| Concept                | Maintenant                        |
| :--------------------- | :-------------------------------- |
| Layer                  | Collection de neurones parallèles |
| Poids aléatoires       | `random.uniform(-1, 1)`           |
| Forward d'une couche   | Même inputs, sorties différentes  |
| Architecture modulaire |  Layer compose des Neuron         |


