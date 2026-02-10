<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🚀 Étape 3.3 : Intégration avec le main.py - Guide Complet

## 🎯 Objectif de cette étape

Faire fonctionner le `main.py` fourni par ton prof avec ton code (`neuron.py`, `layer.py`, `network.py`) sans erreur, et comprendre **chaque partie** de l'intégration.[^1]

***

## 📋 Plan d'action

1. **Analyser le main.py** : comprendre ce qu'il attend
2. **Adapter ton code** : respecter l'API exacte
3. **Tester chaque composant** : Neuron → Layer → Network
4. **Valider les résultats** : vérifier que les sorties sont correctes

***

## 🔍 PARTIE 1 : Analyse du main.py

### **Ce que fait le main.py**[^2]

Le fichier test **3 niveaux** d'abstraction :

```
1. Neuron individuel  → Test du calcul de base
2. Layer (couche)     → Test de plusieurs neurones
3. Network (réseau)   → Test de plusieurs couches enchaînées
```


### **Imports nécessaires**

```python
from network import Network
from layer import Layer
from neuron import Neuron
from activations import act_relu, act_threshold, act_identity
```

⚠️ **Attention** : Le main.py importe depuis `activations` (avec un 's'), vérifie le nom de ton fichier !

***

## 🧱 PARTIE 2 : Implémentation complète

### **📄 1. `neuron.py` - Le composant de base**

```python
class Neuron:
    """
    Un neurone effectue un produit scalaire + biais.
    
    Formule mathématique :
        z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b
    
    Pas d'activation ici ! Le neurone retourne juste z (valeur brute).
    """
    
    def __init__(self, weights, bias=0.0):
        """
        Initialise un neurone avec ses poids et son biais.
        
        Args:
            weights (list): Liste des poids [w1, w2, ..., wn]
            bias (float): Biais du neurone (par défaut 0.0)
        
        Exemple:
            n = Neuron(weights=[0.2, -0.1, 0.4], bias=0.0)
        """
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        """
        Calcule la sortie brute du neurone (forward pass).
        
        Args:
            inputs (list): Vecteur d'entrée [x1, x2, ..., xn]
        
        Returns:
            float: Valeur brute z (non activée)
        
        Exemple:
            >>> n = Neuron([0.2, -0.1, 0.4], 0.0)
            >>> n.forward([1.0, 2.0, 4.0])
            1.6
            
        Explication du calcul :
            z = 0.2*1.0 + (-0.1)*2.0 + 0.4*4.0 + 0.0
            z = 0.2 - 0.2 + 1.6 + 0.0
            z = 1.6
        """
        # Produit scalaire manuel (sans numpy)
        z = 0.0
        for weight, input_value in zip(self.weights, inputs):
            z += weight * input_value
        
        # Ajout du biais
        z += self.bias
        
        return z


# ============================================
# TEST UNITAIRE (ne s'exécute que si lancé directement)
# ============================================
if __name__ == "__main__":
    print("="*50)
    print("TEST NEURON")
    print("="*50)
    
    # Création d'un neurone avec des poids fixés
    n1 = Neuron(weights=[0.2, -0.1, 0.4], bias=0.0)
    
    # Test avec l'entrée du main.py
    x = [1.0, 2.0, 4.0]
    result = n1.forward(x)
    
    print(f"Input: {x}")
    print(f"Poids: {n1.weights}")
    print(f"Biais: {n1.bias}")
    print(f"Résultat: {result}")
    print(f"Attendu: 1.6")
    print(f"✅ Test OK" if abs(result - 1.6) < 0.0001 else "❌ Erreur")
```


#### **💡 Pourquoi ce design ?**

- **Pas d'activation dans le neurone** : Cela permet de réutiliser le même neurone avec différentes activations[^2]
- **Retourne un scalaire** : Un neurone produit une seule valeur, pas une liste
- **Produit scalaire manuel** : On fait `z = Σ(w·x)` sans bibliothèque externe

***

### **📄 2. `layer.py` - Collection de neurones**

```python
from neuron import Neuron


class Layer:
    """
    Une couche = plusieurs neurones qui reçoivent les MÊMES inputs.
    
    Schéma mental:
        Input: [x1, x2, x3]
               ↓   ↓   ↓
           Neuron1  Neuron2  Neuron3
               ↓       ↓       ↓
           [out1,   out2,   out3]
    
    Sur TensorFlow Playground, une colonne de cercles = 1 Layer.
    """
    
    def __init__(self, weights_list, biases_list):
        """
        Initialise une couche avec plusieurs neurones.
        
        Args:
            weights_list (list of lists): 
                Chaque sous-liste = poids d'un neurone
                Exemple: [[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]]
                         └─ neurone 1 ─┘   └─ neurone 2 ─┘
            
            biases_list (list): 
                Liste des biais, un par neurone
                Exemple: [0.0, 0.1]
        
        Contrainte importante:
            len(weights_list) == len(biases_list)
            (autant de lignes de poids que de biais)
        """
        self.neurons = []
        
        # Créer un neurone pour chaque ligne de poids
        for weights, bias in zip(weights_list, biases_list):
            neuron = Neuron(weights=weights, bias=bias)
            self.neurons.append(neuron)
    
    def forward(self, inputs):
        """
        Propage les inputs à travers tous les neurones de la couche.
        
        Args:
            inputs (list): Vecteur d'entrée [x1, x2, ...]
        
        Returns:
            list: Sorties brutes de chaque neurone [z1, z2, ...]
        
        Exemple:
            >>> layer = Layer(
            ...     weights_list=[[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]],
            ...     biases_list=[0.0, 0.1]
            ... )
            >>> layer.forward([1.0, 2.0, 4.0])
            [1.6, 0.7]
        
        Explication:
            - Neurone 1: 0.2*1.0 + (-0.1)*2.0 + 0.4*4.0 + 0.0 = 1.6
            - Neurone 2: (-0.4)*1.0 + 0.3*2.0 + 0.1*4.0 + 0.1 = 0.7
        """
        outputs = []
        
        # Chaque neurone calcule sa sortie indépendamment
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        
        return outputs


# ============================================
# TEST UNITAIRE
# ============================================
if __name__ == "__main__":
    print("="*50)
    print("TEST LAYER")
    print("="*50)
    
    # Création d'une couche avec 2 neurones
    layer = Layer(
        weights_list=[
            [0.2, -0.1, 0.4],   # Poids neurone 1
            [-0.4, 0.3, 0.1],   # Poids neurone 2
        ],
        biases_list=[0.0, 0.1]
    )
    
    # Test
    x = [1.0, 2.0, 4.0]
    result = layer.forward(x)
    
    print(f"Input: {x}")
    print(f"Nombre de neurones: {len(layer.neurons)}")
    print(f"Résultat: {result}")
    print(f"Attendu: [1.6, 0.7]")
    print(f"✅ Test OK" if result == [1.6, 0.7] else "❌ Erreur")
```


#### **💡 Points clés**

- **Tous les neurones reçoivent les mêmes inputs** : C'est le principe d'une couche fully-connected
- **Retourne une liste** : Autant de valeurs que de neurones
- **Pas d'activation** : La couche retourne les valeurs brutes, l'activation se fait dans le Network

***

### **📄 3. `network.py` - Orchestration complète**

```python
from layer import Layer


class Network:
    """
    Un réseau de neurones = plusieurs couches enchaînées.
    
    Flux de données:
        Input → Layer 1 → Activation → Layer 2 → Activation → ... → Output
    
    L'activation est appliquée APRÈS chaque couche, pas DANS la couche.
    """
    
    def __init__(self, input_size, activation):
        """
        Initialise un réseau vide.
        
        Args:
            input_size (int): Nombre d'entrées du réseau (ex: 3)
            activation (function): Fonction d'activation à appliquer
                                   (ex: act_sigmoid, act_relu)
        
        Exemple:
            >>> from math import exp
            >>> def act_sigmoid(x):
            ...     return 1 / (1 + exp(-x))
            >>> net = Network(input_size=3, activation=act_sigmoid)
        """
        self.input_size = input_size
        self.activation = activation
        self.layers = []  # Liste vide, on ajoutera les couches avec add()
    
    def add(self, weights, biases):
        """
        Ajoute une couche au réseau.
        
        Args:
            weights (list of lists): Matrice de poids
                [[w11, w12, ...], [w21, w22, ...], ...]
            biases (list): Vecteur de biais [b1, b2, ...]
        
        Exemple:
            >>> net.add(
            ...     weights=[[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]],
            ...     biases=[0.0, 0.1]
            ... )
        
        Remarque:
            - La première couche doit avoir len(weights[^0]) == input_size
            - Les couches suivantes doivent avoir len(weights[^0]) == 
              nombre de neurones de la couche précédente
        """
        layer = Layer(weights_list=weights, biases_list=biases)
        self.layers.append(layer)
    
    def feedforward(self, inputs):
        """
        Propage les inputs à travers tout le réseau (forward pass).
        
        Étapes:
            1. Calcul brut : Layer.forward(inputs)
            2. Activation : appliquer self.activation à chaque sortie
            3. Les sorties activées deviennent les inputs de la couche suivante
            4. Répéter pour toutes les couches
        
        Args:
            inputs (list): Vecteur d'entrée [x1, x2, ...]
        
        Returns:
            list: Sorties finales activées du réseau
        
        Exemple complet:
            >>> from math import exp
            >>> def act_sigmoid(x):
            ...     return 1 / (1 + exp(-x))
            >>> 
            >>> net = Network(input_size=3, activation=act_sigmoid)
            >>> net.add(weights=[[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]], 
            ...         biases=[0.0, 0.1])
            >>> net.feedforward([1.0, 2.0, 4.0])
            [0.8320183851339245, 0.6681877721681662]
        
        Explication mathématique:
            Couche 1 brut: [1.6, 0.7]
            Couche 1 activé: [sigmoid(1.6), sigmoid(0.7)]
                           = [0.832..., 0.668...]
        """
        current = inputs  # Les inputs initiaux
        
        # Boucle sur chaque couche
        for layer in self.layers:
            # 1. Forward brut (sans activation)
            raw_outputs = layer.forward(current)
            
            # 2. Application de l'activation sur chaque sortie
            activated_outputs = [self.activation(z) for z in raw_outputs]
            
            # 3. Les sorties activées deviennent les inputs de la couche suivante
            current = activated_outputs
        
        # Retourner la dernière sortie activée
        return current


# ============================================
# TEST UNITAIRE COMPLET
# ============================================
if __name__ == "__main__":
    from math import exp
    
    def act_sigmoid(x):
        return 1 / (1 + exp(-x))
    
    print("="*50)
    print("TEST NETWORK")
    print("="*50)
    
    # Création du réseau
    net = Network(input_size=3, activation=act_sigmoid)
    
    # Ajout de la première couche (3 inputs → 2 neurones)
    net.add(
        weights=[
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1],
        ],
        biases=[0.0, 0.1]
    )
    
    # Test
    x = [1.0, 2.0, 4.0]
    result = net.feedforward(x)
    
    print(f"Input: {x}")
    print(f"Nombre de couches: {len(net.layers)}")
    print(f"Résultat: {result}")
    print(f"Attendu: [0.832..., 0.668...]")
    print(f"✅ Test OK" if 0.83 < result[^0] < 0.84 else "❌ Erreur")
```


#### **💡 Logique de feedforward**

```python
# Visualisation du flux
Input: [1.0, 2.0, 4.0]
    ↓
Layer 1 forward → [1.6, 0.7]                    (brut)
    ↓
Activation      → [0.832, 0.668]                 (activé)
    ↓
Layer 2 forward → [0.282, 0.117, 0.116]         (brut)
    ↓
Activation      → [0.570, 0.529, 0.529]         (activé)
    ↓
Layer 3 forward → [0.123, -0.020]               (brut)
    ↓
Activation      → [0.530, 0.494]                (activé - OUTPUT FINAL)
```


***

### **📄 4. `activations.py` - Fonctions d'activation**

```python
def act_identity(x):
    """
    Fonction identité : f(x) = x
    
    Utilisée pour les couches de sortie en régression.
    """
    return x


def act_threshold(x):
    """
    Fonction de seuil (Heaviside) : f(x) = 1 si x ≥ 0, sinon 0
    
    Utilisée dans les perceptrons classiques (portes logiques AND, OR).
    
    Exemple:
        >>> act_threshold(0.5)
        1.0
        >>> act_threshold(-0.2)
        0.0
    """
    return 1.0 if x >= 0 else 0.0


def act_relu(x):
    """
    Fonction ReLU (Rectified Linear Unit) : f(x) = max(0, x)
    
    Très utilisée dans les réseaux modernes (introduit la non-linéarité).
    
    Exemple:
        >>> act_relu(2.5)
        2.5
        >>> act_relu(-1.3)
        0.0
    """
    return max(0.0, x)


# Sigmoid est défini dans le main.py, pas besoin de le dupliquer ici
```


***

## 🧪 PARTIE 3 : Tests et validation

### **Test 1 : Neurone individuel**

```python
# Dans le main.py
n1 = Neuron(weights=[0.2, -0.1, 0.4], bias=0.0)
out_n1 = n1.forward([1.0, 2.0, 4.0])
# Résultat attendu: 1.6
```

**Calcul manuel** :

```
z = 0.2×1.0 + (−0.1)×2.0 + 0.4×4.0 + 0.0
z = 0.2 − 0.2 + 1.6 + 0.0
z = 1.6 ✅
```


***

### **Test 2 : Couche**

```python
layer = Layer(
    weights_list=[[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]],
    biases_list=[0.0, 0.1]
)
raw = layer.forward([1.0, 2.0, 4.0])
# Résultat attendu: [1.6, 0.7]
```

**Calcul manuel** :

```
Neurone 1: 0.2×1.0 − 0.1×2.0 + 0.4×4.0 + 0.0 = 1.6 ✅
Neurone 2: −0.4×1.0 + 0.3×2.0 + 0.1×4.0 + 0.1 = 0.7 ✅
```


***

### **Test 3 : Réseau complet**

```python
net = Network(input_size=3, activation=act_sigmoid)
net.add(weights=[[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]], biases=[0.0, 0.1])
net.add(weights=[[0.5, -0.2], [-0.3, 0.4], [0.1, 0.2]], biases=[0.0, 0.1, -0.1])
net.add(weights=[[0.3, -0.1, 0.2], [-0.5, 0.4, 0.1]], biases=[-0.1, 0.0])

y = net.feedforward([1.0, 2.0, 4.0])
# Résultat attendu: [0.5309442148001715, 0.494901997674804]
```

**Tu as obtenu exactement ça** ✅

***

## 🎯 PARTIE 4 : Checklist finale

| Critère | Statut | Détail |
| :-- | :-- | :-- |
| **Neuron.forward()** | ✅ | Retourne un scalaire brut |
| **Layer.forward()** | ✅ | Retourne une liste de scalaires bruts |
| **Network.add()** | ✅ | Ajoute dynamiquement des couches |
| **Network.feedforward()** | ✅ | Applique activation après chaque couche |
| **Network.layers** | ✅ | Attribut accessible (liste des couches) |
| **Compatibilité main.py** | ✅ | Toutes les sorties correspondent |


***

## 🏆 Résultat final obtenu

```
Input: [1.0, 2.0, 4.0]

--- Test Neuron ---
Neurone h1 (brut): 1.6                         ✅
Neurone h2 (brut): 0.7                         ✅

--- Test Layer ---
Couche (valeurs brutes): [1.6, 0.7]            ✅

--- Test Network ---
Sorties activées : [0.5309..., 0.4949...]      ✅
```

**🎉 TU AS RÉUSSI L'INTÉGRATION COMPLÈTE !**

***

## 💡 Points clés à retenir

### **1. Séparation des responsabilités**

- `Neuron` et `Layer` : calcul brut uniquement
- `Network` : gère l'activation et l'enchaînement


### **2. Architecture modulaire**

```
Neuron (produit scalaire + biais)
    ↓
Layer (collection de Neuron)
    ↓
Network (enchaînement de Layer + activation)
```


### **3. Pas de numpy, juste du Python pur**

- Produit scalaire manuel avec `zip()`
- List comprehensions pour l'activation
- Boucles for pour l'enchaînement

***

## 🚀 Prochaine étape

Tu es maintenant prêt pour l'**Étape 3.4 : Vérifications et ajustements**  où tu vas :[^1]

- Nettoyer ton code
- Ajouter des tests supplémentaires
- Implémenter des exemples classiques (AND, OR, XOR)

**Bravo, ton Toyceptron est opérationnel ! 🔥**

<div align="center">⁂</div>

[^1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

[^2]: main.py

