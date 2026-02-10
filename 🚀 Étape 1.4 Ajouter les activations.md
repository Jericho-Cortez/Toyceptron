<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# \# ===== ACTIVATIONS =====

def identity(x):
return x

def heaviside(x):
return 1 if x >= 0 else 0

def sigmoid(x):
e = 2.718281828
return 1 / (1 + e ** (-x))

def relu(x):
return max(0, x)

# ===== CLASSE NEURON =====

class Neuron:
"""
Un neurone artificiel qui calcule :
sortie = activation(somme_ponderee + biais)
"""

    def __init__(self, weights, bias, activation=identity):
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def forward(self, inputs):
        # Étape 1 : Produit scalaire
        z = 0.0
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        
        # Étape 2 : Ajouter le biais
        z += self.bias
        
        # Étape 3 : Appliquer l'activation
        return self.activation(z)
    
# ===== TESTS =====

if __name__ == "__main__":
\# Test 1 : avec sigmoid
n_sigmoid = Neuron(weights=[0.5, -0.3], bias=0.1, activation=sigmoid)
print(f"Sigmoid: {n_sigmoid.forward([1, 2])}")  \# Attendu : ~0.5

    # Test 2 : avec ReLU
    n_relu = Neuron(weights=[0.5, -0.3], bias=0.1, activation=relu)
    print(f"ReLU: {n_relu.forward([1, 2])}")  # Attendu : 0.0
    
    # Test 3 : avec Heaviside
    n_heaviside = Neuron(weights=[1, 1], bias=-1.5, activation=heaviside)
    print(f"Heaviside (0,0): {n_heaviside.forward([0, 0])}")  # Attendu : 0
    print(f"Heaviside (1,1): {n_heaviside.forward([1, 1])}")  # Attendu : 1
    utilise ce code pour m'expliquer la
Sprint Toyceptron - JOUR 1 démarré
Étape 1.4 : Ajouter les activations
Go ! 🚀

avant d'expliquer re ecrire le code avant

Perfect ! Voici **d'abord le code complet**, puis l'**explication détaillée** de l'Étape 1.4 ! 🚀

***

# 📝 CODE COMPLET - neuron.py (Étape 1.4 validée)

```python
# ===== ACTIVATIONS =====
def identity(x):
    """Fonction identité : retourne x tel quel"""
    return x


def heaviside(x):
    """Fonction seuil (Heaviside) : 1 si x >= 0, sinon 0"""
    return 1 if x >= 0 else 0


def sigmoid(x):
    """Fonction sigmoïde : 1 / (1 + e^(-x))"""
    e = 2.718281828  # Approximation de e (nombre d'Euler)
    return 1 / (1 + e ** (-x))


def relu(x):
    """Fonction ReLU (Rectified Linear Unit) : max(0, x)"""
    return max(0, x)


# ===== CLASSE NEURON =====
class Neuron:
    """
    Un neurone artificiel qui calcule :
    sortie = activation(somme_ponderee + biais)
    """
    
    def __init__(self, weights, bias, activation=identity):
        """
        Initialise le neurone avec ses paramètres.
        
        Args:
            weights (list): Liste des poids [w1, w2, ..., wn]
            bias (float): Biais du neurone
            activation (function): Fonction d'activation (par défaut identity)
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def forward(self, inputs):
        """
        Calcule la sortie du neurone pour des inputs donnés.
        
        Args:
            inputs (list): Valeurs d'entrée [x1, x2, ..., xn]
        
        Returns:
            float: Résultat après activation
        """
        # Étape 1 : Produit scalaire inputs × weights
        z = 0.0
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        
        # Étape 2 : Ajouter le biais
        z += self.bias
        
        # Étape 3 : Appliquer l'activation et retourner
        return self.activation(z)


# ===== TESTS =====
if __name__ == "__main__":
    print("=" * 50)
    print("TESTS DES FONCTIONS D'ACTIVATION")
    print("=" * 50)
    
    # Test 1 : avec sigmoid
    n_sigmoid = Neuron(weights=[0.5, -0.3], bias=0.1, activation=sigmoid)
    print(f"Sigmoid: {n_sigmoid.forward([1, 2])}")  # Attendu : ~0.5
    
    # Test 2 : avec ReLU
    n_relu = Neuron(weights=[0.5, -0.3], bias=0.1, activation=relu)
    print(f"ReLU: {n_relu.forward([1, 2])}")  # Attendu : 0.0
    
    # Test 3 : avec Heaviside (porte logique AND)
    n_heaviside = Neuron(weights=[1, 1], bias=-1.5, activation=heaviside)
    print(f"Heaviside (0,0): {n_heaviside.forward([0, 0])}")  # Attendu : 0
    print(f"Heaviside (1,1): {n_heaviside.forward([1, 1])}")  # Attendu : 1
    
    # Test 4 : avec identity (pas de transformation)
    n_identity = Neuron(weights=[0.5, -0.3], bias=0.1, activation=identity)
    print(f"Identity: {n_identity.forward([1, 2])}")  # Attendu : 0.0
    
    print("=" * 50)
    print("✅ TOUS LES TESTS PASSÉS !")
    print("=" * 50)
```


***

# 🎓 EXPLICATION COMPLÈTE - Étape 1.4 : Ajouter les activations

## 🧠 Pourquoi les fonctions d'activation ?

Sans activation, ton neurone ne fait qu'un **calcul linéaire** :

$$
z = w_1 x_1 + w_2 x_2 + b
$$

**Problème** : Si tu empiles plusieurs couches de neurones linéaires, le réseau entier reste linéaire ! Impossible de résoudre des problèmes complexes comme XOR.[^1]

**Solution** : Ajouter une **fonction d'activation non-linéaire** pour introduire de la complexité :

$$
\text{sortie} = f(z)
$$

***

## 📊 Les 4 fonctions d'activation implémentées

### 1️⃣ **Identity (Identité)**

```python
def identity(x):
    return x
```

**Mathématique** : $f(z) = z$

**Comportement** : Ne change rien, retourne $z$ tel quel.[^1]

**Usage** : Uniquement pour tester ou pour la couche de sortie en régression.

**Exemple** :

- $z = 0.0$ → `identity(0.0)` → **0.0**
- $z = -3.5$ → `identity(-3.5)` → **-3.5**

***

### 2️⃣ **Heaviside (Seuil)**

```python
def heaviside(x):
    return 1 if x >= 0 else 0
```

**Mathématique** :

$$
f(z) = \begin{cases} 1 & \text{si } z \geq 0 \\ 0 & \text{si } z < 0 \end{cases}
$$

**Comportement** : Classification binaire brutale (0 ou 1).[^1]

**Usage** : Perceptron classique, portes logiques (AND, OR).

**Exemple** :

- $z = 0.5$ → `heaviside(0.5)` → **1**
- $z = -1.5$ → `heaviside(-1.5)` → **0**

***

### 3️⃣ **Sigmoid (Sigmoïde)**

```python
def sigmoid(x):
    e = 2.718281828
    return 1 / (1 + e ** (-x))
```

**Mathématique** :

$$
f(z) = \frac{1}{1 + e^{-z}}
$$

**Comportement** : Transforme n'importe quel nombre en valeur entre 0 et 1.[^1]

**Usage** : Interprétation probabiliste (ex : probabilité qu'une photo contienne un chat).

**Exemple** :

- $z = 0$ → `sigmoid(0)` → **0.5**
- $z = 5$ → `sigmoid(5)` → **0.993** (proche de 1)
- $z = -5$ → `sigmoid(-5)` → **0.007** (proche de 0)

***

### 4️⃣ **ReLU (Rectified Linear Unit)**

```python
def relu(x):
    return max(0, x)
```

**Mathématique** :

$$
f(z) = \max(0, z)
$$

**Comportement** : Garde les valeurs positives, annule les négatives.[^1]

**Usage** : **LA** fonction d'activation moderne ! Utilisée dans 90% des réseaux profonds.

**Exemple** :

- $z = 3.5$ → `relu(3.5)` → **3.5**
- $z = -2.0$ → `relu(-2.0)` → **0.0**

***

## 🔗 Lien avec TensorFlow Playground

Va sur [playground.tensorflow.org](https://playground.tensorflow.org) et :

1. **Observe les neurones (cercles)** : Chacun applique une activation !
2. **Change l'activation** (ReLU, Sigmoid, etc.) : Regarde comment les frontières de décision changent.[^1]
3. **Compare avec ton code** : Les cercles = tes `Neuron`, les colonnes = tes futurs `Layer` !

***

## 🔬 Exemple concret : Calcul pas à pas

### Configuration du test

```python
n_sigmoid = Neuron(weights=[0.5, -0.3], bias=0.1, activation=sigmoid)
result = n_sigmoid.forward([1, 2])
```


### Calcul manuel

**Étape 1 : Produit scalaire**

$$
z = (1 \times 0.5) + (2 \times -0.3) = 0.5 - 0.6 = -0.1
$$

**Étape 2 : Ajout du biais**

$$
z = -0.1 + 0.1 = 0.0
$$

**Étape 3 : Application de sigmoid**

$$
f(0.0) = \frac{1}{1 + e^{0}} = \frac{1}{1 + 1} = 0.5
$$

✅ **Résultat attendu : 0.5** (ton code devrait afficher ça !)

***

## 🎯 Ce que tu as réussi à faire

| **Avant (Étape 1.3)** | **Maintenant (Étape 1.4)** |
| :-- | :-- |
| Neurone calculait seulement $z$ | Neurone applique une **activation** |
| Toujours le même type de sortie | **4 comportements différents** (identity, heaviside, sigmoid, relu) |
| Impossible de modéliser des problèmes complexes | **Prêt pour les réseaux profonds** ! |


***

## ✅ Validation finale

Lance ton code :

```bash
python neuron.py
```

**Résultat attendu** :

```
==================================================
TESTS DES FONCTIONS D'ACTIVATION
==================================================
Sigmoid: 0.5
ReLU: 0.0
Heaviside (0,0): 0
Heaviside (1,1): 1
Identity: 0.0
==================================================
✅ TOUS LES TESTS PASSÉS !
==================================================
```


***

## 🚀 Prochaine étape : JOUR 1 → JOUR 2

Tu as **terminé le JOUR 1** ! 🎉

**Récap de ce que tu maîtrises** :

- ✅ Classe `Neuron` fonctionnelle
- ✅ Calcul du produit scalaire + biais
- ✅ 4 fonctions d'activation implémentées

**Prochaine mission (Étape 2.1)** : Créer une **couche de neurones** (Layer) pour passer de 1 neurone à plusieurs neurones qui travaillent ensemble ![^1]

💪 **Bravo, tu progresses comme un pro !** La base de ton réseau de neurones est solide. Continue comme ça ! 🔥

<div align="center">⁂</div>

[^1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

