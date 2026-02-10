<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Sprint Toyceptron - JOUR 3 démarré

Étape 3.2 : Coder network.py - Version simple
Go ! 🚀

Excellent ! On attaque **l'étape 3.2 : network.py version simple**. C'est le moment de faire communiquer toutes tes couches entre elles ! 🧠⚡

## 🎯 Concept : Le réseau = enchaînement de couches

Un réseau de neurones, c'est **simple** :

- **Entrée** → passe dans **Layer 1** → résultat passe dans **Layer 2** → ... → **Sortie finale**
- Chaque couche transforme ses inputs et les transmet à la suivante

Sur [TensorFlow Playground](https://playground.tensorflow.org), c'est **exactement ça** : les colonnes de neurones représentent tes layers, et les valeurs "coulent" de gauche à droite.[^1]

***

## 📐 Architecture de `network.py`

Voici la structure attendue :

```python
from layer import Layer

class Network:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes : liste des tailles [nb_inputs, nb_hidden1, ..., nb_outputs]
                      Ex: [2, 3, 1] = 2 entrées, 3 neurones cachés, 1 sortie
        activations : liste des fonctions d'activation (une par couche)
                      Ex: [relu, sigmoid]
        """
        self.layers = []
        # TODO: créer les couches
    
    def forward(self, inputs):
        """
        Propage les inputs à travers toutes les couches
        """
        # TODO: enchaîner les forward()
        pass
```


***

## ⚙️ Étapes de codage

### **1️⃣ Initialisation : construire les couches**

Dans `__init__`, tu dois créer autant de `Layer` que nécessaire.

**Logique** :

- Si `layer_sizes = [2, 3, 1]` → tu dois créer **2 layers** :
    - Layer 1 : **3 neurones** qui prennent **2 inputs** chacun
    - Layer 2 : **1 neurone** qui prend **3 inputs** (sorties du Layer 1)

**Code à écrire** :

```python
for i in range(len(layer_sizes) - 1):
    num_inputs = layer_sizes[i]
    num_neurons = layer_sizes[i + 1]
    activation = activations[i]
    
    layer = Layer(num_neurons, num_inputs, activation)
    self.layers.append(layer)
```

**Explication** :

- `len(layer_sizes) - 1` : si tu as 3 tailles, tu crées **2 layers**[^1]
- `layer_sizes[i]` = nombre d'entrées pour cette couche
- `layer_sizes[i+1]` = nombre de neurones dans cette couche
- Tu ajoutes chaque `Layer` à `self.layers`[^1]

***

### **2️⃣ Forward pass : faire circuler les données**

Dans `forward()`, tu dois **propager les inputs à travers chaque couche**.

**Principe** :

- Commence avec `current = inputs`
- Pour chaque layer, fais `current = layer.forward(current)`
- À la fin, `current` contient la sortie finale

**Code à écrire** :

```python
def forward(self, inputs):
    current = inputs
    for layer in self.layers:
        current = layer.forward(current)
    return current
```

**Pourquoi ça marche ?**

- `layer.forward()` retourne une **liste** (les sorties des neurones)
- Cette liste devient l'**input** de la couche suivante
- Dernier layer → sortie finale du réseau[^1]

***

## 🧪 Test immédiat

Ajoute ce code **à la fin de `network.py`** :

```python
if __name__ == "__main__":
    from activations import relu, sigmoid
    
    # Réseau : 2 entrées → 3 neurones cachés → 1 sortie
    net = Network(layer_sizes=[2, 3, 1], activations=[relu, sigmoid])
    
    result = net.forward([1.0, 2.0])
    print(f"Sortie du réseau : {result}")
```

**Résultat attendu** :

- Une liste avec **1 valeur** (car 1 neurone en sortie)
- Ex : `[0.734821...]` (valeur bizarre car poids aléatoires)[^1]

***

## ✅ Validation

Ton code fonctionne si :

1. ✅ Le test s'exécute sans erreur
2. ✅ Tu obtiens une liste de la bonne taille (ici `[valeur]`)
3. ✅ Tu peux changer `layer_sizes` et ça s'adapte automatiquement

***

## 🔥 Fichier complet `network.py`

```python
from layer import Layer

class Network:
    def __init__(self, layer_sizes, activations):
        """
        Construit un réseau de neurones multi-couches.
        
        Args:
            layer_sizes: liste [nb_inputs, nb_hidden1, ..., nb_outputs]
            activations: liste de fonctions d'activation (une par couche)
        """
        self.layers = []
        
        # Créer chaque couche
        for i in range(len(layer_sizes) - 1):
            num_inputs = layer_sizes[i]
            num_neurons = layer_sizes[i + 1]
            activation = activations[i]
            
            layer = Layer(num_neurons, num_inputs, activation)
            self.layers.append(layer)
    
    def forward(self, inputs):
        """
        Propagation avant : fait passer les inputs à travers toutes les couches.
        
        Args:
            inputs: liste de valeurs d'entrée
        
        Returns:
            Liste des sorties du réseau
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current


# Test rapide
if __name__ == "__main__":
    from activations import relu, sigmoid
    
    # Réseau 2 → 3 → 1
    net = Network(layer_sizes=[2, 3, 1], activations=[relu, sigmoid])
    result = net.forward([1.0, 2.0])
    print(f"Sortie du réseau : {result}")
    
    # Réseau plus complexe 4 → 5 → 3 → 1
    net2 = Network(layer_sizes=[4, 5, 3, 1], activations=[relu, relu, sigmoid])
    result2 = net2.forward([1.0, 2.0, 3.0, 4.0])
    print(f"Sortie réseau complexe : {result2}")
```


***

## 🚀 Prochaine étape

Une fois que ton test fonctionne, passe à **l'étape 3.3 : intégration avec `main.py`** pour vérifier que ton code est compatible avec le code fourni par ton prof ![^1]

💪 **Tu viens de coder l'architecture complète d'un réseau de neurones.** C'est énorme. Tout est prêt pour la phase de validation !

<div align="center">⁂</div>

[^1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

