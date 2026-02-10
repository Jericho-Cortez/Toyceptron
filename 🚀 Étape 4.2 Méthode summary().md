
# 📚 RÉCAPITULATIF COMPLET - Sprint Toyceptron JOUR 4 - Étape 4.2

## 🎯 Objectif de l'étape 4.2

**Ajouter une méthode `summary()` dans la classe `Network`** pour afficher l'architecture du réseau de manière claire et lisible.[^1]

Cette méthode permet de **visualiser rapidement** :

- Combien de couches dans le réseau
- Combien de neurones par couche
- Quelle fonction d'activation par couche
- Les dimensions (inputs → outputs) de chaque couche

***

## 🔍 Pourquoi cette méthode est importante ?

### Inspiration : TensorFlow/Keras

Dans les frameworks professionnels, on fait :

```python
model.summary()
```

Et on obtient un aperçu complet de l'architecture. C'est un **outil de debugging indispensable**.[^1]

### Dans Toyceptron

Tu construis parfois des réseaux complexes :

```python
Network(layer_sizes=[3, 8, 5, 3, 1], activations=[relu, relu, relu, sigmoid])
```

Sans `summary()`, impossible de vérifier rapidement si l'architecture est correcte !

***

## 🛠️ Ce qu'on a fait - Chronologie complète

### **Étape 1 : Diagnostic initial** 🔍

**Problème rencontré :**

```bash
TypeError: Network.__init__() got an unexpected keyword argument 'input_size'
```

**Cause :** Ton `Network.__init__()` n'acceptait pas les paramètres utilisés par le main.py du prof.

**Solution :** Adapter le constructeur pour accepter **plusieurs modes** :

- Mode explicite : `layer_sizes` + `activations`
- Mode automatique : `input_size` + `activation`
- Mode progressif : Réseau vide + `.add()`

***

### **Étape 2 : Adaptation du constructeur** 🔧

#### Code ajouté dans `network.py` :

```python
class Network:
    def __init__(self, layer_sizes=None, activations=None, 
                 input_size=None, hidden_layers=None, output_size=None, activation=None):
        """
        Constructeur flexible pour Network.
        
        TROIS MODES POSSIBLES :
        
        Mode 1 - Architecture explicite (contrôle total) :
            Network(layer_sizes=[2, 4, 1], activations=[relu, sigmoid])
            → Tu contrôles tout : tailles et activations
        
        Mode 2 - Architecture automatique (simple) :
            Network(input_size=3, hidden_layers=[5,3], output_size=1, activation=relu)
            → Construction automatique avec couches cachées
        
        Mode 3 - Construction progressive (main.py prof) :
            net = Network(input_size=3, activation=sigmoid)
            net.add(weights=[...], biases=[...])
            → Réseau vide au départ, rempli avec .add()
        """
        self.layers = []
        self.input_size = input_size
        self.activation_default = activation
        
        # MODE 1 : Architecture explicite
        if layer_sizes is not None:
            if activations is None:
                raise ValueError("Si 'layer_sizes' est fourni, 'activations' est obligatoire")
            
            if len(activations) != len(layer_sizes) - 1:
                raise ValueError(f"Il faut {len(activations)} activations")
            
            for i in range(len(layer_sizes) - 1):
                num_inputs = layer_sizes[i]
                num_neurons = layer_sizes[i + 1]
                activation_func = activations[i]
                
                layer = Layer(
                    num_neurons=num_neurons,
                    num_inputs=num_inputs,
                    activation=activation_func
                )
                self.layers.append(layer)
        
        # MODE 2 : Architecture automatique UNIQUEMENT si hidden_layers OU output_size fournis
        elif input_size is not None and (hidden_layers is not None or output_size is not None):
            if activation is None:
                raise ValueError("'activation' obligatoire")
            
            if hidden_layers is None:
                hidden_layers = []
            if output_size is None:
                output_size = 1
            
            all_sizes = [input_size] + hidden_layers + [output_size]
            
            for i in range(len(all_sizes) - 1):
                layer = Layer(
                    num_neurons=all_sizes[i + 1],
                    num_inputs=all_sizes[i],
                    activation=activation
                )
                self.layers.append(layer)
        
        # MODE 3 : Si juste input_size + activation → réseau VIDE (pour .add() après)
        # self.layers reste [] (liste vide)
```


#### Explication détaillée :

**Mode 1 - Explicite** : Pour les tests et la flexibilité maximale

```python
net = Network(layer_sizes=[2, 4, 1], activations=[relu, sigmoid])
```

- `layer_sizes=[2, 4, 1]` → 2 inputs, 4 neurones cachés, 1 output
- `activations=[relu, sigmoid]` → relu pour couche 1, sigmoid pour couche 2
- Les couches sont **créées immédiatement**

**Mode 2 - Automatique** : Pour créer rapidement une architecture

```python
net = Network(input_size=3, hidden_layers=[8, 4], output_size=2, activation=relu)
```

- Crée automatiquement : `[^3] → [^8] → [^4] → [^2]`
- Toutes les couches utilisent `relu`

**Mode 3 - Progressif** : Pour le main.py du prof

```python
net = Network(input_size=3, activation=sigmoid)  # Réseau VIDE
net.add(weights=[...], biases=[...])  # Ajout manuel des couches
```

- `self.layers = []` (vide au départ)
- On stocke juste `input_size` et `activation_default` pour référence

***

### **Étape 3 : Problème avec `.add()`** 🐛

**Erreur suivante :**

```bash
TypeError: Network.add() got an unexpected keyword argument 'weights'
```

**Cause :** Le main.py appelle :

```python
net.add(weights=[...], biases=[...])
```

Mais ta méthode `add()` acceptait seulement un objet `Layer`.

#### Solution : Méthode `add()` flexible

```python
def add(self, layer=None, weights=None, biases=None, activation=None):
    """
    Ajoute une couche au réseau.
    
    Mode 1 - Ajouter un objet Layer :
        net.add(Layer(num_neurons=5, num_inputs=3, activation=relu))
    
    Mode 2 - Créer avec poids fournis (main.py prof) :
        net.add(
            weights=[[0.5, -0.3], [0.2, 0.1]],
            biases=[0.0, 0.1],
            activation=sigmoid  # Optionnel, utilise self.activation_default si None
        )
    """
    # MODE 1 : Ajouter un objet Layer directement
    if layer is not None:
        if not isinstance(layer, Layer):
            raise TypeError("Le paramètre 'layer' doit être un objet de type Layer")
        self.layers.append(layer)
        return
    
    # MODE 2 : Créer une couche à partir de weights + biases
    if weights is not None and biases is not None:
        # Si pas d'activation fournie, utiliser celle par défaut du réseau
        if activation is None:
            activation = self.activation_default
        
        new_layer = Layer(
            weights_list=weights,
            biases_list=biases,
            activation=activation
        )
        self.layers.append(new_layer)
        return
    
    # Paramètres invalides
    raise ValueError(
        "Usage invalide de add(). Utilisez :\n"
        "  - add(layer=Layer(...)) pour ajouter une couche existante\n"
        "  - add(weights=[...], biases=[...], activation=...) pour créer une couche"
    )
```


#### Explication détaillée :

**Logique de la méthode :**

1. **Si `layer` fourni** → Ajouter directement
2. **Si `weights` + `biases` fournis** → Créer une couche avec ces poids, puis ajouter
3. **Sinon** → Erreur (paramètres invalides)

**Point clé** : Si `activation=None`, on utilise `self.activation_default` (stocké dans `__init__`).

***

### **Étape 4 : Adaptation de `layer.py`** 🔧

**Problème :**

```bash
TypeError: 'NoneType' object is not callable
```

**Cause :** Le main.py crée des `Layer` avec `weights_list` et `biases_list`, mais ton constructeur ne supportait pas ces paramètres.

#### Solution : Constructeur flexible pour `Layer`

```python
class Layer:
    def __init__(self, num_neurons=None, num_inputs=None, activation=None, 
                 weights_list=None, biases_list=None):
        """
        Crée une couche de neurones.
        
        Mode 1 - Poids aléatoires :
            Layer(num_neurons=3, num_inputs=2, activation=relu)
        
        Mode 2 - Poids fournis (main.py prof) :
            Layer(
                weights_list=[[0.5, -0.3], [0.2, 0.1]],
                biases_list=[0.0, 0.1],
                activation=sigmoid  # Optionnel
            )
        """
        self.neurons = []
        
        # MODE 1 : Créer avec poids aléatoires
        if num_neurons is not None and num_inputs is not None:
            if activation is None:
                from activations import identity
                activation = identity
            
            for _ in range(num_neurons):
                neuron = Neuron(num_inputs=num_inputs, activation=activation)
                self.neurons.append(neuron)
        
        # MODE 2 : Créer avec poids fournis
        elif weights_list is not None and biases_list is not None:
            num_neurons = len(weights_list)
            
            if len(biases_list) != num_neurons:
                raise ValueError(f"Il faut autant de biais ({len(biases_list)}) que de poids ({num_neurons})")
            
            for i in range(num_neurons):
                neuron = Neuron(weights=weights_list[i], bias=biases_list[i], activation=activation)
                self.neurons.append(neuron)
        
        else:
            raise ValueError(
                "Vous devez fournir soit :\n"
                "  - num_neurons + num_inputs (poids aléatoires)\n"
                "  - weights_list + biases_list (poids fournis)"
            )
```


#### Explication détaillée :

**Mode 1** : Construction classique avec poids aléatoires

```python
layer = Layer(num_neurons=3, num_inputs=2, activation=relu)
```

- Crée 3 neurones
- Chaque neurone a 2 poids (générés aléatoirement)
- Activation : relu

**Mode 2** : Construction avec poids fixes (main.py)

```python
layer = Layer(
    weights_list=[[0.5, -0.3], [0.2, 0.1]],
    biases_list=[0.0, 0.1]
)
```

- Crée 2 neurones (longueur de `weights_list`)
- Neurone 1 : poids `[0.5, -0.3]`, biais `0.0`
- Neurone 2 : poids `[0.2, 0.1]`, biais `0.1`

***

### **Étape 5 : Fix de `neuron.py`** 🐛

**Problème :**

```bash
TypeError: 'NoneType' object is not callable
```

**Cause :** Quand on crée un `Neuron` sans `activation`, `self.activation = None`, et dans `forward()` on fait `self.activation(z)` → **CRASH** !

#### Solution : Activation par défaut

```python
class Neuron:
    def __init__(self, weights=None, bias=0.0, num_inputs=None, activation=None):
        """
        Crée un neurone.
        """
        # Gestion des poids
        if weights is not None:
            self.weights = weights
        elif num_inputs is not None:
            self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        else:
            raise ValueError("Vous devez fournir soit 'weights' soit 'num_inputs'")
        
        self.bias = bias
        
        # ✅ CORRECTION : Activation par défaut = identity
        if activation is None:
            from activations import identity
            self.activation = identity
        else:
            self.activation = activation
```


#### Explication détaillée :

**Avant (BUG)** :

```python
self.activation = activation  # Peut être None
```

**Après (FIX)** :

```python
if activation is None:
    self.activation = identity  # Toujours une fonction callable
else:
    self.activation = activation
```

**Pourquoi `identity` ?** C'est la fonction la plus neutre : `f(x) = x` (pas de transformation).

***

### **Étape 6 : Ajout des méthodes utilitaires** 🛠️

#### `forward()` et `feedforward()`

```python
def forward(self, inputs):
    """
    Propagation avant à travers toutes les couches.
    """
    if len(self.layers) == 0:
        raise ValueError("Le réseau est vide ! Ajoutez des couches avec .add()")
    
    current = inputs
    for layer in self.layers:
        current = layer.forward(current)
    
    return current

def feedforward(self, inputs):
    """
    Alias de forward() pour compatibilité avec main.py.
    """
    return self.forward(inputs)
```

**Pourquoi deux méthodes ?**

- `forward()` : Nom standard en deep learning
- `feedforward()` : Nom utilisé par le prof dans le main.py

Les deux font **exactement la même chose** (alias).

***

### **Étape 7 : ENFIN - La méthode `summary()` !** 🎯

C'était l'objectif initial de l'étape 4.2 !

```python
def summary(self):
    """
    Affiche l'architecture du réseau.
    Utile pour vérifier la structure avant d'exécuter.
    """
    print("=" * 60)
    print("ARCHITECTURE DU RÉSEAU")
    print("=" * 60)
    
    if len(self.layers) == 0:
        print("⚠️  Réseau vide - Aucune couche ajoutée")
    else:
        for i, layer in enumerate(self.layers):
            num_neurons = len(layer.neurons)
            
            if num_neurons > 0:
                # Récupérer le nom de la fonction d'activation
                activation_func = layer.neurons[^0].activation
                if activation_func is not None:
                    activation_name = activation_func.__name__
                else:
                    activation_name = "identity"
                
                num_inputs = len(layer.neurons[^0].weights)
                print(f"Layer {i+1}: {num_inputs} inputs → {num_neurons} neurones | Activation: {activation_name}")
            else:
                print(f"Layer {i+1}: Vide")
    
    print("=" * 60)
```


#### Explication ligne par ligne :

```python
for i, layer in enumerate(self.layers):
```

→ Parcourt toutes les couches avec leur index (0, 1, 2...)

```python
num_neurons = len(layer.neurons)
```

→ Compte combien de neurones dans cette couche

```python
activation_func = layer.neurons[^0].activation
```

→ Récupère la fonction d'activation du **premier neurone** (tous les neurones d'une couche ont la même activation)

```python
activation_name = activation_func.__name__
```

→ `__name__` donne le nom de la fonction (ex: `"relu"`, `"sigmoid"`)

```python
num_inputs = len(layer.neurons[^0].weights)
```

→ Compte combien de poids dans le premier neurone = nombre d'inputs de la couche

```python
print(f"Layer {i+1}: {num_inputs} inputs → {num_neurons} neurones | Activation: {activation_name}")
```

→ Affiche : `Layer 1: 3 inputs → 2 neurones | Activation: relu`

***

## 🎉 Résultat final

### Sortie de `net.summary()` :

```
============================================================
ARCHITECTURE DU RÉSEAU
============================================================
Layer 1: 2 inputs → 4 neurones | Activation: relu
Layer 2: 4 inputs → 3 neurones | Activation: relu
Layer 3: 3 inputs → 1 neurones | Activation: sigmoid
============================================================
```


### Visualisation mentale :

```
Input (2 valeurs)
    ↓
[Layer 1] 4 neurones (ReLU)
    ↓ (4 sorties)
[Layer 2] 3 neurones (ReLU)
    ↓ (3 sorties)
[Layer 3] 1 neurone (Sigmoid)
    ↓
Output (1 valeur)
```


***

## 📚 Fichiers finaux complets

### 1️⃣ **neuron.py** (complet)

```python
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
            self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        else:
            raise ValueError("Vous devez fournir soit 'weights' soit 'num_inputs'")
        
        self.bias = bias
        
        # Activation par défaut = identity
        if activation is None:
            from activations import identity
            self.activation = identity
        else:
            self.activation = activation
    
    def forward(self, inputs):
        """
        Calcule la sortie du neurone.
        
        Formule : activation(Σ(wi * xi) + bias)
        """
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
```


***

### 2️⃣ **layer.py** (complet)

```python
from neuron import Neuron


class Layer:
    def __init__(self, num_neurons=None, num_inputs=None, activation=None, 
                 weights_list=None, biases_list=None):
        """
        Crée une couche de neurones.
        
        Mode 1 - Poids aléatoires :
            Layer(num_neurons=3, num_inputs=2, activation=relu)
        
        Mode 2 - Poids fournis :
            Layer(
                weights_list=[[0.5, -0.3], [0.2, 0.1]],
                biases_list=[0.0, 0.1],
                activation=sigmoid
            )
        """
        self.neurons = []
        
        # MODE 1 : Créer avec poids aléatoires
        if num_neurons is not None and num_inputs is not None:
            if activation is None:
                from activations import identity
                activation = identity
            
            for _ in range(num_neurons):
                neuron = Neuron(num_inputs=num_inputs, activation=activation)
                self.neurons.append(neuron)
        
        # MODE 2 : Créer avec poids fournis
        elif weights_list is not None and biases_list is not None:
            num_neurons = len(weights_list)
            
            if len(biases_list) != num_neurons:
                raise ValueError(f"Il faut autant de biais que de neurones")
            
            for i in range(num_neurons):
                neuron = Neuron(
                    weights=weights_list[i],
                    bias=biases_list[i],
                    activation=activation
                )
                self.neurons.append(neuron)
        
        else:
            raise ValueError(
                "Vous devez fournir soit :\n"
                "  - num_neurons + num_inputs (poids aléatoires)\n"
                "  - weights_list + biases_list (poids fournis)"
            )
    
    def forward(self, inputs):
        """
        Propagation avant à travers tous les neurones de la couche.
        
        Retourne une liste de sorties (une par neurone).
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return outputs
```


***

### 3️⃣ **network.py** (complet et final)

```python
from layer import Layer


class Network:
    def __init__(self, layer_sizes=None, activations=None, 
                 input_size=None, hidden_layers=None, output_size=None, activation=None):
        """
        Constructeur flexible pour Network.
        
        Mode 1 - Architecture explicite :
            Network(layer_sizes=[2, 4, 1], activations=[relu, sigmoid])
        
        Mode 2 - Architecture automatique :
            Network(input_size=3, hidden_layers=[5,3], output_size=1, activation=relu)
        
        Mode 3 - Construction progressive :
            net = Network(input_size=3, activation=sigmoid)
            net.add(weights=[...], biases=[...])
        """
        self.layers = []
        self.input_size = input_size
        self.activation_default = activation
        
        # MODE 1 : Architecture explicite
        if layer_sizes is not None:
            if activations is None:
                raise ValueError("'activations' obligatoire")
            
            if len(activations) != len(layer_sizes) - 1:
                raise ValueError(f"Il faut {len(activations)} activations")
            
            for i in range(len(layer_sizes) - 1):
                layer = Layer(
                    num_neurons=layer_sizes[i + 1],
                    num_inputs=layer_sizes[i],
                    activation=activations[i]
                )
                self.layers.append(layer)
        
        # MODE 2 : Architecture automatique
        elif input_size is not None and (hidden_layers is not None or output_size is not None):
            if activation is None:
                raise ValueError("'activation' obligatoire")
            
            if hidden_layers is None:
                hidden_layers = []
            if output_size is None:
                output_size = 1
            
            all_sizes = [input_size] + hidden_layers + [output_size]
            
            for i in range(len(all_sizes) - 1):
                layer = Layer(
                    num_neurons=all_sizes[i + 1],
                    num_inputs=all_sizes[i],
                    activation=activation
                )
                self.layers.append(layer)
        
        # MODE 3 : Réseau vide (pour .add() progressif)
        # self.layers reste []
    
    def add(self, layer=None, weights=None, biases=None, activation=None):
        """
        Ajoute une couche au réseau.
        
        Mode 1 : net.add(Layer(...))
        Mode 2 : net.add(weights=[...], biases=[...], activation=...)
        """
        # MODE 1
        if layer is not None:
            if not isinstance(layer, Layer):
                raise TypeError("'layer' doit être un objet Layer")
            self.layers.append(layer)
            return
        
        # MODE 2
        if weights is not None and biases is not None:
            if activation is None:
                activation = self.activation_default
            
            new_layer = Layer(
                weights_list=weights,
                biases_list=biases,
                activation=activation
            )
            self.layers.append(new_layer)
            return
        
        raise ValueError("Paramètres invalides pour add()")
    
    def forward(self, inputs):
        """
        Propagation avant à travers toutes les couches.
        """
        if len(self.layers) == 0:
            raise ValueError("Le réseau est vide !")
        
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        
        return current
    
    def feedforward(self, inputs):
        """
        Alias de forward() pour compatibilité.
        """
        return self.forward(inputs)
    
    def summary(self):
        """
        Affiche l'architecture du réseau.
        """
        print("=" * 60)
        print("ARCHITECTURE DU RÉSEAU")
        print("=" * 60)
        
        if len(self.layers) == 0:
            print("⚠️  Réseau vide - Aucune couche ajoutée")
        else:
            for i, layer in enumerate(self.layers):
                num_neurons = len(layer.neurons)
                
                if num_neurons > 0:
                    activation_func = layer.neurons[^0].activation
                    activation_name = activation_func.__name__ if activation_func else "identity"
                    num_inputs = len(layer.neurons[^0].weights)
                    
                    print(f"Layer {i+1}: {num_inputs} inputs → {num_neurons} neurones | Activation: {activation_name}")
                else:
                    print(f"Layer {i+1}: Vide")
        
        print("=" * 60)
```


***

## 🎓 Concepts clés compris

### 1. **Architecture multi-modes**

Tu as appris à concevoir une classe flexible qui supporte **plusieurs interfaces** sans duplication de code.

### 2. **Gestion des valeurs par défaut**

```python
if activation is None:
    activation = self.activation_default
```

→ Mécanisme de **fallback** intelligent.

### 3. **Introspection Python**

```python
activation_func.__name__  # Récupère le nom de la fonction
```

→ Utilisation de `__name__` pour l'affichage.

### 4. **Validation robuste**

Checks de dimensions, types, cohérence des paramètres → **code production-ready**.

***

## ✅ Validation finale - Étape 4.2 COMPLÈTE

### Ce qui fonctionne parfaitement :

✅ **Neuron** : Calcul, activation, poids fixes/aléatoires
✅ **Layer** : Construction multi-modes, forward pass
✅ **Network** : 3 modes de construction, add(), forward(), feedforward()
✅ **summary()** : Affichage clair de l'architecture ← **OBJECTIF ATTEINT** 🎯
✅ **Compatibilité** : main.py du prof fonctionne à 100%

***

## 🚀 Prochaine étape

**Étape 4.3 : README.md** (30 min)[^1]

Tu veux :

1. **Passer à la documentation (README.md)** ?
2. **Faire une pause et sauvegarder** ?
3. **Ajouter des bonus** (forward_debug, perceptron AND/OR) ?

Dis-moi ! 💪🔥

<div align="center">⁂</div>

[^1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

