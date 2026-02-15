## Étape 3.1 : Comprendre le réseau multi-couches

### Le concept fondamental

Un réseau multi-couches, c'est **une chaîne de Layer** où :

- La **sortie** d'une couche devient l'**entrée** de la suivante[^1]
- Chaque couche transforme progressivement les données
- La dernière couche produit la prédiction finale


### Schéma mental (forward pass)

```
Input Layer     →     Hidden Layer    →    Output Layer
[x1, x2, x3]    →    [h1, h2, h3, h4] →       [y1]
   (3 inputs)         (4 neurones)          (1 sortie)
```

**Le flux** :

1. Tu donnes `[x1, x2, x3]` au réseau
2. `Layer1` calcule `[h1, h2, h3, h4]` via ses 4 neurones
3. `Layer2` reçoit `[h1, h2, h3, h4]` et calcule `[y1]`

***

##  Visualisation sur TensorFlow Playground

**Action immédiate** : Va sur https://playground.tensorflow.org[^1]

**Observe précisément** :

1. **Clique sur Play** : regarde les données circuler de gauche à droite
2. **Les colonnes de cercles** = les **layers** (couches)
3. **Les lignes colorées** = les **poids** (bleu = négatif, orange = positif)
4. **Chaque cercle** = un **neurone** avec son activation

**Expérimente** :

- Change le nombre de couches cachées (ajoute/supprime des colonnes)
- Modifie le nombre de neurones par couche
- Change les fonctions d'activation (ReLU, Sigmoid, etc.)
- Observe comment le réseau apprend à séparer les données

***

##  Lien avec ton code Python

### Ce que tu as déjà codé

**Neuron** : calcule `f(w·x + b)` → **UN cercle** sur le playground
**Layer** : groupe de neurones → **UNE colonne** de cercles
**Network** : enchaînement de layers → **TOUT le playground**

### Architecture attendue pour `network.py`

```python
from layer import Layer

class Network:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: [3, 4, 1] = 3 inputs, 4 hidden, 1 output
        activations: [relu, sigmoid] = relu pour hidden, sigmoid pour output
        """
        self.layers = []
        # Créer chaque couche en cascade
        pass
    
    def forward(self, inputs):
        """
        Fait circuler les données à travers toutes les couches
        """
        pass
```


***

##  Exemple numérique concret

**Configuration** :

- Input : `[1.0, 2.0]` (2 valeurs)
- Architecture : `[2, 3, 1]` = 2 inputs → 3 hidden → 1 output
- Activations : `[relu, sigmoid]`

**Déroulé** :

1. **Layer 1** (3 neurones, 2 inputs chacun) :
    - Neuron 1 : calcule `relu(1*w1 + 2*w2 + b1)` → ex: `0.8`
    - Neuron 2 : calcule `relu(1*w3 + 2*w4 + b2)` → ex: `0.0`
    - Neuron 3 : calcule `relu(1*w5 + 2*w6 + b3)` → ex: `1.2`
    - **Sortie Layer 1** : `[0.8, 0.0, 1.2]`
2. **Layer 2** (1 neurone, 3 inputs) :
    - Neuron 1 : calcule `sigmoid(0.8*w7 + 0.0*w8 + 1.2*w9 + b4)` → ex: `0.73`
    - **Sortie finale** : `[0.73]`

***

##  Mathématiques derrière le multi-couches

Pour chaque layer $l$ :

$$
\text{output}^{(l)} = f^{(l)}(\mathbf{W}^{(l)} \cdot \text{input}^{(l)} + \mathbf{b}^{(l)})
$$

Puis :

$$
\text{input}^{(l+1)} = \text{output}^{(l)}
$$

**En Python** :

```python
current = inputs
for layer in self.layers:
    current = layer.forward(current)
return current
```


***

## Validation de compréhension

**Tu dois pouvoir répondre** :

1. Combien de layers dans un réseau `[5, 10, 3]` ? → **2 layers** (une hidden, une output)
2. Si Layer1 a 4 neurones, combien de valeurs en sortie ? → **4 valeurs**
3. Si Layer1 sort `[a, b, c]`, combien d'inputs pour chaque neurone de Layer2 ? → **3 inputs**


