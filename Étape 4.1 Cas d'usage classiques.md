## Objectif de l'étape

Démontrer que ton réseau fonctionne vraiment en implémentant **2 exemples classiques** :

1. **Perceptron AND** (porte logique) avec poids fixés
2. **Démonstration XOR impossible** avec un seul neurone

***

## 1. Perceptron AND — Poids fixés

### Concept mathématique

La porte AND retourne 1 seulement si **toutes les entrées valent 1** :

| x₁ | x₂ | Sortie |
| :-- | :-- | :-- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**Comment un neurone peut apprendre ça ?**

- Produit scalaire : $z = w_1 x_1 + w_2 x_2 + b$
- Activation : Heaviside (seuil) → retourne 1 si $z \geq 0$, sinon 0

**Poids magiques** :

- $w_1 = 1, w_2 = 1, b = -1.5$
- Si $x_1 = 0, x_2 = 0$ → $z = 0 + 0 - 1.5 = -1.5$ → sortie = **0**
- Si $x_1 = 1, x_2 = 1$ → $z = 1 + 1 - 1.5 = 0.5$ → sortie = **1**

***

### Code à implémenter

Crée un fichier **`test_and.py`** :

```python
from neuron import Neuron
from activations import heaviside

# Neurone configuré pour résoudre AND
neuron_and = Neuron(
    weights=[1, 1],
    bias=-1.5,
    activation=heaviside
)

# Table de vérité
print("=" * 40)
print("PERCEPTRON AND - Table de vérité")
print("=" * 40)

tests = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

for x1, x2, expected in tests:
    result = neuron_and.forward([x1, x2])
    status = "✓" if result == expected else "✗"
    print(f"{status} AND({x1}, {x2}) = {result} (attendu: {expected})")

print("=" * 40)
```

**Validation attendue** :

```
✓ AND(0, 0) = 0 (attendu: 0)
✓ AND(0, 1) = 0 (attendu: 0)
✓ AND(1, 0) = 0 (attendu: 0)
✓ AND(1, 1) = 1 (attendu: 1)
```


***

## 2. Impossibilité de XOR avec un seul neurone

### Pourquoi XOR est impossible ?

La porte XOR :

| x₁ | x₂ | Sortie |
| :-- | :-- | :-- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Problème** : Ces points ne sont **pas linéairement séparables**. Aucune ligne droite ne peut séparer les 0 des 1.

Sur le playground TensorFlow, si tu essaies avec **0 hidden layer**, le réseau échoue toujours sur XOR. Il faut **au moins 1 couche cachée** pour créer une frontière de décision non-linéaire.

***

### Démonstration code

Crée **`test_xor_impossible.py`** :

```python
from neuron import Neuron
from activations import heaviside

print("=" * 50)
print("DÉMONSTRATION : XOR IMPOSSIBLE avec 1 neurone")
print("=" * 50)

# Essayons plusieurs configurations de poids
configs = [
    {"weights": [1, 1], "bias": -0.5, "name": "Conf 1"},
    {"weights": [1, -1], "bias": 0, "name": "Conf 2"},
    {"weights": [-1, 1], "bias": 0, "name": "Conf 3"},
]

xor_truth_table = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

for config in configs:
    n = Neuron(
        weights=config["weights"],
        bias=config["bias"],
        activation=heaviside
    )
    
    print(f"\n{config['name']} → w={config['weights']}, b={config['bias']}")
    errors = 0
    
    for x1, x2, expected in xor_truth_table:
        result = n.forward([x1, x2])
        status = "✓" if result == expected else "✗"
        if result != expected:
            errors += 1
        print(f"  {status} XOR({x1}, {x2}) = {result} (attendu: {expected})")
    
    print(f"  → Erreurs : {errors}/4")

print("\n" + "=" * 50)
print("CONCLUSION : Impossible de résoudre XOR avec")
print("un seul neurone. Il faut une couche cachée !")
print("=" * 50)
```

**Sortie attendue** :
Toutes les configurations auront **au moins 1 erreur**.

***

## 3. BONUS : XOR avec couche cachée

Si tu veux aller plus loin, démontre que **XOR devient possible avec 2 couches** :

```python
from network import Network
from activations import heaviside

# Architecture : 2 inputs → 2 hidden → 1 output
net = Network(
    layer_sizes=[2, 2, 1],
    activations=[heaviside, heaviside]
)

# Avec des poids aléatoires, ce ne sera pas parfait
# Mais tu peux montrer que l'architecture PERMET de résoudre XOR
# (même si sans entraînement, les résultats sont incorrects)

print("\nXOR avec couche cachée (poids non entraînés) :")
for x1, x2, expected in [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]:
    result = net.forward([x1, x2])[^0]
    print(f"XOR({x1}, {x2}) = {result:.2f} (attendu: {expected})")

print("\n⚠️ Sans entraînement (backpropagation), les poids sont aléatoires.")
print("Mais l'ARCHITECTURE permet mathématiquement de résoudre XOR !")
```


***

## Checklist de validation

- [x] `test_and.py` affiche 4/4 réussites ✓
- [x] `test_xor_impossible.py` montre que XOR échoue avec 1 neurone
- [x] Tu comprends **pourquoi** XOR nécessite une couche cachée
- [x] Tu peux expliquer le lien avec le playground TensorFlow


