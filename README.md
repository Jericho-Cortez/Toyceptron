# Toyceptron - Perceptron Multi-Couches

## 📝 Description
Implémentation **from scratch** d'un réseau de neurones multi-couches (perceptron) en **Python pur**, sans aucune bibliothèque externe (numpy, torch, etc.).

L'objectif pédagogique est de **comprendre en profondeur** le fonctionnement interne d'un réseau de neurones en codant chaque composant à la main.

## ✨ Fonctionnalités
- Construction d'un neurone avec poids, biais et fonction d'activation
- Création de couches (layers) avec initialisation aléatoire des poids
- Architecture réseau multi-couches personnalisable
- 4 fonctions d'activation : identité, Heaviside, sigmoïde, ReLU
- Forward pass complète à travers le réseau

## 🏗️Architecture
Toyceptron/
├── neuron.py # Classe Neuron (calcul forward d'un neurone)
├── layer.py # Classe Layer (couche de neurones)
├── network.py # Classe Network (réseau multi-couches)
├── activations.py # Fonctions d'activation
├── main.py # Script de démonstration
└── README.md # Ce fichier

## 🚀 Utilisation
### Lancer le projet
```bash
python main.py

from network import Network
from activations import relu, sigmoid

# Créer un réseau : 2 inputs → 3 hidden → 1 output
net = Network(layer_sizes=, activations=[relu, sigmoid])[1]

# Forward pass
result = net.forward([1.0, 2.0])
print(f"Sortie : {result}")


### 5. **Concepts clés**
```markdown
## 🧠 Concepts clés

### Neurone artificiel
Un neurone effectue 3 opérations :
1. **Produit scalaire** : z = w₁x₁ + w₂x₂ + ... + wₙxₙ
2. **Ajout du biais** : z = z + b
3. **Activation** : sortie = f(z)

### Couche (Layer)
Collection de neurones recevant les **mêmes inputs** et produisant une liste de sorties.

### Réseau (Network)
Empilement de couches où les sorties d'une couche deviennent les inputs de la suivante.

## 📚 Exemples

### Porte logique AND
```python
from neuron import Neuron
from activations import heaviside

# Neurone configuré pour AND
n = Neuron(weights=, bias=-1.5, activation=heaviside)[1]

print(n.forward())  # → 0
print(n.forward())  # → 1[1]


### 7. **Technologies et contraintes**
```markdown
## 🛠️ Technologies
- **Python 3.x** (pur, sans bibliothèques externes)
- Structures de données natives : listes, boucles, fonctions
- POO : classes, méthodes, attributs

## ⚠️ Contraintes
- **Aucune bibliothèque externe** (numpy, torch, sklearn interdits)
- Tout est codé à la main pour maximiser l'apprentissage
- Pas d'optimisation de performance : priorité à la clarté du code

## 👨‍💻 Auteur
**[Ton prénom/nom]**  
Projet réalisé dans le cadre de la formation en administration système/réseau  
Sprint Toyceptron - Février 2026

## 📖 Ressources
- [Learn X in Y minutes - Python](https://learnxinyminutes.com/python/)
- [TensorFlow Playground](https://playground.tensorflow.org) (pour visualiser)
