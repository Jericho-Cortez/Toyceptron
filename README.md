# Toyceptron - Perceptron Multi-Couches

## 📝 Description

Implémentation **from scratch** d'un réseau de neurones multi-couches (perceptron) en **Python pur**, sans aucune bibliothèque externe (numpy, torch, sklearn, etc.).

L'objectif pédagogique est de **comprendre en profondeur** le fonctionnement interne d'un réseau de neurones en codant chaque composant à la main : neurones, couches, forward pass, fonctions d'activation.

---

## ✨ Fonctionnalités

- ✅ Construction d'un neurone avec poids, biais et fonction d'activation
- ✅ Création de couches (layers) avec initialisation aléatoire des poids
- ✅ Architecture réseau multi-couches personnalisable
- ✅ 4 fonctions d'activation : identité, Heaviside, sigmoïde, ReLU
- ✅ Forward pass complète à travers le réseau
- ✅ Méthode `summary()` pour afficher l'architecture (optionnel)

---

## 🏗️ Architecture du projet

Toyceptron/
├── neuron.py # Classe Neuron (calcul forward d'un neurone)
├── layer.py # Classe Layer (couche de neurones)
├── network.py # Classe Network (réseau multi-couches)
├── activations.py # Fonctions d'activation
├── main.py # Script de démonstration
└── README.md # Ce fichier


---

## 🚀 Utilisation

### Lancer le projet

```bash
python main.py
```
#Exemple basique
from network import Network
from activations import relu, sigmoid

# Créer un réseau : 2 inputs → 3 hidden → 1 output
net = Network(layer_sizes=, activations=[relu, sigmoid])[1]

# Forward pass
result = net.forward([1.0, 2.0])
print(f"Sortie : {result}")

🧠 Concepts clés
Neurone artificiel
Un neurone effectue 3 opérations :

Produit scalaire : 
z
=
w
1
x
1
+
w
2
x
2
+
.
.
.
+
w
n
x
n
z=w 
1
 x 
1
 +w 
2
 x 
2
 +...+w 
n
 x 
n
 

Ajout du biais : 
z
=
z
+
b
z=z+b

Activation : 
sortie
=
f
(
z
)
sortie=f(z)

Couche (Layer)
Collection de neurones recevant les mêmes inputs et produisant une liste de sorties.

Exemple : une couche de 3 neurones avec 2 inputs produit 3 sorties.

Réseau (Network)
Empilement de couches où les sorties d'une couche deviennent les inputs de la suivante.

Schéma :
