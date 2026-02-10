# Toyceptron - Perceptron Multi-Couches

## Description

Réseau de neurones codé **from scratch** en Python pur (sans numpy/torch).  
Objectif : comprendre comment fonctionne un perceptron en le codant à la main.

---

## Structure

- neuron.py → Classe Neuron
- layer.py → Classe Layer
- network.py → Classe Network
- activations.py → Fonctions d'activation
- main.py → Démonstration

---

## Utilisation

```bash
python main.py
```

## Exemple rapide
from network import Network
from activations import relu, sigmoid

net = Network(layer_sizes=, activations=[relu, sigmoid])[1]
result = net.forward([1.0, 2.0])
print(result)

---

## Fonctionnalités
- Neurones avec poids, biais et activation

- Couches de neurones avec init aléatoire

- Réseau multi-couches personnalisable

- Activations : identité, Heaviside, sigmoïde, ReLU

- Forward pass complète

---

## Exemple : Porte AND

```python
from neuron import Neuron
from activations import heaviside

n = Neuron(weights=, bias=-1.5, activation=heaviside)[1]
print(n.forward())  # → 0
print(n.forward())  # → 1[1]
```

---

## Contraintes
- Python pur uniquement (pas de librairies)

- Pas d'entraînement (poids aléatoires fixes)

- Code pédagogique, pas optimisé