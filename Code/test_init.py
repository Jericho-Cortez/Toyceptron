import random
from neuron import Neuron
from layer import Layer
from activations import identity

# Pas de seed → poids différents à chaque exécution

print("=== Test unicité des neurones ===")
n1 = Neuron(num_inputs=3)
n2 = Neuron(num_inputs=3)
print(f"Neurone 1: {n1.weights}")
print(f"Neurone 2: {n2.weights}")

if n1.weights == n2.weights:
    print("❌ ERREUR: Les poids sont identiques!")
else:
    print("✅ Les poids sont différents")

print("\n=== Test couche avec 4 neurones ===")
layer = Layer(num_neurons=4, num_inputs=2, activation=identity)
for i, neuron in enumerate(layer.neurons):
    print(f"Neurone {i+1}: poids={neuron.weights}, biais={neuron.bias:.3f}")

print("\n=== Forward pass ===")
result = layer.forward([1.0, 2.0])
print(f"Entrées: [1.0, 2.0]")
print(f"Sorties: {result}")
