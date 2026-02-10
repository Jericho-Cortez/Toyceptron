from math import exp

from network import Network
from layer import Layer
from neuron import Neuron
from activations import identity, relu, heaviside, sigmoid

def act_sigmoid(x):
    return 1 / (1 + exp(-x))


# --- Input ---
x = [2, 3, 1]
print("Input:", x)

# --- Test neurone individuel ---
print("\n--- Test Neuron ---")
# les neurones ont 3 biais pour correspondre à la taille de notre input
n1 = Neuron(weights=[0.2, -0.1, 0.4], bias=0.0)
n2 = Neuron(weights=[-0.4, 0.3, 0.1], bias=0.1)

out_n1 = n1.forward(x)
out_n2 = n2.forward(x)
print("Neurone h1 (brut):", out_n1)  # 1.6
print("Neurone h2 (brut):", out_n2)  # 0.7
print("Neurone h1 (activé):", act_sigmoid(out_n1))  # 0.8320183851339245
print("Neurone h2 (activé):", act_sigmoid(out_n2))  # 0.6681877721681662


# --- Test couche cachée ---
print("\n--- Test Layer ---")
# Mêmes valeurs, mais on ajoute les deux neurones en une seule fois dans la couche
# On devrait avoir des résultats identiques
layer = Layer(
    weights_list=[
        [0.2, -0.1, 0.4],
        [-0.4, 0.3, 0.1],
    ],
    biases_list=[0.0, 0.1],
)

raw = layer.forward(x)
activated = [act_sigmoid(o) for o in raw]
print("Couche (valeurs brutes):", raw)  # [1.6, 0.7]
print(
    "Couche (valeurs activées):", activated
)  # [0.8320183851339245, 0.6681877721681662]

# --- Test du réseau entier ---
print("\n--- Test Network ---")
net = Network(input_size=3, activation=act_sigmoid)
# 1. On commence par une couche "cachée"
net.add(
    weights=[
        [0.2, -0.1, 0.4],
        [-0.4, 0.3, 0.1],
    ],
    biases=[0.0, 0.1],
)

# 2. La deuxième couche est cachée aussi
net.add(
    weights=[
        [0.5, -0.2],  # la taille correspond au nombre de neurones couche 1
        [-0.3, 0.4],
        [0.1, 0.2],
    ],
    biases=[0.0, 0.1, -0.1],
)

# 3. Couche de sortie
net.add(
    weights=[
        [0.3, -0.1, 0.2],  # taille = nombre de neurones couche 2
        [-0.5, 0.4, 0.1],
    ],
    biases=[-0.1, 0.0],
)

# Feedforward complet (via Network)
y = net.feedforward(x)
print("\nSorties activées :", y)  # [0.5309442148001715, 0.494901997674804]

# Feedforward couche par couche (mêmes résultats en théorie)
inputs = x
for i, layer in enumerate(net.layers):
    raw = layer.forward(inputs)
    activated = [act_sigmoid(o) for o in raw]
    print(f"\nCouche {i + 1} (valeurs brutes):", raw)
    print(f"Couche {i + 1} (valeurs activées):", activated)
    inputs = activated