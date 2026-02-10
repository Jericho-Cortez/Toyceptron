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

