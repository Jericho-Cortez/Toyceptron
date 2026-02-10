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
