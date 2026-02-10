from neuron import Neuron
from activations import identity, relu, heaviside, sigmoid

# Test : Neurone avec activation identité
n = Neuron(weights=[1, 1], bias=0, activation=identity)
result = n.forward([2, 3])
assert result == 5, f"Erreur : attendu 5, obtenu {result}"
print("✅ Test neurone : OK")

# Test : Dimensions Layer
from layer import Layer
layer = Layer(num_neurons=3, num_inputs=2, activation=identity)
outputs = layer.forward([1.0, 2.0])
assert len(outputs) == 3, f"Erreur : attendu 3 sorties, obtenu {len(outputs)}"
print("✅ Test layer : OK")

# Test : Propagation Network
from network import Network
net = Network([2, 3, 1], [identity, identity])
final = net.forward([1.0, 1.0])
assert len(final) == 1, f"Erreur : attendu 1 sortie, obtenu {len(final)}"
print("✅ Test network : OK")

print("\n🎉 Tous les tests passent !")
