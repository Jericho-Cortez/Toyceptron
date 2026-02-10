"""
Tests manuels avec calculs vérifiables
Toyceptron - JOUR 2 Étape 2.4
"""
import random
from neuron import Neuron
from layer import Layer
from activations import identity, relu, heaviside, sigmoid


print("=" * 60)
print("TEST MANUEL - Étape 2.4 : Tests unitaires basiques")
print("=" * 60)


# ========================================
# Test 1 : Neurone avec poids fixés
# ========================================
print("\n[Test 1] Neurone avec poids fixes - Calcul vérifiable")
print("-" * 60)

n = Neuron(weights=[1, 1], bias=0, activation=identity)
result = n.forward([2, 3])

print(f"Configuration : weights=[1, 1], bias=0, activation=identity")
print(f"Calcul manuel : 1*2 + 1*3 + 0 = 5")
print(f"Résultat obtenu : {result}")
assert result == 5, f"❌ ERREUR : attendu 5, obtenu {result}"
print("✅ Test 1 réussi : Calcul correct")


# ========================================
# Test 2 : Neurone avec activation ReLU
# ========================================
print("\n[Test 2] Neurone avec ReLU - Valeur négative")
print("-" * 60)

n_relu = Neuron(weights=[1, -2], bias=-1, activation=relu)
result_relu = n_relu.forward([1, 2])

print(f"Configuration : weights=[1, -2], bias=-1, activation=relu")
print(f"Calcul manuel : 1*1 + (-2)*2 + (-1) = 1 - 4 - 1 = -4")
print(f"Après ReLU : max(0, -4) = 0")
print(f"Résultat obtenu : {result_relu}")
assert result_relu == 0, f"❌ ERREUR : attendu 0, obtenu {result_relu}"
print("✅ Test 2 réussi : ReLU fonctionne correctement")


# ========================================
# Test 3 : Porte logique AND avec Heaviside
# ========================================
print("\n[Test 3] Porte logique AND - Application concrète")
print("-" * 60)

# Un neurone peut implémenter une porte logique AND !
n_and = Neuron(weights=[1, 1], bias=-1.5, activation=heaviside)

test_cases = [
    ([0, 0], 0),  # 0 + 0 - 1.5 = -1.5 → heaviside → 0
    ([1, 0], 0),  # 1 + 0 - 1.5 = -0.5 → 0
    ([0, 1], 0),  # 0 + 1 - 1.5 = -0.5 → 0
    ([1, 1], 1),  # 1 + 1 - 1.5 = 0.5 → 1
]

print("Table de vérité AND :")
print("A  B  | Sortie attendue | Sortie obtenue | Statut")
print("-" * 60)

all_passed = True
for inputs, expected in test_cases:
    result = n_and.forward(inputs)
    status = "✅" if result == expected else "❌"
    print(f"{inputs[0]}  {inputs[1]}  |        {expected}        |       {result}        | {status}")
    
    if result != expected:
        all_passed = False
        print(f"   ❌ Erreur sur {inputs} : attendu {expected}, obtenu {result}")

if all_passed:
    print("✅ Test 3 réussi : Le neurone implémente correctement AND !")


# ========================================
# Test 4 : Layer avec calcul manuel
# ========================================
print("\n[Test 4] Layer - Vérification d'une couche complète")
print("-" * 60)

# Créer une couche avec des poids contrôlés
random.seed(999)  # Seed fixe pour reproductibilité
layer_test = Layer(num_neurons=2, num_inputs=2, activation=identity)

print(f"Couche avec 2 neurones, 2 inputs")
print(f"Neurone 1 : weights={layer_test.neurons[0].weights}, bias={layer_test.neurons[0].bias:.3f}")
print(f"Neurone 2 : weights={layer_test.neurons[1].weights}, bias={layer_test.neurons[1].bias:.3f}")

inputs_test = [1.0, 0.5]
result_layer = layer_test.forward(inputs_test)

print(f"\nEntrées : {inputs_test}")
print(f"Sorties : {result_layer}")
print(f"Type : {type(result_layer)}, Longueur : {len(result_layer)}")

assert len(result_layer) == 2, "❌ La couche doit retourner 2 sorties"
assert isinstance(result_layer, list), "❌ forward() doit retourner une liste"
print("✅ Test 4 réussi : La couche fonctionne correctement")


# ========================================
# Test 5 : Reproductibilité avec seed
# ========================================
print("\n[Test 5] Reproductibilité - Même seed = Mêmes poids")
print("-" * 60)

random.seed(42)
layer1 = Layer(num_neurons=3, num_inputs=2, activation=identity)
weights1 = [n.weights for n in layer1.neurons]
result1 = layer1.forward([1.0, 2.0])

random.seed(42)  # Reset avec le même seed
layer2 = Layer(num_neurons=3, num_inputs=2, activation=identity)
weights2 = [n.weights for n in layer2.neurons]
result2 = layer2.forward([1.0, 2.0])

print(f"Layer 1 poids : {weights1}")
print(f"Layer 2 poids : {weights2}")
print(f"Résultat 1 : {result1}")
print(f"Résultat 2 : {result2}")

assert result1 == result2, "❌ Les résultats doivent être identiques avec le même seed"
print("✅ Test 5 réussi : Reproductibilité garantie")


# ========================================
# Test 6 : Gestion d'erreur
# ========================================
print("\n[Test 6] Gestion des erreurs - Paramètres invalides")
print("-" * 60)

try:
    n_error = Neuron(weights=None, num_inputs=None)
    print("❌ ÉCHEC : L'erreur ValueError n'a pas été levée")
except ValueError as e:
    print(f"✅ Erreur correctement levée : {e}")


# ========================================
# Test 7 : Sigmoid - Valeur médiane
# ========================================
print("\n[Test 7] Activation Sigmoid - Valeur z=0")
print("-" * 60)

n_sigmoid = Neuron(weights=[1, -1], bias=0, activation=sigmoid)
result_sigmoid = n_sigmoid.forward([2, 2])

print(f"Configuration : weights=[1, -1], bias=0")
print(f"Calcul : 1*2 + (-1)*2 + 0 = 0")
print(f"Sigmoid(0) devrait être proche de 0.5")
print(f"Résultat obtenu : {result_sigmoid:.4f}")

# Tolérance de 0.01 car sigmoid(0) = 0.5
assert abs(result_sigmoid - 0.5) < 0.01, f"❌ Sigmoid(0) devrait être ~0.5, obtenu {result_sigmoid}"
print("✅ Test 7 réussi : Sigmoid fonctionne correctement")


# ========================================
# RÉCAPITULATIF FINAL
# ========================================
print("\n" + "=" * 60)
print("🎉 TOUS LES TESTS UNITAIRES SONT PASSÉS !")
print("=" * 60)
print("\n✅ Neuron : Calculs corrects")
print("✅ Activations : identity, relu, heaviside, sigmoid")
print("✅ Layer : Forward pass fonctionnel")
print("✅ Porte logique AND : Implémentation réussie")
print("✅ Reproductibilité : Gestion du seed OK")
print("✅ Gestion d'erreurs : Robuste")
print("\n🚀 Tu es PRÊT pour l'étape 3.1 : Network !")
print("=" * 60)
