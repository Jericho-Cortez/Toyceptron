<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## 🚀 Étape 1.3 : Coder `neuron.py` — C'EST PARTI !

Parfait, on attaque le cœur du projet ! Je vais t'aider à créer ton premier neurone artificiel en Python pur.[^1_1]

## 🧠 Rappel théorique ultra-rapide

Un neurone fait **3 choses** dans l'ordre  :[^1_1]

1. **Produit scalaire** : $z = w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n$
2. **Ajout du biais** : $z = z + b$
3. **Activation** : $\text{sortie} = f(z)$

**Exemple concret** : Si inputs = `[1, 2]`, poids = `[0.5, -0.3]`, biais = `0.1`[^1_1]

- Calcul : $1 \times 0.5 + 2 \times (-0.3) + 0.1 = 0.5 - 0.6 + 0.1 = 0.0$


## 📝 CODE COMPLET DE `neuron.py`

```python
class Neuron:
    """
    Un neurone artificiel qui calcule :
    sortie = activation(somme_ponderee + biais)
    """
    
    def __init__(self, weights, bias):
        """
        Initialise le neurone avec ses paramètres.
        
        Args:
            weights (list): Liste des poids [w1, w2, ..., wn]
            bias (float): Biais du neurone
        """
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        """
        Calcule la sortie du neurone pour des inputs donnés.
        
        Args:
            inputs (list): Valeurs d'entrée [x1, x2, ..., xn]
        
        Returns:
            float: Résultat du calcul (sans activation pour l'instant)
        """
        # Étape 1 : Produit scalaire inputs × weights
        z = 0.0
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        
        # Étape 2 : Ajouter le biais
        z += self.bias
        
        # Étape 3 : Retourner le résultat
        return z


# TEST IMMÉDIAT
if __name__ == "__main__":
    # Test avec l'exemple de la roadmap
    n = Neuron(weights=[0.5, -0.3], bias=0.1)
    result = n.forward([1, 2])
    print(f"Résultat: {result}")  # Doit afficher 0.0
    
    # Test supplémentaire
    n2 = Neuron(weights=[1, 1], bias=0)
    result2 = n2.forward([2, 3])
    print(f"Résultat 2: {result2}")  # Doit afficher 5.0
```


## ✅ EXPLICATION LIGNE PAR LIGNE

### Le constructeur `__init__`

```python
self.weights = weights  # Stocke la liste des poids
self.bias = bias        # Stocke le biais (un seul nombre)
```

Ces deux attributs sont la "mémoire" du neurone.[^1_1]

### La méthode `forward`

```python
z = 0.0  # Initialise l'accumulateur à zéro
for i in range(len(inputs)):
    z += inputs[i] * self.weights[i]  # Multiplie chaque entrée par son poids
```

Cette boucle fait le **produit scalaire** : $x_1 \times w_1 + x_2 \times w_2 + ...$[^1_1]

```python
z += self.bias  # Ajoute le biais au résultat
return z        # Retourne la valeur finale
```


## 🧪 VALIDATION — Lance le test !

Ouvre un terminal et exécute  :[^1_1]

```bash
python neuron.py
```

**Tu DOIS voir** :

```
Résultat: 0.0
Résultat 2: 5.0
```

Si ça marche : **BRAVO, ton neurone fonctionne !** 🎉

## 🎯 PROCHAINE ÉTAPE (Étape 1.4)

Maintenant qu'on a le calcul de base, on va ajouter les **fonctions d'activation** pour rendre le neurone "intelligent"  :[^1_1]

- `identity` (identité)
- `heaviside` (seuil)
- `sigmoid` (sigmoïde)
- `relu` (ReLU)

**Tu veux qu'on attaque directement l'Étape 1.4, ou tu veux d'abord tester ton `neuron.py` et me dire si ça marche ?** 💪

<div align="center">⁂</div>

[^1_1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

