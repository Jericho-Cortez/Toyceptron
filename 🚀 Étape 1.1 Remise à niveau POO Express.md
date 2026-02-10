<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Sprint Toyceptron - JOUR 1 démarré

Étape 1.1 : test POO avec classe Point
Go ! 🚀

Parfait ! 🚀 Le Sprint Toyceptron est lancé ! Attaquons l'**Étape 1.1** : ton échauffement POO avant de plonger dans les neurones.[^1_1]

## Étape 1.1 : Test POO avec la classe Point

### Objectif

Te remettre dans le bain de la programmation orientée objet (POO) en Python : classes, `__init__`, `self`, méthodes.[^1_1]

### Ce que tu dois faire maintenant

**1. Rappel rapide des bases**

Jette un œil à [learnxinyminutes.com/python/](https://learnxinyminutes.com/python/) et survole :

- Les **listes** (`[]`, `.append()`, boucles `for`)
- Les **classes** (mot-clé `class`, constructeur `__init__`, `self`)
- Les **méthodes** (fonctions à l'intérieur d'une classe)

**2. Code le test**

Crée un fichier `test_poo.py` et code exactement ceci  :[^1_1]

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self):
        return (self.x**2 + self.y**2)**0.5

p = Point(3, 4)
print(p.distance())  # Doit afficher 5.0
```

**3. Exécute**

```bash
python test_poo.py
```

**4. Résultat attendu**

Tu dois voir s'afficher :

```
5.0
```


### Pourquoi ce test ?

- **Classe** : Tu crées un "modèle" d'objet Point
- **Constructeur `__init__`** : Initialise les coordonnées x et y
- **`self`** : Fait référence à l'instance courante (l'objet créé)
- **Méthode** : `distance()` calcule la distance à l'origine (théorème de Pythagore)
- **Instance** : `p = Point(3, 4)` crée un objet concret

C'est **exactement** ce que tu feras avec `Neuron`, `Layer` et `Network` : créer des classes avec des attributs (poids, biais) et des méthodes (`forward()`).[^1_1]

### ✅ Validation

Si tu vois `5.0` s'afficher → **Bravo, tu es prêt pour l'étape 1.2 !**

Si erreur → partage-moi le message d'erreur et je t'aide à corriger.

***

**Prêt à passer à l'étape 1.2 (Comprendre le Perceptron) ?** Dis-moi quand c'est bon ! 💪

<div align="center">⁂</div>

[^1_1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

