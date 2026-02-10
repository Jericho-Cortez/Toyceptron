# Sprint Toyceptron - JOUR 1 démarré

## Étape 1.1 : Test POO avec la classe Point

### Objectif

Te remettre dans le bain de la programmation orientée objet (POO) en Python : classes, `__init__`, `self`, méthodes.

### Ce que tu dois faire maintenant

**1. Rappel rapide des bases**

Jette un œil à [learnxinyminutes.com/python/](https://learnxinyminutes.com/python/) et survole :

- Les **listes** (`[]`, `.append()`, boucles `for`)
- Les **classes** (mot-clé `class`, constructeur `__init__`, `self`)
- Les **méthodes** (fonctions à l'intérieur d'une classe)

**2. Code le test**

Crée un fichier `test_poo.py` et code exactement ceci  :

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

C'est **exactement** ce que tu feras avec `Neuron`, `Layer` et `Network` : créer des classes avec des attributs (poids, biais) et des méthodes (`forward()`).





