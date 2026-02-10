class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self):
        return (self.x**2 + self.y**2)**0.5

p = Point(3, 4)
print(p.distance())  # Doit afficher 5.0


"""
Pourquoi ce test ?
Classe : Tu crées un "modèle" d'objet Point

Constructeur __init__ : Initialise les coordonnées x et y

self : Fait référence à l'instance courante (l'objet créé)

Méthode : distance() calcule la distance à l'origine (théorème de Pythagore)

Instance : p = Point(3, 4) crée un objet concret
"""