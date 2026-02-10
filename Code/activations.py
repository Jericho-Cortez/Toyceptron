def act_identity(x):
    """Fonction identité : f(x) = x"""
    return x


def act_threshold(x):
    """Fonction de seuil (Heaviside) : f(x) = 1 si x ≥ 0, sinon 0"""
    return 1.0 if x >= 0 else 0.0


def act_relu(x):
    """Fonction ReLU : f(x) = max(0, x)"""
    return max(0.0, x)


# act_sigmoid est déjà défini dans le main.py, pas besoin de le remettre ici
