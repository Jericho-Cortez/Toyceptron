
## Qu'est-ce qu'un Perceptron ?

Le perceptron est le **bloc élémentaire** d'un réseau de neurones. C'est un modèle mathématique ultra-simple qui prend plusieurs entrées, les combine avec des poids, et produit une sortie.

**Analogie** : Imagine un neurone comme un **décideur** qui reçoit plusieurs signaux (tes inputs), leur attribue une importance différente (les poids), fait un total, ajoute un biais personnel, puis décide quoi répondre via une fonction d'activation.

***

##  Les 3 opérations fondamentales d'un neurone

Voici **TOUTE la magie** d'un neurone en 3 étapes  :

### 1. **Produit scalaire** (weighted sum)

$$
z = w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n
$$

Tu multiplies chaque entrée $x_i$ par son poids $w_i$, puis tu additionnes tout.

### 2. **Ajout du biais** (bias)

$$
z = z + b
$$

Le biais $b$ est une **constante** que tu ajoutes au résultat. Ça permet au neurone de décaler sa décision, comme un seuil d'activation personnalisé.

### 3. **Fonction d'activation**

$$
sortie = f(z)
$$

La fonction $f$ transforme le résultat $z$ en sortie finale. Exemples : ReLU, sigmoïde, Heaviside, identité.

***

## Exemple concret (calcul à la main)

Prenons des valeurs numériques  :

**Données** :

- Inputs : $x_1 = 1$, $x_2 = 2$
- Poids : $w_1 = 0.5$, $w_2 = -0.3$
- Biais : $b = 0.1$
- Activation : **ReLU** (max(0, z))

**Calcul** :

1. Produit scalaire :

$$
z = (1 \times 0.5) + (2 \times -0.3) = 0.5 - 0.6 = -0.1
$$
2. Ajout du biais :

$$
z = -0.1 + 0.1 = 0.0
$$
3. Activation ReLU :

$$
sortie = \text{max}(0, 0.0) = 0.0
$$

**Résultat final** : 0.0

***

## Visualisation interactive sur TensorFlow Playground

Va sur **https://playground.tensorflow.org**  :

1. **Clique sur "Play"** et regarde le réseau s'entraîner en temps réel
2. **Observe** :
    - Les **lignes** entre neurones = les poids (bleu = positif, orange = négatif)
    - Les **cercles** = les neurones
    - Le **fond coloré** = ce que le réseau prédit pour chaque zone
3. **Expérimente** : Change le nombre de neurones, de couches, les fonctions d'activation → tu verras directement l'impact sur l'apprentissage

C'est **l'outil parfait** pour comprendre visuellement ce que tu vas coder.

***

## Validation de l'étape

Tu dois être capable d'expliquer avec tes propres mots  :

- **Produit scalaire** : Combiner les inputs avec les poids
- **Biais** : Décalage ajouté à la somme
- **Activation** : Transformation finale du résultat
