<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## 🚀 Étape 1.2 : Comprendre le Perceptron (1h) - C'est parti !

Parfait, on démarre ! L'objectif de cette étape est de **saisir la logique mathématique d'un neurone** avant de coder quoi que ce soit. C'est la fondation de tout ton projet Toyceptron.[^1_1]

***

## 🧠 Qu'est-ce qu'un Perceptron ?

Le perceptron est le **bloc élémentaire** d'un réseau de neurones. C'est un modèle mathématique ultra-simple qui prend plusieurs entrées, les combine avec des poids, et produit une sortie.[^1_2][^1_3][^1_4][^1_5]

**Analogie** : Imagine un neurone comme un **décideur** qui reçoit plusieurs signaux (tes inputs), leur attribue une importance différente (les poids), fait un total, ajoute un biais personnel, puis décide quoi répondre via une fonction d'activation.[^1_6]

***

## 🎯 Les 3 opérations fondamentales d'un neurone

Voici **TOUTE la magie** d'un neurone en 3 étapes  :[^1_1]

### 1. **Produit scalaire** (weighted sum)

$$
z = w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n
$$

Tu multiplies chaque entrée $x_i$ par son poids $w_i$, puis tu additionnes tout.[^1_7][^1_3]

### 2. **Ajout du biais** (bias)

$$
z = z + b
$$

Le biais $b$ est une **constante** que tu ajoutes au résultat. Ça permet au neurone de décaler sa décision, comme un seuil d'activation personnalisé.[^1_6][^1_1]

### 3. **Fonction d'activation**

$$
sortie = f(z)
$$

La fonction $f$ transforme le résultat $z$ en sortie finale. Exemples : ReLU, sigmoïde, Heaviside, identité.[^1_8][^1_7][^1_1]

***

## 📐 Exemple concret (calcul à la main)

Prenons des valeurs numériques  :[^1_1]

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

**Résultat final** : 0.0[^1_1]

***

## 🎮 Visualisation interactive sur TensorFlow Playground

Va sur **https://playground.tensorflow.org**  :[^1_9][^1_10]

1. **Clique sur "Play"** et regarde le réseau s'entraîner en temps réel
2. **Observe** :
    - Les **lignes** entre neurones = les poids (bleu = positif, orange = négatif)[^1_9]
    - Les **cercles** = les neurones
    - Le **fond coloré** = ce que le réseau prédit pour chaque zone[^1_9]
3. **Expérimente** : Change le nombre de neurones, de couches, les fonctions d'activation → tu verras directement l'impact sur l'apprentissage[^1_1]

C'est **l'outil parfait** pour comprendre visuellement ce que tu vas coder.[^1_10][^1_1]

***

## ✅ Validation de l'étape

Tu dois être capable d'expliquer avec tes propres mots  :[^1_1]

- **Produit scalaire** : Combiner les inputs avec les poids
- **Biais** : Décalage ajouté à la somme
- **Activation** : Transformation finale du résultat

**Test mental** : Si je te donne des inputs [3, -1], des poids [0.2, 0.4], un biais de -0.1 et une activation ReLU, tu dois pouvoir calculer la sortie à la main.

<details>
<summary>💡 Solution du test mental (clique pour révéler)</summary>

1. Produit scalaire : \(z = 3 \times 0.2 + (-1) \times 0.4 = 0.6 - 0.4 = 0.2\)
2. Biais : \(z = 0.2 - 0.1 = 0.1\)
3. ReLU : \(\text{max}(0, 0.1) = 0.1\)

**Réponse** : 0.1
</details>

***

## 🔜 Prochaine étape

Une fois que tu maîtrises ces 3 concepts, tu es **prêt pour l'étape 1.3** : coder ta classe `Neuron` en Python  ![^1_1]

**Timing** : Prends bien ton heure pour bien comprendre, n'hésite pas à jouer avec le Playground et à refaire des calculs à la main. La compréhension maintenant = du code facile après 💪

Tu as des questions sur les maths ou tu veux qu'on passe directement au code ?
<span style="display:none">[^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20]</span>

<div align="center">⁂</div>

[^1_1]: ROADMAP-TOYCEPTRON-Mode-Sprint-3-4-jours.md

[^1_2]: https://www.geeksforgeeks.org/machine-learning/what-is-perceptron-the-simplest-artificial-neural-network/

[^1_3]: https://blog.damavis.com/en/simple-perceptron-mathematical-definition-and-properties/

[^1_4]: https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/index.html

[^1_5]: https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/

[^1_6]: https://www.picsellia.com/post/activation-functions-and-neural-networks

[^1_7]: https://www.tensorflow.org/guide/core/mlp_core

[^1_8]: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

[^1_9]: https://playground.tensorflow.org

[^1_10]: https://deepwiki.com/tensorflow/playground/2.2-visualizations

[^1_11]: https://blog.tensorflow.org/2019/02/mit-introduction-to-deep-learning.html

[^1_12]: https://www.tensorflow.org/guide/keras/working_with_rnns

[^1_13]: https://www.tensorflow.org/tutorials/distribute/dtensor_ml_tutorial

[^1_14]: https://www.tensorflow.org/guide/autodiff

[^1_15]: https://www.tensorflow.org/tensorboard

[^1_16]: https://js.tensorflow.org/api/1.0.0/

[^1_17]: https://projector.tensorflow.org

[^1_18]: https://fr.wikipedia.org/wiki/Perceptron

[^1_19]: https://www.math.univ-toulouse.fr/~xgendre/ens/m2se/M2SE_TP5_en.pdf

[^1_20]: https://www.reddit.com/r/learnmachinelearning/comments/18can28/using_activation_function_before_taking_dot/

