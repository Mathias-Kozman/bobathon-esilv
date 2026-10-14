# Workbook — Bobathon ESILV

**Projet :** prédiction du vrai score OFF MDS-UPDRS dans la maladie de Parkinson  
**Hub :** `bobathon-esilv`  
**Toutes les valeurs chiffrées sont recalculées localement** depuis `data/` avec les mêmes
splits GroupKFold(5, groupé par `patient_id`) que l'ensemble du projet.

---

## Day 1 — EDA

### Q1 — Taille du jeu de données

**Énoncé :** Combien de visites et de patients distincts en train et test ? Nombre de visites
par patient (typique, max) ?

**Réponse :**

| Split | Visites | Patients |
|-------|--------:|--------:|
| Train | 44 590  | 5 576   |
| Test  | 11 013  | 1 395   |

Visites par patient (train) : médiane = **7**, moyenne = **8.0**, max = **12**.  
Visites par patient (test) : médiane = **7**, moyenne = **7.9**, max = **12**.  
Les patients train et test sont **disjoints** (`patient_id` holdout, comme le split Kaggle).

---

### Q2 — Valeurs manquantes et couverture on/off

**Énoncé :** Quelles sont les trois colonnes les plus manquantes ? Quelle fraction de visites
n'a pas de score `off` ? Pas de score `on` ? Les deux à la fois ?

**Réponse :**

| Rang | Colonne                 | % manquant |
|------|-------------------------|----------:|
| 1    | `time_since_intake_off` | **78.8 %** |
| 2    | `time_since_intake_on`  | **46.4 %** |
| 3    | `off`                   | **42.4 %** |
| 4    | `ledd`                  | 36.6 %     |
| 5    | `gene`                  | 32.4 %     |

- Visites sans score `off` : **42.4 %**  
- Visites sans score `on` : **29.6 %**  
- Visites sans l'un et l'autre (les deux manquants) : **0.0 %** — chaque visite a au
  moins un score clinique.

---

### Q3 — Distribution de la variable cible (`target`)

**Énoncé :** Quelle est la plage observée par rapport à 0–132 ? La distribution est-elle asymétrique ?

**Réponse :**

- Plage observée : **0.0 – 109.5** (loin du plafond théorique de 132 du MDS-UPDRS).
- Moyenne : **37.5**, médiane : **37.3**, écart-type : **16.5**.
- Asymétrie (skewness) : **0.05** — distribution quasi-symétrique, légèrement à droite.
- La distribution est en forme de cloche, ce qui est favorable pour un modèle RMSE sans
  transformation de la cible.

---

### Q4 — Corrélations avec `target` ; `off` et `on` sont-ils des copies ?

**Énoncé :** Quelles colonnes sont les plus corrélées à `target` ? `off` est-il une copie de
`target` ?

**Réponse (corrélations de Pearson, valeur absolue) :**

| Colonne               | Corrélation avec `target` |
|-----------------------|-------------------------:|
| `off`                 | **0.871**                |
| `on`                  | 0.669                    |
| `age`                 | 0.310                    |
| `ledd`                | 0.298                    |
| `age_at_diagnosis`    | 0.133                    |
| `time_since_intake_off`| 0.008                   |
| `sexM`                | 0.001                    |
| `time_since_intake_on` | 0.000                   |

`off` est fortement corrélé à `target` (r = 0.87) mais **ce n'est pas une copie** :
quand les deux sont présents, `off − target` a une moyenne de **−5.4** et un écart-type
de **8.7** points — l'écart-type représente plus de 50 % de l'écart-type de la cible.
C'est le biais systématique que le challenge demande de corriger (subjectivité du rater,
timing du médicament). `on` est lui aussi différent de `target` (r = 0.67).

---

### Q5 — `ledd` manquant par cohorte

**Énoncé :** Le taux de manquant de `ledd` est-il uniforme entre cohortes ?

**Réponse :**

| Cohorte | % `ledd` manquant |
|---------|-----------------:|
| B       | **42.6 %**        |
| A       | **35.9 %**        |

Le taux diffère selon la cohorte : c'est un manquant **non complètement aléatoire
(il dépend de la cohorte)**. Les données seules ne permettent pas de prouver du MNAR
strictement (qui supposerait que la probabilité de manquant dépend de la valeur manquante
elle-même), mais le fait que les deux cohortes aient des taux différents suffit pour
conclure que le manque n'est pas aléatoire.

Cliniquement, un `ledd` absent peut indiquer :
- un patient en **stade précoce**, pas encore sous lévodopa ;
- ou une **cohorte** qui n'enregistre pas systématiquement la dose.

Le manque n'est pas un défaut à imputer aveuglément : c'est un **signal** qui porte de
l'information sur le profil du patient.

---

### Q6 — Colonnes à exclure du modèle

**Énoncé :** Quelles colonnes ne faut-il pas donner au modèle ?

**Réponse :**

- **`Index`** : identifiant de ligne unique (unique\_ratio = 1.0). Le modèle mémoriserait
  chaque ligne, sans généraliser.
- **`patient_id`** : identifiant de patient. Le modèle mémoriserait les patients du train,
  alors que Kaggle évalue sur des patients inconnus.
- **`target`** : c'est la variable à prédire — la donner au modèle serait de la fuite pure.

`off` et `on` **ne sont pas de la fuite** : ils sont disponibles dans `X_test` et leur
présence est prévue par le protocole clinique (visits ON-only ou OFF-only). Le modèle
peut légitimement s'appuyer sur leur valeur — et sur leur absence, qui est également un
signal.

---

## Day 1 — Dummy

### Q7 — RMSE dummy et écart-type de `target`

**Énoncé :** Quel est le RMSE du Dummy ? Pourquoi est-il proche de l'écart-type de `target` ?

**Réponse :**

- RMSE Dummy (split aléatoire 80/20) : **16.48** (recalculé localement)
- Écart-type de `target` : **16.50**

Ces deux valeurs sont quasi identiques car le Dummy prédit toujours la **moyenne du
train**. Son erreur sur un holdout est donc :

$$\text{RMSE}_{\text{dummy}} = \sqrt{\frac{1}{n}\sum(y_i - \bar{y}_{\text{train}})^2} \approx \sigma_y$$

Les légères différences viennent de l'écart entre la moyenne train et la moyenne holdout.

---

### Q8 — Graphe prédit vs réel

**Énoncé :** Quelle forme prend le graphe prédit vs réel pour le Dummy ?

**Réponse :**  
Le Dummy produit une **ligne horizontale** : toutes les prédictions valent la même
constante (≈ 37.5). La diagonale `y = x` serait le graphe d'un modèle parfait.
L'épaisseur du nuage autour de cette ligne horizontale représente l'erreur résiduelle.

---

### Q9 — Seuil d'amélioration réelle

**Énoncé :** De combien doit-on battre le Dummy pour que l'amélioration soit réelle ?

**Réponse (CV groupé 5 folds) :**

- Dummy CV moyen : **16.50**, écart-type inter-folds : **0.31**
- RMSE folds : [16.59, 16.88, 15.95, 16.47, 16.61]

Une amélioration est considérée réelle si elle dépasse nettement **l'écart-type entre
folds** (~0.31) et reste **cohérente sur tous les folds** (pas seulement en moyenne).
En pratique, on cherche un gain de plusieurs points, pas quelques dixièmes.

---

## Day 2 — Ridge

### Q10 — Gain de Ridge vs Dummy

**Énoncé :** De quel pourcentage Ridge réduit-il le RMSE par rapport au Dummy ? Est-ce un
grand gain ?

**Réponse (GroupKFold 5 folds) :**

- Dummy CV moyen : **16.50**
- Ridge CV moyen (α = 1.0) : **10.60**, folds : [10.66, 11.02, 10.45, 10.32, 10.53]
- Gain : **(16.50 − 10.60) / 16.50 = 35.8 %**

C'est un gain **significatif** : Ridge enlève plus d'un tiers de l'erreur en exploitant
les 8 colonnes numériques — surtout `off`, `on` et `age`.

---

### Q11 — Coefficients Ridge : signe et amplitude

**Énoncé :** Quel coefficient a le plus grand poids absolu ? Les signes de `off` et `on`
sont-ils attendus ?

**Réponse (Ridge α = 1.0, ajusté sur tout le train) :**

| Variable               | Coefficient |
|------------------------|------------:|
| `time_since_intake_on` |  **+1.704** |
| `age`                  |   +1.073    |
| `age_at_diagnosis`     |   −0.958    |
| `off`                  |   +0.590    |
| `on`                   |   +0.478    |
| `sexM`                 |   −0.200    |
| `time_since_intake_off`|   −0.155    |
| `ledd`                 |   −0.000    |

Le coefficient de **plus grande valeur absolue** est `time_since_intake_on` (+1.70).

Les signes de `off` et `on` sont **positifs**, comme attendu : plus le score OFF/ON
mesuré est élevé (symptômes sévères), plus le vrai OFF est élevé.

---

### Q12 — Sans standardisation : comparaison des coefficients impossible

**Énoncé :** Pourquoi les coefficients bruts ne permettent pas de comparer l'importance des variables ?

**Réponse :**  
Un coefficient Ridge représente la variation de prédiction pour une augmentation
d'**une unité** de la variable. `age` est en années (~40–80), `ledd` est en mg/jour
(~100–1000) : leurs échelles sont incomparables. Un coefficient de 1.07 sur `age` n'est
pas comparable à −0.0005 sur `ledd`. De plus, la pénalité Ridge pénalise davantage les
coefficients des variables à grande échelle. Pour comparer les importances, il faudrait
standardiser les variables (soustraction de la moyenne, division par l'écart-type) avant
d'ajuster Ridge.

---

### Q13 — Balayage d'alpha

**Énoncé :** Tester α ∈ {0.1, 1, 10, 100}. La plage de RMSE change-t-elle ? Les coefficients
bougent-ils ?

**Réponse :**

| α    | RMSE CV moyen |
|------|-------------:|
| 0.1  | **10.5965**   |
| 1.0  | **10.5965**   |
| 10.0 | **10.5965**   |
| 100.0| **10.5965**   |

Le RMSE est **identique** à 4 décimales pour tous les alpha testés.
Les coefficients changent à peine (exemple : `off` passe de 0.5897 à 0.5897 entre
α = 0.1 et α = 100 ; `time_since_intake_on` de 1.7042 à 1.6990).

**Explication :** avec seulement **8 variables** et **44 590 observations**, le système
est très sur-déterminé. Ridge ne peut réduire que la variance liée à la multi-colinéarité
entre variables (ici faible). L'alpha change quasiment rien car il n'y a presque pas de
variance à réguler.

---

## Day 2 — CV groupé

### Q14 — Split aléatoire vs GroupKFold : lequel change le plus ?

**Énoncé :** Comparer les RMSE en split aléatoire et en CV groupé pour Dummy et Ridge.

**Réponse :**

| Modèle  | Split aléatoire (80/20) | GroupKFold CV moyen | Différence |
|---------|------------------------:|--------------------:|----------:|
| Dummy   | 16.48                   | 16.50               | +0.02      |
| Ridge   | 10.49                   | 10.60               | +0.11      |

Le **Dummy** ne bouge presque pas (il prédit toujours la même constante, sans mémoire).
**Ridge** monte légèrement quand on groupe par patient : le split aléatoire laissait des
visites du même patient des deux côtés de la coupure, ce qui avantageait légèrement le
modèle. Le CV groupé est plus honnête. Avec seulement 8 variables et un modèle linéaire,
la fuite est faible.

---

### Q15 — Quel estimé croire ?

**Réponse :**

Le **GroupKFold à 5 folds, groupé par `patient_id`**, est notre référence pour comparer les modèles localement, car il évite de répartir les visites d'un même patient entre les ensembles d'entraînement et de validation.

Notre évaluation finale donne les résultats suivants :

- Score public Kaggle : **3,873**
- Score privé Kaggle : **3,825**
- RMSE final sur Skore Hub : **4,23**

Les scores public et privé proviennent de l'évaluation Kaggle sur des données distinctes de notre validation locale. Le RMSE Skore Hub repose sur son propre protocole d'évaluation : il ne doit donc pas être comparé directement aux scores Kaggle.

Le modèle finalement retenu est l'expérience `17_stacked_improvements`.
---

## Day 2 — HistGradientBoosting (HGBR)

### Q16 — HGBR bat-il Ridge sur chaque fold ?

**Énoncé :** HGBR bat-il Ridge sur chaque fold ou seulement en moyenne ?

**Réponse :**

| Fold | Ridge  | HGBR   | HGBR < Ridge ? |
|------|-------:|-------:|:-------------- |
| 0    | 10.663 | 7.831  | ✓               |
| 1    | 11.018 | 7.931  | ✓               |
| 2    | 10.446 | 7.581  | ✓               |
| 3    | 10.320 | 7.598  | ✓               |
| 4    | 10.535 | 7.818  | ✓               |
| **Moy.** | **10.596** | **7.752** | ✓ sur **5/5 folds** |

HGBR gagne sur **chaque** fold : le gain est robuste, pas un artefact de moyenne.
Gain moyen : **(10.60 − 7.75) / 10.60 = 26.8 %**.

---

### Q17 — Temps de calcul

**Énoncé :** Combien de fois HGBR est-il plus lent que Ridge ?

**Réponse (mesuré sur le fold 0, train = 35 672 lignes, val = 8 918 lignes) :**

| Modèle | Fit     | Predict  | Ratio fit |
|--------|--------:|---------:|----------:|
| Ridge  | 0.24 s  | 0.025 s  | ×1        |
| HGBR   | 11.8 s  | 0.168 s  | **×50**   |

HGBR est environ **50× plus lent** en fit (les 100 arbres sur histogrammes sont
coûteux), et **7× plus lent** en predict. Le gain de −2.8 points de RMSE justifie
ce surcoût en production par lots ; ce serait à reévaluer en contexte temps-réel.

---

### Q18 — Importance par permutation HGBR (dernier fold)

**Énoncé :** Quelles sont les features les plus importantes pour HGBR ? Pourquoi le classement
diffère-t-il des coefficients Ridge ?

**Réponse (permutation importance, dernier fold, RMSE decrease) :**

| Feature                 | Importance (Δ RMSE) |
|-------------------------|--------------------:|
| `off`                   | **7.33**            |
| `on`                    | 5.11                |
| `age`                   | 2.26                |
| `age_at_diagnosis`      | 1.84                |
| `time_since_intake_on`  | 0.32                |
| `ledd`                  | 0.05                |
| `sexM`                  | 0.005               |
| `time_since_intake_off` | 0.003               |

**Classement différent de Ridge :**
Ridge classe `time_since_intake_on` premier car son coefficient brut de 1.70 est élevé
en termes absolus — mais c'est un **artefact d'échelle** : `time_since_intake_on` varie
entre 0 et 6 h environ, tandis que `ledd` varie entre ~100 et ~1 000 mg/j et `age` entre
~35 et ~80 ans. Des variables à échelles très différentes produisent des coefficients bruts
non comparables. L'importance par permutation ne dépend pas de l'échelle : elle mesure la
perte de RMSE quand on mélange une colonne sur un jeu de validation.
HGBR utilise aussi les **valeurs manquantes comme signal** (ex. `off` absent = visite
ON-only), ce que Ridge ne peut pas faire sans imputation. Il peut également exploiter des
interactions non linéaires (un arbre peut conditionner sur `age` puis sur `off`), ce qui
pourrait contribuer au gain — cette interaction n'a pas été testée explicitement.
C'est pourquoi `off` domine chez HGBR alors que `time_since_intake_on` domine chez Ridge.

---

## Day 2 — tabular_pipeline (skrub)

### Q19 — RMSE avec gene/cohort vs HGBR numérique seul

**Énoncé :** Ajouter `gene`, `cohort`, `rater_id` via `tabular_pipeline` améliore-t-il le RMSE
au-delà de l'écart inter-folds ?

**Réponse :**

| Fold | HGBR num. | skrub tabular | Δ (HGBR − tabular) |
|------|----------:|--------------:|--------------------:|
| 0    | 7.8308    | 7.5543        | **+0.28**           |
| 1    | 7.9314    | 7.6762        | **+0.26**           |
| 2    | 7.5813    | 7.3348        | **+0.25**           |
| 3    | 7.5980    | 7.4186        | **+0.18**           |
| 4    | 7.8184    | 7.6236        | **+0.19**           |
| Moy. | 7.7520    | **7.5145**    | **+0.238**          |

skrub tabular gagne sur **5/5 folds**, avec un gain par fold de **+0.18 à +0.28**.
L'amélioration est cohérente : ce n'est pas un artefact d'un seul fold.

---

### Q20 — Encodeurs choisis par TableVectorizer

**Énoncé :** Quel encodeur `TableVectorizer` choisit pour chaque colonne texte ? Concordance
avec la cardinalité ?

**Réponse (pipeline ajusté sur tout le train) :**

| Colonne    | Cardinalité | Encodeur         | Règle skrub (seuil = 40)          |
|-----------|------------:|:----------------:|:----------------------------------|
| `cohort`  | **2**        | `ToCategorical`  | < 40 → encodage catégoriel natif  |
| `gene`    | **4**        | `ToCategorical`  | < 40 → encodage catégoriel natif  |
| `rater_id`| **50**       | `StringEncoder`  | > 40 → encodage dense             |

`ToCategorical` **ne fait pas du one-hot** : il convertit la colonne en type
`category` pandas et la passe telle quelle à HGBR, qui supporte nativement ce
type (splits discrets sur les catégories). Le seuil de cardinalité dans skrub est
**40** : en dessous → `ToCategorical` (cardinalité faible, HGBR gère les catégories
directement) ; au-dessus → `StringEncoder` (cardinalité élevée, encodage dense dans
un espace numérique compact). One-hot serait utilisé dans un pipeline avec estimateur
non catégoriel (ex. Ridge).

---

## Day 2 — DataOps (skrub)

### Q21 — Score DataOps vs étape 5

**Énoncé :** Le score DataOps (exp 06) est-il identique à l'étape 5 ?  
Quelles colonnes contient `X_op` ?

**Réponse :**  
Dans `06_dataops.py`, `X_op` est construit via :

```python
features = data[FEATURE_COLS]  # ['sexM', 'age_at_diagnosis', ..., 'off']
X = features.skb.mark_as_X(...)
```

`X_op` contient **uniquement les 8 colonnes numériques** de `FEATURE_COLS` —
`Index` et `patient_id` en sont **absents**. Il n'y a pas de fuite identifiants.

Comparaison :
- **Exp 04 (skore.evaluate, HGBR numérique)** : RMSE = **7.7520 ± 0.138**
- **Exp 06 (DataOps cross_validate, même HGBR)** : RMSE = **7.7520 ± 0.155** (source : code exp 06)

La moyenne est identique (7.7520) ; l'écart-type diffère légèrement : **0.138 vs 0.155**.
Cela vient du choix de `ddof` : `np.std()` utilise par défaut `ddof=0`
(diviseur = n = 5) → 0.138 ; `ddof=1` (diviseur = n−1 = 4) → 0.155. Les deux sont
calculés sur les mêmes 5 RMSE de fold ; seule la formule de la variance diffère.

Les prédictions sont identiques : le graphe DataOps produit exactement le même modèle sur
les mêmes features. La valeur ajoutée de DataOps est organisationnelle : les `groups`
et le `GroupKFold` sont attachés au graphe, éliminant le risque d'oublier les groupes
lors d'un refactor.

> **Note :** l'exemple du lab (DAY2.md) montre `X_op = data.drop("target")` — dans ce
> cas `Index` et `patient_id` resteraient dans `X_op`. Notre code utilise `data[FEATURE_COLS]`
> à la place, ce qui est plus propre. Aucune correction n'est nécessaire.

---

### Q22 — Index/patient_id dans les features : risque ?

**Énoncé :** Si `Index` ou `patient_id` étaient dans les features, quel serait le problème ?

**Réponse :**  
Si un identifiant entrait dans le modèle, celui-ci apprendrait à associer un ID à un
score cible — ce qui ne généralise pas à de **nouveaux patients** (Kaggle évalue sur
des patients inconnus). Cela se détecterait par une importance par permutation élevée
pour l'identifiant. Dans notre code, ce risque est éliminé dès la construction de `X_op`.

**Correction si nécessaire (à ne pas appliquer sans accord) :** remplacer
`data.drop("target")` par `data.drop(columns=["Index", "patient_id", "target"])`, en
extrayant `groups = data["patient_id"]` avant le drop.

---

## Day 2 — Wrap-up

### Q23 — Quel changement a donné le plus grand gain ?

**Énoncé :** Classer les changements par gain de RMSE.

**Réponse (RMSE en CV groupé à 5 folds, lorsque applicable) :**

Étape   Changement                                                              RMSE CV / Hub    Gain vs étape précédente
0       Dummy (référence)                                                       16.50            ---
1       + Ridge (8 variables numériques, NaN imputés par la médiane)            10.60            −5.90 (−35.8 %)
2       + HGBR (arbres, NaN conservés)                                          7.75             −2.85 (−26.8 %)
3       + tabular_pipeline (cohort, gene, rater_id)                             7.51             −0.24 (−3.1 %)
4       Meilleur modèle Skore Hub (LightGBM + feature engineering + ensemble)   4.23             −3.28 (−43.7 %)



Classement par gain absolu de RMSE :

Dummy → Ridge (−5.90) : le plus gros gain. Le modèle exploite principalement off, on et age.

tabular_pipeline → meilleur modèle Skore Hub (−3.28) : ajout de features patient (agrégats, voisins temporels, position de visite, indicateurs de valeurs manquantes), encodage du rater, LightGBM et ensemble.

Ridge → HGBR (−2.85) : passage à un modèle à base d'arbres, en conservant les valeurs manquantes.

HGBR → tabular_pipeline (−0.24) : gain plus faible, mais observé sur les 5 folds (voir Q19).

Résultats finaux de la soumission Kaggle — expérience 17_stacked_improvements :

Score public : 3.873

Score privé : 3.825

RMSE final sur Skore Hub : 4.23

Les scores Kaggle correspondent à l'évaluation externe de la soumission finale. Ils ne sont pas directement comparables au RMSE Skore Hub ni aux résultats historiques de validation croisée, car les protocoles d'évaluation diffèrent.

Gain direct nul : le GroupKFold. La validation croisée groupée n'améliore pas directement les prédictions du modèle. Elle fournit une estimation plus fiable de ses performances en empêchant les visites d'un même patient d'apparaître à la fois dans les ensembles d'entraînement et de validation.

Limites d'interprétation :

Les étapes 2 et 4 regroupent plusieurs changements. À l'étape 2, le modèle passe d'une régression linéaire à des arbres et le traitement des valeurs manquantes change également (imputation → gestion native des NaN). À l'étape 4, de nouvelles features, un nouveau modèle et un ensemble sont introduits. Sans expériences d'ablation dédiées, il est impossible d'isoler précisément la contribution de chaque changement.

Le RMSE de 4.23 provient du meilleur rapport Skore Hub. Les scores Kaggle finaux sont de 3.873 en public et 3.825 en privé. Ces valeurs ne doivent pas être comparées comme si elles provenaient du même protocole d'évaluation.

Le RMSE de l'expérience 13 est calculé à l'exécution et n'est pas directement enregistré dans le code. L'expérience 16 mentionne une valeur de 4.12 dans un commentaire, mais celle-ci n'a pas été recalculée indépendamment.

---

### Q24 — Visites les plus mal prédites

**Énoncé :** Quelles visites le meilleur modèle (skrub tabular) prédit-il le plus mal ?

**Réponse (OOF, skrub tabular, top 20 erreurs absolues) :**

- **`off` manquant** dans 11/20 cas (55 %) vs 42.4 % en global — écart de +13 points.
  L'absence de `off` prive le modèle de son meilleur prédicteur.
- **`target` élevé** : moyenne 77.0 dans le top 20, vs 37.5 globalement. La `target`
  minimale du top 20 est 45.1. Le modèle **sous-estime les patients sévères** : prédit
  vers la moyenne (≈ 37), alors que leur vraie valeur est 70–110.

En synthèse : les pires prédictions concernent principalement des **patients sévères
(target ≥ 45, moyenne 77) que le modèle ramène vers la moyenne globale**, aggravé par
l'absence de `off` dans un peu plus de la moitié des cas.

---

### Q25 — Problèmes non traités et features proposées

**Énoncé :** Quels problèmes de modélisation n'ont pas été traités au niveau du workbook Day 2 ?
Quelles features construire ?

**Réponse (d'après CONTEXT.md) :**

**Problèmes non traités dans le workbook Day 2 :**
1. **Progression temporelle** : chaque visite est prédite indépendamment, sans tenir
   compte de l'évolution du patient dans le temps.
2. **Timing du médicament** : `time_since_intake_off` est présent à 21 % seulement ;
   modéliser la pharmacodynamique de la lévodopa pourrait débiaiser le score.

**Features à construire :**

| Feature                                      | Formule / logique                                     |
|----------------------------------------------|-------------------------------------------------------|
| `years_since_diagnosis`                      | `age − age_at_diagnosis`                              |
| Rang de visite par patient                   | Rang de la visite dans la séquence du patient         |
| Nb de visites par patient                    | `patient_id.map(count)` sur tout le jeu               |
| Agrégats patient de on/off                   | Moyenne/médiane de `on` et `off` par patient (valides sur test aussi) |
| `levo_decay`                                 | `exp(−time_since_intake_off / T_HALF)` ; NaN conservé + indicateur `levo_decay_na=1` si manquant (jamais remplacer par 0) |
| Indicateurs de manquant                      | `off_is_nan`, `on_is_nan`, `ledd_is_nan` (0/1)        |
| Delta depuis visite précédente               | Différence d'âge entre deux visites consécutives      |

Ces features sont toutes calculables sur `X_test` (pas de fuite) car elles n'utilisent
que des informations disponibles à chaque visite.

**Note :** les expériences 09–16 ont déjà implémenté la majorité de ces pistes.
En particulier, **l'exp 13** ajoute les blocs suivants (source : `build_features()` dans
`experiments/13_best_model.py`) :

| Bloc | Features ajoutées dans exp 13 |
|------|-------------------------------|
| B    | Agrégats patient : mean/median/std/min/max/count/slope par variable (`off_mean_pat`, `on_slope_pat`, etc.) |
| C    | Features voisines temporelles : `off_prev`, `off_next`, `off_age_gap_prev/_next` (triées par `age`) |
| D    | Position de visite : `visit_rank`, `time_since_first_visit`, `time_to_last_visit`, `rel_visit_pos` |
| E    | Indicateurs de manquant : `off_na`, `on_na`, `ledd_na`, `gene_na`, `time_since_intake_on_na`, `time_since_intake_off_na` |
| F    | Fréquence de visite du rater : `rater_visit_count` |
| G    | Intercept aléatoire patient (MixedLM off ~ age + tsd) : `lme_intercept_pat`, `lme_pred_off` |
| H    | Encodage OOF du rater (mean(target−off) par rater, calculé en-fold) : `rater_oof_bias` |
| I    | Dérivées : `time_since_diagnosis`, `on_minus_off`, `on_div_off` + interactions (diff/ratio/prod) |
