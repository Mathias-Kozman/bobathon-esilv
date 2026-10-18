# Expérience 17 : ajout de plusieurs améliorations au modèle LightGBM
# Cette expérience reprend les améliorations des expériences 15 et 16, puis ajoute trois nouvelles modifications.
# Les améliorations précédentes :
# - exp15 : ajout de statistiques sur les visites précédentes de chaque
#   patient (off_hist_mean, off_hist_count, off_hist_std, ledd_hist_mean
#   et hist_n_visits).
# - exp16 : utilisation de la calibration isotonic pour corriger les
#   prédictions du modèle à partir des prédictions OOF.
#
# Les trois nouveautés de cette expérience :
#
# A - Entraînement avec 5 seeds
#     - Le modèle est entraîné 5 fois avec des seeds différents (de 0 à 4).
#     - On fait ensuite la moyenne des prédictions pour obtenir un résultat plus stable. Cette méthode est utilisée uniquement pour les prédictions finales, pas pendant la validation croisée.
#
# B - Limitation des prédictions entre 0 et 132
#     Le score MDS-UPDRS Part III est compris entre 0 et 132.
#     On limite donc les prédictions à cet intervalle pour éviter d'obtenir des valeurs impossibles.
#
# C - Nouvelle feature liée au temps depuis la prise du médicament
#     On utilise exp(-time_since_intake_off / 2.0) lorsque cette information est disponible, et NaN sinon.
#     Cette feature donne une estimation simple de l'effet du médicament au cours du temps, en utilisant une demi-vie d'environ 2 heures.
#     Elle est calculée à partir des données disponibles, sans utiliser la variable cible.
#
# Remarque :
# Nous avons essayé un encodage de la cible par patient, mais les résultats en validation étaient beaucoup moins bons. Les patients présents dans l'entraînement avaient des moyennes calculées à partir de leurs scores, alors que les patients de validation recevaient la moyenne générale (environ 37.5).
# Le modèle avait donc du mal à utiliser cette feature correctement.
# Nous avons gardé les statistiques historiques de l'expérience 15, qui permettent déjà de prendre en compte les différences entre patients.
#
# Fichiers utilisés :
# - data/X_train.csv : features des données d'entraînement.
# - data/y_train.csv : scores MDS-UPDRS à prédire.
# - data/X_test.csv  : features des données de test.
#
# Quelques colonnes importantes :
# patient_id, Index, off, on, ledd, age, age_at_diagnosis, time_since_intake_on, time_since_intake_off, rater_id et gene.
#
# Résultats :
# - Fichier de soumission : submissions/17_stacked_improvements.csv
# - Colonnes du fichier : Index, target
# - Skore Hub : 17_stacked_improvements
#
# Pour exécuter le script (dans le command terminal):
# .venv\Scripts\python.exe experiments/17_stacked_improvements.py


import warnings # Ignorer les avertissements pour garder une sortie plus lisible (on a eu des problèmes avec les avertissements)

import numpy as np
import pandas as pd
import skore
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from skore import Project, login
from skrub import tabular_pipeline

import lightgbm as lgb

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# Supprime les FutureWarning de pandas/LightGBM qui parasitent la sortie console sans affecter les résultats.
warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"


# 1. Chargement des données

# On charge les trois fichiers bruts. La fusion X_train + y_train se fait sur la colonne Index (identifiant de visite unique). Les assertions vérifient que ni X_train ni X_test ne contiennent la colonne target : toute présence de target à ce stade signalerait une fuite.
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

assert "target" not in X_train.columns
assert "target" not in X_test.columns

# Fusion pour obtenir un DataFrame visits complet (features + target).
visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train)

# y contient les scores MDS-UPDRS (cible de régression, entiers 0-132).
# groups contient les patient_id pour le GroupKFold (on ne doit jamais voir le même patient à la fois en train et en validation).
y      = visits["target"].reset_index(drop=True)
groups = visits["patient_id"].reset_index(drop=True)

print(f"Train: {len(visits)} rows, {groups.nunique()} patients")
print(f"Test : {len(X_test)} rows, {X_test['patient_id'].nunique()} patients")


# 2. Pré-calcul : historique patient (exp15 Bloc S) — fait une seule fois

# Le bloc S est pré-calculé UNE FOIS sur l'ensemble train et sur le test, avant le CV. Pour chaque visite k d'un patient, on calcule des statistiques des visites 1..k-1 (shift(1) → exclut la visite courante).

# Pourquoi on pré-calcule avant le CV:
#   Le bloc S n'utilise que des colonnes X (off, ledd) — jamais la target.
#   Il est donc safe de le calculer sur l'ensemble du train sans risque de fuite vers y. Le calculer une seule fois évite un coût O(n × n_folds).
#   Pendant le CV on découpe simplement hist_train.iloc[tr] / .iloc[val].

# Garantie anti-fuite du bloc S :
#   Le shift(1) assure que la visite courante ne contribue pas à sa propre statistique. Les patients de validation contribuent à leur propre historique (leurs visites antérieures) mais pas à l'historique des patients d'entraînement, car le calcul est intra-patient. 
#   Il n'y a donc pas de contamination croisée entre patients.

def compute_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    
"""Calcule des informations sur les visites précédentes de chaque patient.

    Pour chaque visite, on regarde uniquement les visites précédentes du même patient, en suivant l'ordre de la colonne Index.

    On utilise seulement les colonnes off et ledd, sans utiliser le score à prédire (target), pour éviter les fuites de données.

    Paramètres:
    df : pd.DataFrame - Tableau contenant les colonnes patient_id, Index, off et ledd.

    Retour:
    pd.DataFrame - Un tableau contenant cinq nouvelles colonnes :
        - off_hist_mean : moyenne des valeurs off des visites précédentes.
        - off_hist_count : nombre de valeurs off disponibles auparavant.
        - off_hist_std : écart-type des valeurs off précédentes.
        - ledd_hist_mean : moyenne des valeurs ledd des visites précédentes.
        - hist_n_visits : nombre de visites précédentes du patient.
          Pour la première visite, cette valeur est égale à 0.
    """
    assert "target" not in df.columns
    # Tri par (patient_id, Index) pour respecter l'ordre chronologique.
    s = df[["patient_id", "Index", "off", "ledd"]].sort_values(
        ["patient_id", "Index"]).copy()

    # Masques de validité : 1.0 si la valeur est présente, 0.0 sinon.
    # Utilisés comme dénominateur pour obtenir des moyennes correctes même en présence de NaN (expanding().sum() ignore les NaN par défaut).
    s["_off_valid"]  = s["off"].notna().astype(float)
    s["_ledd_valid"] = s["ledd"].notna().astype(float)

    # Somme cumulée décalée d'un rang (shift(1)) → exclut la visite courante.
    off_cs  = s.groupby("patient_id")["off"].transform(
        lambda x: x.expanding().sum().shift(1))
    off_cc  = s.groupby("patient_id")["_off_valid"].transform(
        lambda x: x.expanding().sum().shift(1))

    # Moyenne cumulée : somme / compteur (0 → NaN pour éviter la division par zéro).
    s["off_hist_mean"]  = off_cs / off_cc.replace(0, np.nan)
    s["off_hist_count"] = off_cc

    # Variance en ligne via E[X²] - E[X]² (formula de König-Huygens).
    # NaN si moins de 2 visites précédentes (variance non définie).
    off_css = s.groupby("patient_id")["off"].transform(
        lambda x: (x**2).expanding().sum().shift(1))
    var = (off_css / off_cc.replace(0, np.nan)) - s["off_hist_mean"] ** 2
    s["off_hist_std"]   = np.where(off_cc >= 2, np.sqrt(var.clip(lower=0)), np.nan)

    ledd_cs = s.groupby("patient_id")["ledd"].transform(
        lambda x: x.expanding().sum().shift(1))
    ledd_cc = s.groupby("patient_id")["_ledd_valid"].transform(
        lambda x: x.expanding().sum().shift(1))
    s["ledd_hist_mean"] = ledd_cs / ledd_cc.replace(0, np.nan)

    # Rang de visite (0 = première) : utile pour les features de position.
    s["hist_n_visits"]  = s.groupby("patient_id").cumcount()

    hist_cols = ["off_hist_mean", "off_hist_count", "off_hist_std",
                 "ledd_hist_mean", "hist_n_visits"]
    # Réindexe sur l'index original de df (df peut être dans n'importe quel ordre).
    return s[hist_cols].loc[df.index]


# visits_x : version features-only du train (sans la colonne target).
visits_x   = visits[list(X_train.columns)].copy()
hist_train = compute_patient_history(visits_x)
hist_test  = compute_patient_history(X_test)
print("Patient history pre-computed.")

# 3. Helpers de construction des features

# Colonnes numériques agrégées au niveau patient (Bloc B).
AGG_COLS_NUM  = ["off", "on", "ledd", "age", "age_at_diagnosis",
                 "time_since_intake_on", "time_since_intake_off"]

# Colonnes utilisées pour les features de voisinage temporel (Bloc C).
NEIGHBOR_COLS = ["off", "on", "ledd", "time_since_intake_off"]

# Demi-vie de la lévodopa en heures (Bloc D_levo, nouveau exp17).
# Source : pharmacocinétique standard. Demi-vie plasmatique ≈ 1-2 h ;
# on utilise 2.0 comme approximation conservative documentée dans CONTEXT.md.
LEVO_T_HALF   = 2.0

# Bornes physiques du score MDS-UPDRS Part III (Bloc B_clip, nouveau exp17).
# 0 = pas de symptômes moteurs, 132 = atteinte maximale (33 items × 4 points).
TARGET_MIN    = 0.0
TARGET_MAX    = 132.0


def _vectorised_slope(pid_arr, x_arr, y_arr):
    
"""Calcule la pente entre l'âge et une variable pour chaque patient.

    Pour chaque patient, on utilise une régression linéaire pour voir comment une variable évolue avec l'âge.

    On garde uniquement les visites où l'âge et la variable sont disponibles. Si un patient a moins de 3 visites valides, sa pente est remplacée par NaN, car le résultat serait moins fiable.

    Le calcul est fait avec groupby et agg pour traiter les patients sans avoir besoin d'une boucle pour chacun d'entre eux.

    Paramètres:
    pid_arr : Tableau contenant l'identifiant de chaque patient.
    x_arr : Valeurs de l'âge.
    y_arr : Valeurs de la variable étudiée (off, on, ledd, etc.).

    Retour:
    pd.Series:
        Une série contenant la pente calculée pour chaque patient.
        Si un patient a moins de 3 visites valides, la valeur est NaN.
    """

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() == 0:
        return pd.Series(dtype=float)
    pids, x, yv = pid_arr[mask], x_arr[mask], y_arr[mask]
    # Calcul des composantes OLS agrégées par patient :
    # pente = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
    agg = pd.DataFrame({"p": pids, "x": x, "y": yv, "xy": x*yv, "x2": x*x}
                       ).groupby("p", sort=False).agg(
        n=("x","count"), sx=("x","sum"), sy=("y","sum"),
        sxy=("xy","sum"), sx2=("x2","sum"))
    num = agg["n"]*agg["sxy"] - agg["sx"]*agg["sy"]
    den = agg["n"]*agg["sx2"] - agg["sx"]**2
    # where(den.abs() > 1e-10) : évite la division par zéro quand tous les x d'un patient sont identiques (âge constant → pente indéfinie).
    return (num / den.where(den.abs() > 1e-10)).where(agg["n"] >= 3)


def _compute_pat_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    
"""Calcule des statistiques pour chaque patient (Bloc B).

    Pour chaque patient, calcule la moyenne, la médiane, l'écart-type, le minimum, le maximum et le nombre de valeurs pour chaque colonne numérique de AGG_COLS_NUM.

    On calcule aussi la pente entre l'âge et chaque variable pour voir comment les valeurs évoluent au fil du temps.

    Les calculs sont faits uniquement à partir du tableau donné en entrée, qui peut correspondre aux données d'entraînement, de validation ou de test.

    Attention :
    Pendant la validation croisée, les statistiques sont calculées séparément pour les données d'entraînement et de validation.
    Les statistiques de validation utilisent donc uniquement les visites présentes dans ce groupe. Cela peut donner un résultat légèrementoptimiste pour les patients répartis entre plusieurs groupes.
    Cependant, on n'utilise jamais la colonne target pour ces calculs, donc il n'y a pas de fuite liée aux scores à prédire.

    Paramètres
    df : pd.DataFrame
        Tableau contenant les données des patients, sans la colonne target.
        Il doit contenir patient_id et les colonnes de AGG_COLS_NUM.

    Retour
    pd.DataFrame:
        Un tableau avec une ligne par patient et toutes les statistiques calculées. Les colonnes ajoutées se terminent par _pat.
        Ce tableau peut ensuite être associé aux données d'origine grâce à la colonne patient_id.
    """

    assert "target" not in df.columns
    gb = df.groupby("patient_id", sort=False)
    agg_funcs: dict = {}
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        for fn in ["mean", "median", "std", "min", "max", "count"]:
            agg_funcs[f"{col}_{fn}_pat"] = (col, fn)
    pat_df = gb.agg(**agg_funcs)
    pat_df = pat_df.join(
        gb["age"].agg(n_visits_pat="count", age_first_pat="min", age_last_pat="max"),
        how="left")

    # Durée totale de suivi du patient (en années si age en années).
    pat_df["followup_dur_pat"] = pat_df["age_last_pat"] - pat_df["age_first_pat"]

    # Première et dernière valeur non-NaN de off/on par patient, triées par Index.
    for col in ["off", "on"]:
        if col not in df.columns:
            continue
        notna = df.sort_values("Index")[["patient_id", col]].dropna(subset=[col])
        pat_df[f"{col}_first_pat"] = notna.groupby("patient_id")[col].first()
        pat_df[f"{col}_last_pat"]  = notna.groupby("patient_id")[col].last()

    # Pentes OLS (âge = variable) par patient pour capturer la trajectoire.
    pid_arr = df["patient_id"].values
    age_arr = df["age"].values.astype(float)
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        pat_df[f"{col}_slope_pat"] = _vectorised_slope(
            pid_arr, age_arr, pd.to_numeric(df[col], errors="coerce").values)
    return pat_df


def _rater_oof_bias(df_tr: pd.DataFrame, y_tr: np.ndarray) -> dict:
    
"""Calcule le biais moyen de chaque évaluateur à partir des données d'entraînement.

    Pour chaque évaluateur (rater_id), on calcule la différence moyenne entre le score réel (y_tr) et la valeur off.

    Si certaines valeurs off sont manquantes, on les remplace par la médiane des valeurs off du groupe d'entraînement.

    Le résultat permet de voir si certains évaluateurs ont tendance à donner des scores plus élevés ou plus faibles que prévu. Ces informations sont ensuite utilisées comme nouvelle variable (Bloc H).

    On utilise uniquement les données d'entraînement pour éviter d'utiliser les scores du groupe de validation dans les calculs.

    Paramètres:
    df_tr : pd.DataFrame
        Tableau des données d'entraînement contenant les colonnes
        off et rater_id.
    y_tr : np.ndarray
        Scores réels des patients dans le groupe d'entraînement.

    Retour:
    dict
        Dictionnaire contenant le biais moyen de chaque évaluateur,
        associé à son identifiant rater_id.
    """

    off_vals  = df_tr["off"].values
    raters    = df_tr["rater_id"].values
    # Fallback pour les NaN : médiane du fold (évite de perdre des lignes).
    off_fill  = np.where(np.isnan(off_vals), float(np.nanmedian(off_vals)), off_vals)
    residuals = y_tr - off_fill
    return {r: float(np.nanmean(residuals[raters == r])) for r in np.unique(raters)}


def _lgbm_encode(X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
   
"""Prépare les données avant de les utiliser avec LightGBM.

    Les colonnes catégorielles sont converties au format category pour que LightGBM puisse les traiter correctement.

    Les colonnes de type int8 sont converties en float pour éviter les avertissements lors de la préparation des données.

    Paramètres: 
    X : pd.DataFrame - tableau contenant les variables du modèle.
    cat_cols : list - liste des colonnes à convertir au format category.

    Retour:
    pd.DataFrame - une copie du tableau avec les types de colonnes adaptés.
    """

    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in X.select_dtypes("int8").columns:
        X[c] = X[c].astype(float)
    return X


def build_features(df_raw: pd.DataFrame,
                   rater_bias: dict,
                   pat_agg: pd.DataFrame,
                   hist_df: pd.DataFrame) -> pd.DataFrame:
   
"""Prépare toutes les variables utilisées par le modèle.

    Cette fonction regroupe les différentes variables créées dans les expériences précédentes (exp12 à exp16) et ajoute une nouvelle variable liée au temps depuis la prise de lévodopa (exp17).

    Elle est utilisée séparément pour les données d'entraînement, de validation et de test. Elle ne fait jamais appel à la colonne target.

    Les différents groupes de variables :
    
    B : Statistiques par patient (moyenne, médiane, écart-type, minimum,
        maximum, nombre de valeurs et pente).

    I : Nouvelles variables calculées à partir des données, comme le temps
        depuis le diagnostic et les différences entre on et off.

    E : Variables indiquant si certaines valeurs sont manquantes.

    D_pos : Position de chaque visite dans l'historique du patient
            (rang de la visite et temps écoulé depuis la première visite).

    C : Valeurs de la visite précédente et suivante pour certaines
        variables, ainsi que les différences d'âge entre les visites.

    A : Combinaisons entre les variables d'origine et les statistiques
        calculées pour chaque patient.

    F : Nombre de visites associées à chaque évaluateur.

    H : Biais moyen de chaque évaluateur, calculé à partir des données
        d'entraînement uniquement. Si un évaluateur n'est pas présent
        dans l'entraînement, on utilise une valeur moyenne par défaut.

    S : Informations sur les visites précédentes du patient, comme les
        moyennes de off et ledd, ainsi que le nombre de visites passées.
        On ajoute aussi la différence entre off et sa moyenne historique.

    D_levo : Nouvelle variable de l'expérience 17 qui estime l'effet de la
             lévodopa selon le temps écoulé depuis la prise du médicament.
             Si cette information est absente, la valeur est NaN.

    Paramètres
    df_raw : pd.DataFrame - Tableau contenant les variables d'origine, sans la colonne target.
    rater_bias : dict - Biais calculés pour chaque évaluateur à partir des données d'entraînement.
    pat_agg : pd.DataFrame
        Tableau contenant les statistiques calculées pour chaque patient.
    hist_df : pd.DataFrame
        Tableau contenant les informations sur les visites précédentes de chaque patient.

    Retour
    pd.DataFrame
        Tableau final contenant toutes les variables du modèle.
        Les colonnes patient_id et Index sont supprimées, car elles servent uniquement à identifier les patients et les visites.

        Les valeurs trop grandes dans les variables de type produit ou ratio sont limitées entre -10000 et 10000 pour éviter les valeurs extrêmes.
    """

    assert "target" not in df_raw.columns
    orig_index = df_raw.index.copy()
    feat = df_raw.copy()

    # Bloc B : agrégats patient
    # Jointure sur patient_id pour que chaque visite hérite des statistiques de son patient calculées sur le sous-ensemble courant (pas sur tout le train = pas de fuite inter-folds).

    feat = feat.join(pat_agg, on="patient_id", how="left")

    # Bloc I : features dérivées intra-visite
    feat["time_since_diagnosis"] = feat["age"] - feat["age_at_diagnosis"]
    mask_both = feat["off"].notna() & feat["on"].notna()
    feat["on_minus_off"] = np.where(mask_both, feat["on"] - feat["off"], np.nan)
    feat["on_div_off"]   = np.where(
        mask_both & (feat["off"].abs() > 1e-8), feat["on"] / feat["off"], np.nan)

    # Bloc E : indicateurs de données manquantes
    # LightGBM gère nativement les NaN, mais exposer le motif de manquant aide le modèle à apprendre des règles sur la disponibilité des données.
    for col in ["off", "on", "ledd", "gene",
                "time_since_intake_on", "time_since_intake_off"]:
        if col in feat.columns:
            feat[f"{col}_na"] = feat[col].isna().astype(np.int8)

    # Bloc D_pos : position de la visite dans la trajectoire patient visit_rank et rel_visit_pos permettent au modèle de distinguer les visites de début, milieu et fin de suivi (trajectoire non linéaire).

    feat_s = feat.sort_values(["patient_id", "age"])
    feat["visit_rank"] = feat_s.groupby("patient_id").cumcount().loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1)

    # Bloc C : voisins temporels
    # Pour chaque visite, fournit la valeur de la visite précédente et suivante ainsi que le gap d'âge. 
    # Capture la dérivée locale de la  progression sans utiliser la target.

    feat_nb = feat.sort_values(["patient_id", "age"])
    for col in NEIGHBOR_COLS:
        if col not in feat_nb.columns:
            continue
        prev_v = feat_nb.groupby("patient_id")[col].shift(1)
        next_v = feat_nb.groupby("patient_id")[col].shift(-1)
        age_s  = feat_nb["age"]
        age_p  = feat_nb.groupby("patient_id")["age"].shift(1)
        age_n  = feat_nb.groupby("patient_id")["age"].shift(-1)
        feat[f"{col}_prev"]         = prev_v.loc[orig_index].values
        feat[f"{col}_next"]         = next_v.loc[orig_index].values
        feat[f"{col}_age_gap_prev"] = (age_s - age_p).loc[orig_index].values
        feat[f"{col}_age_gap_next"] = (age_n - age_s).loc[orig_index].values

    # Bloc A : interactions (diff, ratio, prod) entre features brutes et agrégats patient. 
    # Permet au modèle de raisonner sur la déviation d'une visite par rapport au profil moyen du patient.
    for f_col, agg_col in [
        ("off","off_mean_pat"), ("on","on_mean_pat"), ("ledd","ledd_mean_pat"),
        ("age","off_mean_pat"), ("age_at_diagnosis","off_mean_pat"),
        ("age_at_diagnosis","ledd_mean_pat"),
    ]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"]  = feat[f_col] - feat[agg_col]
        denom = feat[agg_col].where(feat[agg_col].abs() > 1e-8)
        feat[f"{stub}_ratio"] = (feat[f_col] / denom).clip(-10, 10)
        feat[f"{stub}_prod"]  = feat[f_col] * feat[agg_col]
    for f_col, agg_col in [
        ("off","off_std_pat"), ("age","off_slope_pat"),
        ("age","on_slope_pat"), ("time_since_first_visit","off_slope_pat"),
    ]:
        if f_col not in feat.columns or agg_col not in feat.columns:
            continue
        stub = f"{f_col}_x_{agg_col}"
        feat[f"{stub}_diff"] = feat[f_col] - feat[agg_col]
        feat[f"{stub}_prod"] = feat[f_col] * feat[agg_col]

    # Bloc F : volume de visites par évaluateur
    # Proxy de la calibration de l'évaluateur (les évaluateurs fréquents tendent à être plus fiables / consistants).
    feat["rater_visit_count"] = feat.groupby("rater_id")["age"].transform("count")

    # Bloc H : biais OOF de l'évaluateur (exp15/16) 
    # rater_bias est calculé sur le fold train uniquement (appel à rater_oof_bias avec y_tr). 
    # Les évaluateurs absents du train reçoivent la moyenne des biais connus — fallback neutre, jamais de fuite target vers le fold de validation.

    global_fb = float(np.mean(list(rater_bias.values()))) if rater_bias else 0.0
    feat["rater_oof_bias"] = feat["rater_id"].map(rater_bias).fillna(global_fb)

    # Bloc S : historique séquentiel patient (exp15, inchangé)
    # hist_df est la tranche de hist_train ou hist_test correspondant aux indices de df_raw. Pré-calculé avant le CV sur X uniquement.
    # off_vs_hist_mean = déviation instantanée de off par rapport à la baseline patient → capte les fluctuations intra-patient.

    for col in hist_df.columns:
        feat[col] = hist_df.loc[orig_index, col].values
    feat["off_vs_hist_mean"] = feat["off"] - feat["off_hist_mean"]

    # Bloc D_levo : proxy de concentration lévodopa [NEW exp17]
    # levo_decay_off = exp(-t / T^1/2) où t = time_since_intake_off (heures)
    # et T½ = 2.0 h. Transforme un temps discret en un signal de concentration continue qui est directement lié à l'effet PD mesuré.
    # NaN si la colonne est absente ou manquante (les patients non médicamentés ne reçoivent pas de valeur fictive).

    t_off = pd.to_numeric(
        feat.get("time_since_intake_off", pd.Series(np.nan, index=feat.index)),
        errors="coerce")
    feat["levo_decay_off"] = np.where(
        t_off.notna(), np.exp(-t_off / LEVO_T_HALF), np.nan)

 
    # Nettoyage final

    assert "target" not in feat.columns  # Vérification anti-fuite finale.
    # Suppression des identifiants (patient_id, Index) : non prédictifs et
    # risque de sur-apprentissage si laissés dans la matrice.
    feat = feat.drop(columns=["patient_id", "Index"], errors="ignore")
    # Clipping des interactions pour stabiliser les gradients sur les outliers.
    for c in feat.select_dtypes("number").columns:
        if c.endswith("_prod") or c.endswith("_ratio"):
            feat[c] = feat[c].clip(-1e4, 1e4)
    assert (feat.index == orig_index).all()
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 5. CV — GroupKFold(5), paramètres LightGBM fixes
# ══════════════════════════════════════════════════════════════════════════════
# GroupKFold(5) : les 5 folds sont construits en garantissant que chaque patient
# n'apparaît que dans un seul fold de validation. C'est essentiel ici car les
# features du Bloc B et du Bloc H sont calculées per-patient : un patient à
# cheval sur train et validation entraînerait une fuite d'information.
#
# Les splits sont matérialisés une seule fois avant la boucle pour que
# le Hub push (section 7) réutilise exactement les mêmes folds.
cv_splits = list(GroupKFold(n_splits=5).split(
    np.arange(len(groups)), groups=groups))

# Hyperparamètres LightGBM — identiques à exp15/16, pas d'Optuna ici.
# Justifications des valeurs clés :
#   learning_rate=0.05 + 800 rounds : convergence lente pour réduire le
#       sur-apprentissage (early stopping arrête tôt si nécessaire).
#   num_leaves=63 : arbres modérément profonds, contrôle la complexité.
#   min_child_samples=50 : évite les feuilles de faible support (données
#       médicales avec peu de patients).
#   subsample=0.8, colsample_bytree=0.8 : sous-échantillonnage stochastique
#       pour la régularisation et la diversité des seeds (section 8).
#   reg_alpha, reg_lambda=0.1 : régularisation L1/L2 légère.
#   random_state=42 : seed de référence pour le CV (la section 8 varie les seeds).
LGBM_PARAMS = {
    "objective":         "regression_l2",
    "metric":            "rmse",
    "verbosity":         -1,
    "n_jobs":            -1,
    "random_state":      42,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
}

print()
print("=" * 60)
print("CV — GroupKFold(5), all improvements stacked")
print("=" * 60)

# oof : tableau des prédictions Out-Of-Fold (OOF). Chaque entrée reçoit la
# prédiction du modèle entraîné sur les 4 autres folds. Utilisé pour la
# recalibration isotonique (section 6) et le calcul du RMSE CV.
oof        = np.full(len(y), np.nan)
fold_rmses = []

for fold_i, (tr, val) in enumerate(cv_splits):
    # Découpe features/targets/groupes selon les indices du fold courant.
    df_tr  = visits_x.iloc[tr]
    df_val = visits_x.iloc[val]
    y_tr   = y.iloc[tr].values
    y_val  = y.iloc[val].values

    # ---------------------------------------------------------------
    # Calculs dépendant du fold (tous sur df_tr uniquement → pas de fuite)
    # rater_bias : calculé sur y_tr et df_tr → biais évaluateur fold-safe.
    # pat_agg_tr : statistiques patient sur le fold train.
    # pat_agg_val : statistiques patient sur le fold val (voir note dans
    #               _compute_pat_aggregates — légèrement optimiste mais sans
    #               fuite target car df_val ne contient pas de target).
    # hist_train.iloc[tr/val] : tranche de l'historique pré-calculé.
    # ---------------------------------------------------------------
    rater_bias  = _rater_oof_bias(df_tr, y_tr)
    pat_agg_tr  = _compute_pat_aggregates(df_tr)
    pat_agg_val = _compute_pat_aggregates(df_val)

    X_tr_f  = build_features(df_tr,  rater_bias, pat_agg_tr,  hist_train.iloc[tr])
    X_val_f = build_features(df_val, rater_bias, pat_agg_val, hist_train.iloc[val])

    # Alignement des colonnes : si une colonne apparaît en train mais pas en
    # val (ex. catégorielle absente du fold), on l'ajoute avec NaN pour que
    # LightGBM puisse prédire sans erreur de structure.
    for c in set(X_tr_f.columns) - set(X_val_f.columns):
        X_val_f[c] = np.nan
    X_val_f = X_val_f[X_tr_f.columns]

    # Détection des colonnes catégorielles (rater_id, gene, etc.) pour l'API
    # LightGBM native (dtype "category" activé dans _lgbm_encode).
    cat_cols = [c for c in X_tr_f.columns
                if str(X_tr_f[c].dtype) in ("object", "string[python]", "str")]
    X_tr_f  = _lgbm_encode(X_tr_f,  cat_cols)
    X_val_f = _lgbm_encode(X_val_f, cat_cols)

    lgb_tr  = lgb.Dataset(X_tr_f,  label=y_tr,  categorical_feature=cat_cols,
                           free_raw_data=False)
    lgb_val = lgb.Dataset(X_val_f, label=y_val, categorical_feature=cat_cols,
                           free_raw_data=False)

    # early_stopping(50) : arrête si le RMSE validation ne s'améliore pas
    # durant 50 rounds consécutifs. Évite le sur-apprentissage sans Optuna.
    model = lgb.train(
        LGBM_PARAMS, lgb_tr,
        num_boost_round=800,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    preds = model.predict(X_val_f)
    rmse  = root_mean_squared_error(y_val, preds)
    fold_rmses.append(rmse)
    # On stocke les prédictions OOF dans le tableau global aux indices du fold.
    oof[val] = preds
    print(f"  fold {fold_i+1}/5: RMSE={rmse:.4f}  best_iter={model.best_iteration}")

mean_rmse = float(np.mean(fold_rmses))
std_rmse  = float(np.std(fold_rmses))
print(f"\nCV result: mean={mean_rmse:.4f}  std={std_rmse:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Recalibration isotonique sur les OOF (exp16, inchangé)
# ══════════════════════════════════════════════════════════════════════════════
# Principe : LightGBM avec objectif L2 régresse vers la moyenne et produit
# des prédictions extrêmes sous-estimées. On ajuste une régression isotonique
# (non décroissante) sur (oof, y) — l'ensemble OOF complet (5 folds) —
# pour corriger cette tendance.
#
# Garantie anti-fuite :
#   Le calibrateur voit oof[i] (prédiction du modèle entraîné SANS la visite i)
#   et y[i]. La paire (oof[i], y[i]) n'a jamais été vue ensemble lors de
#   l'entraînement du modèle qui a produit oof[i]. C'est la garantie
#   fondamentale des prédictions OOF.
#
# Décision conditionnelle :
#   On n'applique la recalibration que si elle améliore le RMSE d'au moins
#   0.002 point. Ce seuil évite de l'appliquer quand le gain est dans le
#   bruit de mesure du CV.
valid      = ~np.isnan(oof)
iso        = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof[valid], y.values[valid])
oof_iso    = oof.copy()
oof_iso[valid] = iso.predict(oof[valid])
rmse_iso   = root_mean_squared_error(y.values[valid], oof_iso[valid])
use_iso    = rmse_iso < mean_rmse - 0.002
print(f"\nIsotonic: raw={mean_rmse:.4f}  calibrated={rmse_iso:.4f}  "
      f"({'applied' if use_iso else 'skipped'})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Push Hub — tabular_pipeline sur la matrice de features complète
#
# Même pattern que exp11-13 : skore.evaluate reçoit la matrice pré-construite
# et les cv_splits identiques à ceux du CV principal. tabular_pipeline
# ("regressor") sert de modèle proxy pour produire un CrossValidationReport
# complet visible sur le Hub (métriques réelles + plots diagnostics).
#
# Note : la matrice X_hub est construite avec rater_bias_hub et pat_agg_hub
# calculés sur tout le train (pas fold par fold). C'est correct car ces
# matrices ne servent qu'à l'affichage du rapport Hub, pas à la métriques CV
# officielle qui est calculée à la section 5 (fold par fold).
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Pushing report to Skore Hub (Mathias Kozman)")
print("=" * 60)

# Biais évaluateur et agrégats patient calculés sur tout le train pour le Hub.
rater_bias_hub = _rater_oof_bias(visits_x, y.values)
pat_agg_hub    = _compute_pat_aggregates(visits_x)
X_hub = build_features(visits_x, rater_bias_hub, pat_agg_hub, hist_train)

hub_report = skore.evaluate(
    tabular_pipeline("regressor"),
    X_hub, y,
    splitter=cv_splits,
)

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("17_stacked_improvements", hub_report)
print("Report pushed: 17_stacked_improvements")
print(f"Hub: https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Modèle final — ensemble 5 seeds [NEW exp17]
# ══════════════════════════════════════════════════════════════════════════════
# Principe de l'ensemble par seeds :
#   LightGBM avec subsample=0.8 et colsample_bytree=0.8 est stochastique :
#   chaque arbre n'utilise qu'un sous-ensemble aléatoire de lignes et de
#   colonnes. Deux entraînements avec des seeds différents produisent des
#   modèles légèrement différents dont les erreurs sont faiblement corrélées.
#   La moyenne de 5 prédictions réduit la variance sans augmenter le biais.
#
#   Seeds 0-4 (et non seed unique 42) : couvre un espace de diversité suffisant
#   sans coût excessif (5× le temps d'entraînement final, ~5 × 3s = 15s).
#
#   Le modèle final est entraîné sur TOUT le train (pas de validation split) :
#   on utilise le nombre de rounds fixé à 800 (early stopping non applicable
#   sans ensemble de validation). C'est cohérent avec le fait que best_iteration
#   varie peu entre folds (observé ~600-750 au CV).
#
# rater_bias_hub et pat_agg_hub sont réutilisés (calculés sur tout le train).
# pat_agg_test_17 : agrégats patient calculés sur X_test (visites de test
#   uniquement) — cohérent avec la convention de build_features où pat_agg
#   reflète toujours les données disponibles pour le sous-ensemble.
print()
print("=" * 60)
print("Training final model (5-seed ensemble)")
print("=" * 60)

# Réutilisation des matrices déjà calculées à la section 7 pour le train.
rater_bias_all  = rater_bias_hub   # biais évaluateurs sur tout le train
pat_agg_all     = pat_agg_hub      # agrégats patients sur tout le train
pat_agg_test_17 = _compute_pat_aggregates(X_test)

X_train_feat = build_features(visits_x, rater_bias_all, pat_agg_all, hist_train)
X_test_feat  = build_features(X_test,   rater_bias_all, pat_agg_test_17, hist_test)

# Alignement train → test : colonnes absentes du test remplies de NaN.
for c in set(X_train_feat.columns) - set(X_test_feat.columns):
    X_test_feat[c] = np.nan
X_test_feat = X_test_feat[X_train_feat.columns]

# Assertions de sécurité : s'assurer que ni la target ni les identifiants
# n'ont filtré dans la matrice finale.
assert "target"     not in X_train_feat.columns
assert "patient_id" not in X_train_feat.columns
assert "Index"      not in X_train_feat.columns

cat_cols_final = [c for c in X_train_feat.columns
                  if str(X_train_feat[c].dtype) in ("object", "string[python]", "str")]
X_train_feat = _lgbm_encode(X_train_feat, cat_cols_final)
X_test_feat  = _lgbm_encode(X_test_feat,  cat_cols_final)

# Boucle sur 5 seeds : chaque seed_params copie LGBM_PARAMS en ne changeant
# que random_state. La moyenne des prédictions est l'ensemble final.
seed_preds = []
for seed in range(5):
    seed_params = {**LGBM_PARAMS, "random_state": seed}
    ds_full = lgb.Dataset(X_train_feat, label=y.values,
                           categorical_feature=cat_cols_final, free_raw_data=False)
    mdl = lgb.train(seed_params, ds_full,
                    num_boost_round=800,
                    callbacks=[lgb.log_evaluation(-1)])
    seed_preds.append(mdl.predict(X_test_feat))
    print(f"  seed {seed} trained")

# Moyenne des prédictions des 5 modèles (réduction de la variance).
preds_test = np.mean(seed_preds, axis=0)

# Application conditionnelle de la recalibration isotonique (si elle a aidé
# sur les OOF à la section 6). Le calibrateur `iso` est fitted sur les OOF
# du CV seed=42 uniquement — approximation acceptable car les 5 seeds
# produisent des distributions de prédictions proches.
if use_iso:
    preds_test = iso.predict(preds_test)

# Clipping physique [0, 132] — dernière étape, après recalibration.
# MDS-UPDRS Part III ne peut pas être négatif ni dépasser 132 ; tout score
# hors de cette plage est une erreur de modèle, pas une information utile.
preds_test = np.clip(preds_test, TARGET_MIN, TARGET_MAX)

# Vérifications finales : aucun NaN / inf ne doit passer dans la soumission.
assert not np.isnan(preds_test).any()
assert not np.isinf(preds_test).any()
assert len(preds_test) == len(pd.read_csv(DATA_DIR / "sample_submission.csv"))

# ══════════════════════════════════════════════════════════════════════════════
# 9. Fichier de soumission
# ══════════════════════════════════════════════════════════════════════════════
# L'ordre des Index doit correspondre exactement à X_test (assertion vérifiée).
submission = X_test[["Index"]].copy()
submission["target"] = preds_test
assert list(submission.columns) == ["Index", "target"]
assert submission["Index"].tolist() == X_test["Index"].tolist()

out = PROJECT_ROOT / "submissions"
out.mkdir(exist_ok=True)
submission.to_csv(out / "17_stacked_improvements.csv", index=False)
print(f"\nSubmission: submissions/17_stacked_improvements.csv  ({len(submission)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Résumé
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"exp16 reference              : RMSE=3.911")
print(f"exp17 CV RMSE (raw)          : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"exp17 CV RMSE (+ isotonic)   : {rmse_iso:.4f}")
print()
print("Methods stacked (cumulative):")
print("  exp13  LightGBM + exp12 feature set")
print("  exp15  + patient sequential history (Block S)")
print("  exp16  + isotonic recalibration on OOF")
print("  exp17  + levodopa decay proxy (new)")
print("  exp17  + 5-seed ensemble (new)")
print("  exp17  + target clipping [0, 132] (new)")
print()
print(f"Hub report : 17_stacked_improvements")
print(f"Submission : submissions/17_stacked_improvements.csv")
print()
print("OK: experiment 17 complete.")
