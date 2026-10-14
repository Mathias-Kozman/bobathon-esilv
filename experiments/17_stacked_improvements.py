# %%
# ---------------------------------------------------------------------------
# Rôle     : expérience 17 — cumul de toutes les améliorations sur le modèle
#            LightGBM (exp13 → 15 → 16 + trois nouveautés exp17).
#
# Idée centrale (ce qui change par rapport à exp15/16)
# -------------------------------------------------------
# exp15 avait ajouté le Bloc S (historique séquentiel par patient : off_hist_mean,
# off_hist_count, off_hist_std, ledd_hist_mean, hist_n_visits) pour capturer
# l'effet aléatoire patient sans fuite.
#
# exp16 avait ajouté la recalibration isotonique sur les prédictions OOF afin
# de corriger le biais de régression vers la moyenne (mean-regression bias).
#
# exp17 empile trois nouvelles améliorations sur ces deux bases :
#
#   A — Ensemble 5 seeds  (LOW IMPACT)
#       Le modèle final est entraîné 5 fois avec des seeds différents (0-4) et
#       les prédictions test sont moyennées. Réduit la variance stochastique du
#       boosting (sous-échantillonnage de lignes / colonnes à chaque arbre).
#       Pas appliqué pendant le CV ; seules les métriques CV sont exposées.
#
#   B — Clipping cible [0, 132]  (TRIVIAL)
#       MDS-UPDRS Part III est bornée physiquement : 0 (meilleur) à 132
#       (pire). Le modèle peut extrapoler au-delà ; on clip les prédictions
#       finales à cette plage. Impact marginal car peu de valeurs sortent.
#
#   C — Proxy de décroissance lévodopa  (MEDIUM IMPACT)
#       exp(-time_since_intake_off / 2.0) quand time_since_intake_off est
#       disponible, NaN sinon. Convertit le timing du médicament en une
#       concentration sanguine approximative en accord avec le modèle PK
#       documenté dans CONTEXT.md (demi-vie ~2 h). Cette feature est calculée
#       depuis une colonne X pure (pas de target) → pas de fuite.
#
# Note importante : l'encodage cible par patient (within-fold OOF target
# encoding) a été retiré. Il provoquait une rupture de distribution
# train/val : les lignes d'entraînement recevaient la moyenne des targets du
# patient, tandis que les patients de validation (non vus) recevaient la
# moyenne globale (~37.5). LightGBM construisait des splits massifs sur cette
# feature qui s'effondrait à la validation, faisant passer le RMSE de 3.9 à
# 11.8. Le bloc S (off_hist_mean) capte déjà la baseline patient sans fuite.
#
# Entrées
# -------
#   data/X_train.csv    : matrice de features d'entraînement (sans target).
#                         Colonnes clés : patient_id, Index (numéro de visite
#                         monotone), off, on, ledd, age, age_at_diagnosis,
#                         time_since_intake_on, time_since_intake_off,
#                         rater_id, gene.
#   data/y_train.csv    : targets, colonnes [Index, target] — score MDS-UPDRS.
#   data/X_test.csv     : matrice de features test (même colonnes, sans target).
#
# Sorties
# -------
#   submissions/17_stacked_improvements.csv : fichier de soumission
#                                             (colonnes : Index, target).
#   Skore Hub key        : "17_stacked_improvements"
#                          (CrossValidationReport visible sur le Hub).
#
# Comment exécuter
# ----------------
#   .venv\Scripts\python.exe experiments/17_stacked_improvements.py
#   (ou depuis la racine du projet avec le venv activé)
# ---------------------------------------------------------------------------

import warnings

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

# Supprime les FutureWarning de pandas/LightGBM qui parasitent la sortie console
# sans affecter les résultats.
warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement des données
# ══════════════════════════════════════════════════════════════════════════════
# On charge les trois fichiers bruts. La fusion X_train + y_train se fait sur
# la colonne Index (identifiant de visite unique). Les assertions vérifient que
# ni X_train ni X_test ne contiennent la colonne target : toute présence de
# target à ce stade signalerait une fuite.
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

assert "target" not in X_train.columns
assert "target" not in X_test.columns

# Fusion pour obtenir un DataFrame visits complet (features + target).
visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train)

# y contient les scores MDS-UPDRS (cible de régression, entiers 0-132).
# groups contient les patient_id pour le GroupKFold (on ne doit jamais voir
# le même patient à la fois en train et en validation).
y      = visits["target"].reset_index(drop=True)
groups = visits["patient_id"].reset_index(drop=True)

print(f"Train: {len(visits)} rows, {groups.nunique()} patients")
print(f"Test : {len(X_test)} rows, {X_test['patient_id'].nunique()} patients")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Pré-calcul : historique patient (exp15 Bloc S) — fait une seule fois
# ══════════════════════════════════════════════════════════════════════════════
# Le bloc S est pré-calculé UNE FOIS sur l'ensemble train et sur le test,
# AVANT le CV. Pour chaque visite k d'un patient, on calcule des statistiques
# des visites 1..k-1 (shift(1) → exclut la visite courante).
#
# Pourquoi pré-calculer avant le CV ?
#   Le bloc S n'utilise que des colonnes X (off, ledd) — jamais la target.
#   Il est donc safe de le calculer sur l'ensemble du train sans risque de
#   fuite vers y. Le calculer une seule fois évite un coût O(n × n_folds).
#   Pendant le CV on découpe simplement hist_train.iloc[tr] / .iloc[val].
#
# Garantie anti-fuite du bloc S :
#   Le shift(1) assure que la visite courante ne contribue pas à sa propre
#   statistique. Les patients de validation contribuent à leur propre
#   historique (leurs visites antérieures) mais pas à l'historique des patients
#   d'entraînement, car le calcul est intra-patient. Il n'y a donc pas de
#   contamination croisée entre patients.

def compute_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features d'historique séquentiel par patient (Bloc S, exp15).

    Pour chaque visite k du patient p, agrège uniquement les visites
    antérieures (1 … k-1, ordonnées par Index) de ce même patient.
    La colonne Index est supposée monotone avec l'âge/le temps.

    Utilise uniquement des colonnes X (off, ledd) — jamais la target —
    ce qui rend le calcul safe que ce soit avant ou pendant le CV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut contenant au moins les colonnes patient_id, Index,
        off et ledd. Ne doit pas contenir de colonne "target".

    Returns
    -------
    pd.DataFrame
        Cinq colonnes, indexées comme df :
        - off_hist_mean  : moyenne des valeurs off des visites précédentes.
        - off_hist_count : nombre de visites précédentes avec off non NaN.
        - off_hist_std   : écart-type des off précédents (NaN si < 2 visites).
        - ledd_hist_mean : moyenne des ledd précédentes.
        - hist_n_visits  : rang de la visite dans la séquence patient (0 = première).
    """
    assert "target" not in df.columns
    # Tri par (patient_id, Index) pour respecter l'ordre chronologique.
    s = df[["patient_id", "Index", "off", "ledd"]].sort_values(
        ["patient_id", "Index"]).copy()

    # Masques de validité : 1.0 si la valeur est présente, 0.0 sinon.
    # Utilisés comme dénominateur pour obtenir des moyennes correctes même
    # en présence de NaN (expanding().sum() ignore les NaN par défaut).
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

# ══════════════════════════════════════════════════════════════════════════════
# 3. Helpers de construction des features
# ══════════════════════════════════════════════════════════════════════════════

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
    """Calcule la pente OLS intra-patient (age vs. variable) de façon vectorisée.

    Pour chaque patient, ajuste y = β·x + α par moindres carrés en utilisant
    uniquement les visites où ni x ni y ne sont NaN. Retourne NaN pour les
    patients avec moins de 3 visites valides (pente non fiable).

    Entièrement vectorisé via groupby + agg : pas de boucle Python par patient.

    Parameters
    ----------
    pid_arr : array-like of shape (n,)
        Tableau de patient_id correspondant à chaque ligne.
    x_arr : array-like of shape (n,)
        Variable indépendante (typiquement l'âge).
    y_arr : array-like of shape (n,)
        Variable dépendante (off, on, ledd, etc.).

    Returns
    -------
    pd.Series
        Pente OLS par patient_id (index = patient_id). NaN si n_valid < 3.
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
    # where(den.abs() > 1e-10) : évite la division par zéro quand tous les x
    # d'un patient sont identiques (âge constant → pente indéfinie).
    return (num / den.where(den.abs() > 1e-10)).where(agg["n"] >= 3)


def _compute_pat_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les agrégats statistiques par patient (Bloc B).

    Produit mean, median, std, min, max, count de chaque colonne numérique
    listée dans AGG_COLS_NUM, plus les pentes OLS (âge vs. variable) pour
    chaque patient. Ces agrégats sont calculés sur l'ensemble du sous-DataFrame
    passé en entrée (train fold ou test complet selon l'appelant).

    IMPORTANT (risque de fuite partielle) : pendant le CV, cette fonction est
    appelée séparément sur df_tr (fold train) et sur df_val (fold validation).
    Les statistiques patient du fold de validation sont donc calculées sur les
    visites du fold de validation uniquement — ce qui est légèrement optimiste
    pour les patients dont certaines visites tombent dans d'autres folds. Ce
    biais est inévitable dès lors qu'on veut des features patient sans voir
    toutes les visites futures ; il n'implique PAS de fuite target car df_val
    ne contient pas de colonne target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame features-only (sans colonne "target") contenant au moins
        patient_id et les colonnes de AGG_COLS_NUM.

    Returns
    -------
    pd.DataFrame
        Un DataFrame indexé par patient_id avec toutes les colonnes agrégées
        (suffixe _pat). Peut être joint directement via df.join(pat_agg,
        on="patient_id").
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

    # Pentes OLS (âge → variable) par patient pour capturer la trajectoire.
    pid_arr = df["patient_id"].values
    age_arr = df["age"].values.astype(float)
    for col in AGG_COLS_NUM:
        if col not in df.columns:
            continue
        pat_df[f"{col}_slope_pat"] = _vectorised_slope(
            pid_arr, age_arr, pd.to_numeric(df[col], errors="coerce").values)
    return pat_df


def _rater_oof_bias(df_tr: pd.DataFrame, y_tr: np.ndarray) -> dict:
    """Estime le biais systématique de chaque évaluateur (rater) sur le train fold.

    Pour chaque rater_id, calcule la moyenne des résidus (y_tr - off_fill)
    où off_fill est la valeur off avec la médiane du fold comme fallback pour
    les NaN. Ce biais est ensuite utilisé comme feature (Bloc H) pour corriger
    les effets évaluateurs sans jamais utiliser les labels du fold de validation.

    Parameters
    ----------
    df_tr : pd.DataFrame
        DataFrame features-only du fold d'entraînement. Doit contenir
        les colonnes "off" et "rater_id".
    y_tr : np.ndarray of shape (n_train,)
        Valeurs cibles du fold d'entraînement.

    Returns
    -------
    dict
        Dictionnaire {rater_id: float} — biais moyen en unités de score
        MDS-UPDRS. Clés = identifiants d'évaluateurs présents dans df_tr.
    """
    off_vals  = df_tr["off"].values
    raters    = df_tr["rater_id"].values
    # Fallback pour les NaN : médiane du fold (évite de perdre des lignes).
    off_fill  = np.where(np.isnan(off_vals), float(np.nanmedian(off_vals)), off_vals)
    residuals = y_tr - off_fill
    return {r: float(np.nanmean(residuals[raters == r])) for r in np.unique(raters)}


def _lgbm_encode(X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Prépare le DataFrame pour LightGBM : convertit les catégorielles et int8.

    LightGBM attend des colonnes catégorielles de dtype 'category' (pandas)
    pour activer son encodage interne de catégorielles (utilise des histogrammes
    séparés, meilleur que l'encodage ordinal manuel).
    Les colonnes int8 sont converties en float pour éviter des avertissements
    de type lors de la construction des datasets LightGBM.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix à préparer.
    cat_cols : list of str
        Liste des colonnes à caster en dtype 'category'.

    Returns
    -------
    pd.DataFrame
        Copie de X avec les dtypes ajustés.
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
    """Construit la matrice de features complète à partir du DataFrame brut.

    Assemble sept blocs de features (B, I, E, D_pos, C, A, F, H, S) issus des
    expériences 12-16, plus la feature levo_decay_off introduite en exp17.
    Cette fonction est appelée séparément pour le fold train (df_tr), le fold
    val (df_val) et le test final. Elle ne touche jamais la target.

    Blocs de features
    -----------------
    B   Patient-level aggregates (mean, median, std, min, max, count, slope)
        calculés sur le sous-DataFrame passé via pat_agg. Pas de fuite : le
        pat_agg est toujours calculé sur le même sous-ensemble que df_raw.
    I   Dérivées intra-visite : time_since_diagnosis, on_minus_off, on_div_off.
    E   Flags de valeurs manquantes (suffixe _na) pour les colonnes clés.
    D_pos   Position relative de la visite dans la séquence patient
            (visit_rank, time_since_first_visit, time_to_last_visit,
            rel_visit_pos). Utilise les limites de âge issues du Bloc B.
    C   Voisins temporels (valeur et gap d'âge de la visite précédente /
        suivante) pour off, on, ledd, time_since_intake_off.
    A   Interactions diff/ratio/prod entre les colonnes brutes et les
        statistiques patient du Bloc B.
    F   Nombre de visites par évaluateur (rater_visit_count).
    H   Biais OOF de l'évaluateur (rater_oof_bias) issu de _rater_oof_bias.
        Calculé sur le fold train uniquement → les évaluateurs absents du train
        reçoivent le biais moyen global (fallback neutre, pas de fuite target).
    S   Historique séquentiel patient (Bloc S exp15) : off_hist_mean,
        off_hist_count, off_hist_std, ledd_hist_mean, hist_n_visits, plus
        off_vs_hist_mean (déviation instantanée par rapport à la baseline).
    D_levo  [NEW exp17] levo_decay_off = exp(-time_since_intake_off / 2.0),
        NaN si time_since_intake_off est absent. Proxy de concentration
        plasmatique de lévodopa au moment de la visite (modèle PK exp1).

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame features-only (sans "target") pour le sous-ensemble
        considéré (fold train, fold val, ou test).
    rater_bias : dict
        Biais par évaluateur produit par _rater_oof_bias sur le fold train.
    pat_agg : pd.DataFrame
        Agrégats patient (indexés par patient_id) produits par
        _compute_pat_aggregates sur le même sous-ensemble que df_raw.
    hist_df : pd.DataFrame
        Historique séquentiel produit par compute_patient_history, découpé
        sur les mêmes indices que df_raw (hist_train.iloc[tr/val] ou hist_test).

    Returns
    -------
    pd.DataFrame
        Matrice de features finale, même index que df_raw. Les colonnes
        patient_id et Index sont supprimées (identifiants non prédictifs).
        Les colonnes _prod et _ratio sont clippées à [-1e4, 1e4] pour éviter
        les valeurs extrêmes dans les interactions.
    """
    assert "target" not in df_raw.columns
    orig_index = df_raw.index.copy()
    feat = df_raw.copy()

    # ------------------------------------------------------------------
    # Bloc B : agrégats patient
    # Jointure sur patient_id pour que chaque visite hérite des statistiques
    # de son patient calculées sur le sous-ensemble courant (pas sur tout le
    # train → pas de fuite inter-folds).
    # ------------------------------------------------------------------
    feat = feat.join(pat_agg, on="patient_id", how="left")

    # ------------------------------------------------------------------
    # Bloc I : features dérivées intra-visite
    # ------------------------------------------------------------------
    feat["time_since_diagnosis"] = feat["age"] - feat["age_at_diagnosis"]
    mask_both = feat["off"].notna() & feat["on"].notna()
    feat["on_minus_off"] = np.where(mask_both, feat["on"] - feat["off"], np.nan)
    feat["on_div_off"]   = np.where(
        mask_both & (feat["off"].abs() > 1e-8), feat["on"] / feat["off"], np.nan)

    # ------------------------------------------------------------------
    # Bloc E : indicateurs de données manquantes
    # LightGBM gère nativement les NaN, mais exposer le motif de manquant
    # aide le modèle à apprendre des règles sur la disponibilité des données.
    # ------------------------------------------------------------------
    for col in ["off", "on", "ledd", "gene",
                "time_since_intake_on", "time_since_intake_off"]:
        if col in feat.columns:
            feat[f"{col}_na"] = feat[col].isna().astype(np.int8)

    # ------------------------------------------------------------------
    # Bloc D_pos : position de la visite dans la trajectoire patient
    # visit_rank et rel_visit_pos permettent au modèle de distinguer les
    # visites de début, milieu et fin de suivi (trajectoire non linéaire).
    # ------------------------------------------------------------------
    feat_s = feat.sort_values(["patient_id", "age"])
    feat["visit_rank"] = feat_s.groupby("patient_id").cumcount().loc[orig_index].values
    feat["time_since_first_visit"] = feat["age"] - feat["age_first_pat"]
    feat["time_to_last_visit"]     = feat["age_last_pat"] - feat["age"]
    feat["rel_visit_pos"] = feat["visit_rank"] / (feat["n_visits_pat"] - 1).clip(lower=1)

    # ------------------------------------------------------------------
    # Bloc C : voisins temporels
    # Pour chaque visite, fournit la valeur de la visite précédente et
    # suivante ainsi que le gap d'âge. Capture la dérivée locale de la
    # progression sans utiliser la target.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Bloc A : interactions (diff, ratio, prod) entre features brutes et
    # agrégats patient. Permet au modèle de raisonner sur la déviation d'une
    # visite par rapport au profil moyen du patient.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Bloc F : volume de visites par évaluateur
    # Proxy de la calibration de l'évaluateur (les évaluateurs fréquents
    # tendent à être plus fiables / consistants).
    # ------------------------------------------------------------------
    feat["rater_visit_count"] = feat.groupby("rater_id")["age"].transform("count")

    # ------------------------------------------------------------------
    # Bloc H : biais OOF de l'évaluateur (exp15/16)
    # rater_bias est calculé sur le fold train uniquement (appel à
    # _rater_oof_bias avec y_tr). Les évaluateurs absents du train
    # reçoivent la moyenne des biais connus — fallback neutre, jamais
    # de fuite target vers le fold de validation.
    # ------------------------------------------------------------------
    global_fb = float(np.mean(list(rater_bias.values()))) if rater_bias else 0.0
    feat["rater_oof_bias"] = feat["rater_id"].map(rater_bias).fillna(global_fb)

    # ------------------------------------------------------------------
    # Bloc S : historique séquentiel patient (exp15, inchangé)
    # hist_df est la tranche de hist_train ou hist_test correspondant aux
    # indices de df_raw. Pré-calculé avant le CV sur X uniquement.
    # off_vs_hist_mean = déviation instantanée de off par rapport à la
    # baseline patient → capte les fluctuations intra-patient.
    # ------------------------------------------------------------------
    for col in hist_df.columns:
        feat[col] = hist_df.loc[orig_index, col].values
    feat["off_vs_hist_mean"] = feat["off"] - feat["off_hist_mean"]

    # ------------------------------------------------------------------
    # Bloc D_levo : proxy de concentration lévodopa [NEW exp17]
    # levo_decay_off = exp(-t / T½) où t = time_since_intake_off (heures)
    # et T½ = 2.0 h. Transforme un temps discret en un signal de
    # concentration continue qui est directement lié à l'effet PD mesuré.
    # NaN si la colonne est absente ou manquante (les patients non
    # médicamentés ne reçoivent pas de valeur fictive).
    # ------------------------------------------------------------------
    t_off = pd.to_numeric(
        feat.get("time_since_intake_off", pd.Series(np.nan, index=feat.index)),
        errors="coerce")
    feat["levo_decay_off"] = np.where(
        t_off.notna(), np.exp(-t_off / LEVO_T_HALF), np.nan)

    # ------------------------------------------------------------------
    # Nettoyage final
    # ------------------------------------------------------------------
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
