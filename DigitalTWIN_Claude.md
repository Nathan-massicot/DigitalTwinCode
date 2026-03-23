# Prompt pour Claude Code — Digital Twin Stroke Rehabilitation Prototype



## Contexte

Je développe un prototype d'application **Digital Twin** pour la rééducation post-AVC (stroke). L'objectif est de prédire l'évolution fonctionnelle d'un patient en rééducation à partir de ses paramètres cliniques d'entrée. C'est un **prototype avec données simulées** (pas de vrai modèle ML pour l'instant).

## Stack technique

- **Python 3.12** avec **uv** comme gestionnaire de packages
- **Streamlit** comme framework UI
- Crée un fichier `pyproject.toml` compatible uv avec les dépendances nécessaires
- L'app doit se lancer avec `uv run streamlit run app.py`
- Utilise `plotly` pour les visualisations (jauge de santé, boxplot)
- Utilise `numpy` et `pandas` pour la manipulation de données

## Architecture fichiers

```
digital-twin-prototype/
├── pyproject.toml
├── app.py                  # Point d'entrée Streamlit
├── data/
│   └── mock_patients.py    # Générateur de patients simulés (cohorte de ~200 patients)
├── engine/
│   ├── predictor.py        # Logique de prédiction (k-NN pondéré sur la cohorte simulée)
│   └── scoring.py          # Calcul du health status score + intervalle de confiance
└── ui/
    ├── sliders.py          # Composant des 20 sliders de paramètres
    ├── health_gauge.py     # Jauge rouge-jaune-vert avec boxplot de confiance
    └── target_cards.py     # Affichage des valeurs cibles prédites
```

## Design UI (voir image wireframe jointe)

### Style visuel
- **Médical clair** : fond blanc, couleurs douces, typographie propre
- Style clinique professionnel, épuré
- Utiliser `st.set_page_config(layout="wide")` pour exploiter toute la largeur

### Layout de haut en bas

#### 1. Header
- **Titre** : "Digital Twin — Stroke Rehabilitation Prototype"
- **Sous-titre** : Sélecteur d'horizon de prédiction avec 3 options :
  - `T0 — Rehab Entry` (valeurs à l'admission)
  - `T1 — Day 7-14` (réévaluation précoce)
  - `OUTCOME — Discharge` (prédiction à la sortie)
- Le sélecteur doit être un `st.radio` horizontal ou un `st.selectbox` bien visible

#### 2. Section des 20 Paramètres (Features)
- **Titre de section** : "Patient Parameters (Top 20 Features by Importance)"
- Disposer les 20 sliders sur **4 rangées de 5 colonnes** (`st.columns(5)`)
- Chaque slider est **VERTICAL** (de haut = max vers bas = min), exactement comme sur le wireframe
- Le nom du paramètre est affiché **en biais/diagonal** au-dessus du slider (utiliser du CSS custom pour la rotation du texte à ~45°)
- Pour les paramètres binaires/catégoriels, utiliser un slider discret ou un `st.selectbox`

**Liste des 20 features avec types et plages :**

| # | Feature | Type | Plage / Options | Valeur par défaut |
|---|---------|------|-----------------|-------------------|
| 1 | FIM Cognitive Subscale | Numérique | 5 – 35 | 22 |
| 2 | FIM Motor Subscale | Numérique | 13 – 91 | 45 |
| 3 | FIM Total Score | Numérique | 18 – 126 | 67 |
| 4 | NIHSS Total Score | Numérique | 0 – 42 | 12 |
| 5 | Walking Distance (m) | Numérique | 0 – 500 | 50 |
| 6 | Stroke Laterality | Catégoriel | Left / Right / Bilateral | Left |
| 7 | Stroke Type | Catégoriel | Ischemic / Hemorrhagic | Ischemic |
| 8 | Thrombectomy | Binaire | Yes / No | No |
| 9 | Neglect Severity | Ordinal | 0 (none) – 3 (severe) | 1 |
| 10 | Aphasia Severity | Ordinal | 0 (none) – 3 (severe) | 1 |
| 11 | Arm Hemiparesis | Ordinal | 0 (none) – 4 (complete) | 2 |
| 12 | Leg Hemiparesis | Ordinal | 0 (none) – 4 (complete) | 2 |
| 13 | Walking Ability | Ordinal | 0 (unable) – 4 (independent) | 1 |
| 14 | Continence Status | Ordinal | 0 (incontinent) – 2 (continent) | 1 |
| 15 | Personal Hygiene | Ordinal | 0 (dependent) – 3 (independent) | 1 |
| 16 | CIRS Total Score | Numérique | 0 – 56 | 8 |
| 17 | Primary Diagnosis (ICD) | Catégoriel | I63.0 / I63.1 / I63.3 / I63.4 / I63.5 / I61.0 / I61.1 | I63.3 |
| 18 | Pre-stroke Walking (m) | Numérique | 0 – 1000 | 500 |
| 19 | Stair Climbing | Ordinal | 0 (unable) – 3 (independent) | 1 |
| 20 | Orientation (time/place/person) | Ordinal | 0 (disoriented) – 3 (fully oriented) | 3 |

#### 3. Section Prédiction du Health Status
- **Titre** : "Prediction of Health Status"
- **Jauge horizontale** en dégradé continu : Rouge (mauvais) → Jaune (modéré) → Vert (bon)
- La jauge représente un **score composite de 0 à 100** :
  - 0-33 : zone rouge (détérioration probable)
  - 34-66 : zone jaune (stable/incertain)
  - 67-100 : zone vert (amélioration probable)
- **Sur la jauge** : afficher un **boxplot superposé** (comme sur le wireframe) qui montre :
  - La **valeur prédite** (trait central épais)
  - L'**intervalle de confiance** (la boîte = Q1-Q3 des patients similaires)
  - Les **whiskers** (min-max des patients similaires)
- Afficher le score numérique à côté : ex. "Score: 72/100 — Confidence: ±8"

#### 4. Section Prédiction des Valeurs Cibles
- **Titre** : "Prediction of Target Values"
- Afficher **4 cartes** côte à côte (utiliser `st.columns(4)` avec `st.metric`) :

| Target | Type | Unité/Plage |
|--------|------|-------------|
| FIM Total at Discharge | Numérique | 18 – 126 |
| FIM Gain (discharge − entry) | Numérique | -20 – 80 |
| Meaningful Improvement (FIM gain ≥ 15) | Binaire | Oui/Non + probabilité % |
| Return Home | Binaire | Oui/Non + probabilité % |

- Pour chaque target, afficher : la valeur prédite, et un delta ou pourcentage de confiance
- Pour les targets binaires, afficher la probabilité en % et un indicateur visuel (vert si >60%, jaune si 40-60%, rouge si <40%)

## Logique de prédiction (prototype)

### Génération de la cohorte simulée (`mock_patients.py`)
- Générer **200 patients simulés** avec des valeurs réalistes et corrélées :
  - Les patients avec un NIHSS élevé ont tendance à avoir des FIM plus bas, plus de déficits moteurs, etc.
  - Les patients avec thrombectomie ont des profils cohérents (AVC ischémique, NIHSS élevé)
  - Le FIM total doit = FIM cognitif + FIM moteur
  - Introduire de la variabilité réaliste
- Pour chaque patient simulé, générer aussi les **outcomes** (les 4 targets) de manière cohérente :
  - Plus le FIM d'entrée est haut et le NIHSS bas → meilleur FIM à la sortie
  - FIM gain = FIM sortie − FIM entrée
  - Meaningful improvement = 1 si FIM gain ≥ 15
  - Return home corrélé au FIM de sortie et à la distance de marche

### Moteur de prédiction (`predictor.py`)
- Utiliser un **k-NN pondéré** (k=20) sur la cohorte simulée :
  1. Normaliser toutes les features (min-max scaling)
  2. Calculer la distance euclidienne entre le patient courant (valeurs des sliders) et chaque patient de la cohorte
  3. Sélectionner les k=20 plus proches voisins
  4. Pondérer par l'inverse de la distance
  5. Prédire chaque target comme la moyenne pondérée des outcomes des voisins

### Score de Health Status (`scoring.py`)
- Calculer un **score composite de 0 à 100** basé sur les prédictions :
  - 40% × (FIM total at discharge normalisé sur 0-100)
  - 25% × (FIM gain normalisé sur 0-100)
  - 20% × (probabilité de meaningful improvement × 100)
  - 15% × (probabilité de return home × 100)
- Pour le **boxplot de confiance** :
  - Calculer Q1, médiane, Q3, min, max du score composite parmi les k voisins
  - Cela donne visuellement l'incertitude de la prédiction

### Réactivité
- Quand l'utilisateur bouge un slider, **toutes les prédictions se recalculent en temps réel**
- Utiliser `st.session_state` pour gérer les états
- Le changement d'horizon (T0/T1/OUTCOME) doit aussi mettre à jour les prédictions (simuler des différences : plus l'horizon est tardif, plus la prédiction est précise → boxplot plus resserré)

## Contraintes techniques importantes

1. **Les sliders doivent paraître verticaux** comme sur le wireframe. Si Streamlit ne supporte pas nativement les sliders verticaux, utiliser du CSS custom injecté via `st.markdown(unsafe_allow_html=True)` pour faire une rotation à 90° des sliders, OU utiliser une autre approche créative (composant HTML custom, colonnes étroites avec slider inversé, etc.)
2. **Les noms des features en diagonal** au-dessus des sliders : utiliser du CSS avec `transform: rotate(-45deg)`
3. **La jauge rouge-jaune-vert** doit être un vrai dégradé continu (pas 3 blocs), réalisé avec Plotly (`plotly.graph_objects.Indicator` ou un bar chart custom)
4. **Le boxplot superposé sur la jauge** : utiliser Plotly pour superposer un boxplot horizontal sur la jauge colorée
5. Le tout doit être **responsive** et fonctionnel
6. Seed le random pour des résultats reproductibles (`np.random.seed(42)`)

## Ce que tu dois produire

1. Tous les fichiers Python listés dans l'architecture
2. Le `pyproject.toml` avec toutes les dépendances
3. Un fichier `README.md` minimal avec les instructions pour lancer le prototype
4. L'app doit fonctionner directement avec `uv run streamlit run app.py`

## Résumé visuel (rappel)

```
┌─────────────────────────────────────────────────────────────┐
│           Digital Twin — Stroke Rehabilitation Prototype      │
│           [T0 — Rehab Entry ▾] [T1 — Day 7-14] [OUTCOME]    │
├─────────────────────────────────────────────────────────────┤
│  Patient Parameters (Top 20 Features by Importance)          │
│                                                              │
│  ╱FIM   ╱FIM   ╱FIM   ╱NIHSS  ╱Walk                        │
│  ╱Cogn  ╱Motor ╱Total ╱Total  ╱Dist   ...  (10 per row)     │
│  ┃ ■    ┃  ■   ┃ ■    ┃■     ┃  ■                          │
│  ┃      ┃      ┃      ┃      ┃        × 2 rows = 20 total  │
│  ┃      ┃      ┃      ┃      ┃                              │
├─────────────────────────────────────────────────────────────┤
│  Prediction of Health Status                                 │
│  ██████████████████████████████████████████████████████████  │
│  RED ■■■■■■■ YELLOW ■■■■■■■ GREEN ■■■■■■■                  │
│              ├──[  ┃██┃  ]──┤  ← boxplot overlay            │
│                    ↑ predicted value     Score: 72 ± 8       │
├─────────────────────────────────────────────────────────────┤
│  Prediction of Target Values                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │FIM Disch.│ │FIM Gain  │ │Meaningful│ │Return    │       │
│  │  94      │ │  +27     │ │Improve.  │ │Home      │       │
│  │  ▲ +27   │ │  ▲ good  │ │  78%  ●  │ │  65%  ●  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```
