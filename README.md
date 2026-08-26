# 🎓 Prédiction de la Réussite Étudiante — Machine Learning

> Un projet de Data Science & Intelligence Artificielle visant à prédire, à partir du comportement et de l'assiduité d'un étudiant, ses chances de réussir son examen final.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Statut-Terminé-success)

---

## 📌 Contexte

Une école souhaite utiliser l'Intelligence Artificielle pour identifier, en amont, les étudiants ayant de fortes chances de **réussir** ou d'**échouer** à leur examen final. Ce projet exploite un historique de **300 étudiants** (assiduité, comportement, résultats) pour entraîner un modèle capable de répondre à la question :

> *« Cet étudiant a-t-il de fortes chances de réussir son examen final ? »*

La variable cible est `reussite` (`0` = échec, `1` = réussite).

---

## 🗂️ Jeu de données

| Variable | Description |
|---|---|
| `heures_etude_semaine` | Nombre d'heures d'étude par semaine |
| `taux_presence_pct` | Taux de présence aux cours (%) |
| `exercices_realises` | Nombre d'exercices réalisés |
| `note_controle_continu` | Note obtenue au contrôle continu (/20) |
| `temps_plateforme_h_semaine` | Temps passé sur la plateforme pédagogique (h/semaine) |
| `devoirs_rendus` | Nombre de devoirs rendus |
| `participation_classe_10` | Niveau de participation en classe (/10) |
| `reussite` | **Variable cible** : 0 = échec, 1 = réussite |

---

## 🔬 Démarche du projet

**1. Exploration des données** — chargement du CSV, aperçu des lignes, dimensions du dataset, typage des variables, recherche de valeurs manquantes, statistiques descriptives, répartition réussite/échec.

**2. Analyse des données** — visualisations univariées et bivariées (boxplots, histogrammes) pour identifier les variables les plus corrélées à la réussite (heatmap, pairplot).

**3. Préparation des données** — traitement des valeurs manquantes, séparation `X` / `y`, split entraînement/test (80/20).

**4. Modélisation** — entraînement d'une **régression logistique** comme modèle de référence.

**5. Évaluation** — Accuracy, Precision, Recall, F1-score et matrice de confusion, pour juger si le modèle est fiable pour repérer les étudiants à risque.

**6. Bonus — Random Forest** — second modèle entraîné et comparé à la régression logistique.

**7. Prédiction sur un nouvel étudiant** — fonction retournant une probabilité de réussite et une prédiction lisible (ex. *Probabilité de réussite : 87,4 % — Prédiction : RÉUSSITE PROBABLE*).

---

## 📁 Structure du dépôt

```
📦 TP_Prediction_Reussite_Etudiants_IA
├── 📂 Rapport
│   └── Conclusion_projet2.pdf          # Synthèse des résultats et limites du modèle
├── 📂 data
│   └── dataset_reussite_etudiants_300...csv
├── 📂 image
│   ├── 📂 UNIVARIE
│   │   ├── boxplot.png
│   │   └── histplot.png
│   ├── 📂 BIVARIE
│   │   ├── boxplot.png
│   │   └── histplot.png
│   ├── 📂 CORRELATION
│   │   ├── heatmap.png
│   │   └── pairplot.png
│   └── 📂 evaluation
│       ├── lregression.png
│       └── rforest.png
├── 📂 notebook
│   └── model.ipynb                     # Notebook principal (code, graphiques, résultats)
├── .gitignore
└── TP_Prediction_Reussite_Etudiants_IA.pdf   # Énoncé du TP
```

---

## ⚙️ Installation & utilisation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/TP_Prediction_Reussite_Etudiants_IA.git
cd TP_Prediction_Reussite_Etudiants_IA

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate   # ou venv\Scripts\activate sous Windows

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Lancer le notebook
jupyter notebook notebook/model.ipynb
```

---

## 🧠 Exemple d'utilisation du modèle

```python
etudiant = {
    "heures_etude_semaine": 12,
    "taux_presence_pct": 91,
    "exercices_realises": 25,
    "note_controle_continu": 15,
    "temps_plateforme_h_semaine": 8,
    "devoirs_rendus": 9,
    "participation_classe_10": 8
}

# → Probabilité de réussite : 87,4 % — Prédiction : RÉUSSITE PROBABLE
```

---

## 📊 Résultats

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Régression Logistique | 0.7500 | 0.7500 | 0.8182 | 0.7826 |
| Forêt Aléatoire (Random Forest) | 0.7167 | 0.7500 | 0.7273 | 0.7385 |

La **Régression Logistique** l'emporte sur 3 des 4 métriques (Accuracy, Recall, F1-score), avec une égalité en Precision. C'est donc le modèle retenu, notamment pour son meilleur **Recall** (81,8 %) — un point important puisqu'il s'agit ici de repérer un maximum d'étudiants réellement à risque d'échec.

> ℹ️ Ces résultats sont obtenus **sans feature engineering** ni **optimisation des hyperparamètres** (paramètres par défaut de Scikit-learn). Une phase de tuning (`GridSearchCV`, création de nouvelles variables, etc.) pourrait améliorer les performances, en particulier celles de la Forêt Aléatoire.

👉 Détails, interprétation et choix final du modèle dans `Rapport/Conclusion_projet2.pdf`.

---

## ⚠️ Limites du système

Une prédiction du type *« 35 % de chances de réussite »* ne doit **jamais** guider seule une décision d'accompagnement. Parmi les limites identifiées :

- Le modèle capture des **corrélations statistiques**, pas des causes réelles d'échec (facteurs personnels, sociaux ou de santé non mesurés).
- Un biais dans les données d'entraînement (échantillon de 300 étudiants) peut conduire à des prédictions peu fiables sur des profils atypiques.

---

## 🛠️ Technologies utilisées

`Python` · `Pandas` · `NumPy` · `Matplotlib` / `Seaborn` · `Scikit-learn` · `Jupyter Notebook`

---

## ✍️ Auteur

Projet réalisé dans le cadre d'un TP Data & Intelligence Artificielle — **Orange Digital Center**, Abidjan.
