# Bank ML Simulator

Application universitaire de classification de la réponse `y` du dataset Bank Marketing. Elle entraîne et compare trois pipelines scikit-learn (`SVM`, `RandomForestClassifier` et `MLPClassifier`), puis expose leurs prédictions dans une interface web FastAPI.

## Préparation

Depuis le dossier du projet, sous PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sous macOS/Linux, l'activation équivalente est `source .venv/bin/activate`.

## Entraîner les modèles

```powershell
python train_models.py
```

Cette commande lit `bank.csv` (séparateur `;`), applique un unique split stratifié 80/20 (`random_state=42`), entraîne les trois pipelines et produit dans `models/` :

- `svm.pkl`, `random_forest.pkl`, `mlp.pkl` ;
- `metrics.json` avec accuracy, précision, rappel, F1-score et matrice de confusion ;
- `schema.json` utilisé pour générer automatiquement le formulaire.

Chaque pipeline contient son propre prétraitement : `StandardScaler` pour les variables numériques et `OrdinalEncoder` pour les variables catégorielles. Aucun `OneHotEncoder` n'est utilisé.

## Lancer l'application

```powershell
python -m uvicorn backend.main:app --reload
```

Ouvrir ensuite <http://127.0.0.1:8000>. La documentation interactive de l'API est disponible sur <http://127.0.0.1:8000/docs>.

## API

`POST /predict` attend les 16 caractéristiques de `bank.csv` et `algorithm`, qui vaut `svm`, `random_forest` ou `mlp`. Exemple :

```json
{
  "age": 30,
  "job": "unemployed",
  "marital": "married",
  "education": "primary",
  "default": "no",
  "balance": 1787,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 19,
  "month": "oct",
  "duration": 79,
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown",
  "algorithm": "random_forest"
}
```

L'accuracy renvoyée est toujours l'accuracy réelle du jeu de test enregistrée dans `models/metrics.json`.
