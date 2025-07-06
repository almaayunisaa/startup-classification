# Success Startup Classification with Ensemble Stacking

This project aims to classify the success of startups using a stacked ensemble of powerful machine learning models. The base models include **CatBoost** and **XGBoost**, while the final meta-classifiers evaluated are **MLPClassifier** and **RandomForestClassifier**.

## 🔍 Dataset

Dataset: `startup data.csv` from Kaggle  
Source: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

## ⚙️ Model Architecture

The project implements **StackingClassifier** with two configurations:

1. **Stacking Model 1 (Meta: MLPClassifier)**
2. **Stacking Model 2 (Meta: RandomForestClassifier)**

Each base learner was tuned using **Optuna** for hyperparameter optimization.

## 📊 Evaluation Metrics

- Accuracy
- ROC AUC

## 🧠 Models Used

- `CatBoostClassifier`
- `XGBClassifier`
- `MLPClassifier` (as meta-learner 1)
- `RandomForestClassifier` (as meta-learner 2)

## 🚀 Usage

```python
from huggingface_hub import hf_hub_download
import pickle

# Load model from Hugging Face Hub
file_path = hf_hub_download(repo_id="almaayunisa/stacking-startup-success", filename="stacking_model_1.pkl")

# Load model with pickle
with open(file_path, "rb") as f:
    model = pickle.load(f)

# Make predictions Model 1 (ensure X_new is ready)
predictions_1 = model.predict(X_new)

# Load model from Hugging Face Hub
file_path = hf_hub_download(repo_id="almaayunisa/stacking-startup-success", filename="stacking_model_2.pkl")

# Load model dengan pickle
with open(file_path, "rb") as f:
    model = pickle.load(f)

# Make predictions Model 2 (ensure X_new is ready)
predictions_2 = model.predict(X_new)
