# 🚢 Titanic Survival Prediction — Advanced ML Pipeline

End-to-end machine learning project for predicting Titanic passenger survival using advanced feature engineering, hyperparameter tuning, and ensemble methods.

---

## 📌 Overview

A production-ready ML pipeline that goes beyond the classic Titanic tutorial with:

- ✅ **15+ engineered features** (titles, family size, deck, ticket frequency, etc.)
- ✅ **Smart missing-value imputation** (group-based medians)
- ✅ **8 ML models** + Voting Ensemble
- ✅ **Stratified K-Fold CV** & **GridSearchCV**
- ✅ **10+ publication-quality visualizations**

---

## 📊 Results

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| **Gradient Boosting (Tuned)** | **84.7%** | **0.904** | **0.806** |
| Voting Ensemble | 84.7% | 0.899 | 0.806 |
| LightGBM (Tuned) | 82.8% | 0.891 | 0.794 |
| AdaBoost (Tuned) | 82.2% | 0.883 | 0.797 |

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, matplotlib, seaborn
- **Scikit-learn** — preprocessing, modeling, evaluation, tuning
- **XGBoost** & **LightGBM** — gradient boosting frameworks

---

## 🚀 Features Engineered

| Feature | Description |
|---------|-------------|
| `Title` | Extracted from name (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | FamilySize == 1 |
| `Deck` | Extracted from Cabin number |
| `AgeBin` | Child, Teen, Adult, Middle, Senior |
| `FareBin` | Low, Med, High, Premium |
| `Age_Fare` | Interaction feature |
| `TicketFreq` | Frequency of ticket number |

---

## 📈 Visualizations

- Missing Values Heatmap
- Survival by Sex, Pclass, Embarked
- Age & Fare Distributions
- Feature Correlation Heatmap
- Model Performance Comparison
- ROC Curves & Confusion Matrices

---

## 🏆 Best Model

**Gradient Boosting (Tuned)**

```python
Best Params: {
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 200,
    'subsample': 1.0
}
