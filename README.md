# ml-predicting-ad-clicks

# Predicting Ad Clicks with Machine Learning

This project explores a variety of machine learning models to predict whether an online advertisement will be clicked. Starting with neural networks and progressing through tree-based models, I found that **XGBoost** delivered the most consistent and accurate results across validation and test sets.

## Problem Overview

**Objective:** Predict whether an online advertisement will be clicked using metadata from users and ads.

**Dataset:** [Outbrain Click Prediction](https://www.kaggle.com/competitions/outbrain-click-prediction) (Kaggle)  
- 100GB+ real-world ad interaction data  
- Clicks are rare (approx. 20% positive class)  
- Requires merging multiple files: page views, events, promoted content, etc.

## Data Processing

Handling such a massive and imbalanced dataset posed several challenges:
- Split 35GB+ CSVs into manageable chunks
- Merged files into a consistent training format
- **Undersampled** non-clicked rows to balance the dataset
- Preprocessed with:
  - One-hot encoding (categorical features)
  - StandardScaler (numerical features)

## Modeling Attempts

### Final Model: XGBoost
```python
{
  "colsample_bytree": 1.0,
  "gamma": 0.2,
  "learning_rate": 0.01,
  "max_depth": 7,
  "min_child_weight": 4,
  "n_estimators": 300,
  "subsample": 0.8
}
```
- Loss: Log Loss
- Evaluation: Accuracy, Precision, Recall, F1, AUC (with Cross-Validation)

| Metric      | Training | Validation | Test     |
|-------------|----------|------------|----------|
| Accuracy    | 0.7166   | 0.6197     | 0.6301   |
| Precision   | 0.7039   | 0.6233     | 0.6223   |
| Recall      | 0.7532   | 0.6500     | 0.6730   |
| F1 Score    | 0.7277   | 0.6364     | 0.6466   |
| AUC         | 0.7898   | 0.6825     | 0.6751   |
| Log Loss    | 0.5881   | 0.6382     | 0.6484   |

### Other Models Explored:
- **Neural Networks (Keras)**  
  High overfitting despite architectural tuning, dropout, and learning rate schedules  
- **Random Forest**  
  Strong recall, poor precision. Better than NN, worse than XGBoost.  
- **Gradient Boosting (sklearn)**  
  Improved over RF, but fell short of XGBoost.

## How to Use

This repo contains scripts and notebooks used throughout the modeling process. Due to the size of the original dataset, only preprocessed samples or chunked-loading scripts are referenced.

## Key Takeaways

- Tree-based models (especially XGBoost) outperform neural nets for sparse, tabular data
- Balancing the dataset with undersampling was critical for model reliability
- Cross-validation and randomized hyperparameter search led to significant performance gains
- Model generalization across validation and test sets confirms robustness

## Future Improvements

- Add threshold tuning for F1 optimization
- Engineer richer categorical features (e.g., session length, ad category)
- Explore LightGBM for faster training on large-scale data


