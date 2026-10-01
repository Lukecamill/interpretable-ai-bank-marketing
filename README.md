# Interpretable AI — Bank Marketing Analysis

An end-to-end interpretable machine learning project exploring how different explainability techniques can be used to understand model predictions on the **Bank Marketing dataset**.

The project moves from inherently interpretable white-box models to more complex ensemble models, applying both **global and local explanation techniques** including permutation importance, Partial Dependence Plots, ALE, surrogate models, LIME and SHAP.

## Overview

Machine learning models can achieve strong predictive performance while making it difficult to understand why a particular prediction was made.

This project explores different approaches to model interpretability by comparing explanations produced by multiple techniques on the same classification problem.

The analysis follows a progression from simple interpretable models to more complex explainability methods:

```text
Data Preprocessing
        ↓
White-Box Models
        ↓
Random Forest
        ↓
Feature Importance
        ↓
Permutation Importance
        ↓
PDP & ALE
        ↓
Global Surrogate Model
        ↓
LIME & SHAP
```

The objective is not only to generate predictions, but to understand the features and relationships driving those predictions.

## Dataset

The project uses the **Bank Marketing dataset**, containing data from a Portuguese bank's direct marketing campaigns.

The classification task is to predict whether a client will subscribe to a term deposit.

The dataset contains:

- **45,211 client records**
- **16 original explanatory features**
- Numerical and categorical variables
- Binary target variable representing subscription
- Approximately **11.7% positive-class observations**

Categorical variables are transformed using one-hot encoding.

### Preventing Data Leakage

The `duration` feature represents the duration of the marketing call.

Because this information is only known **after the call has taken place**, using it to predict whether a client will subscribe would introduce future information into the model.

For this reason, `duration` is excluded from the classification feature set.

## Models

### Linear Regression

Linear regression is used as a transparent white-box model for analysing relationships between client characteristics and call duration.

Because the model coefficients are directly interpretable, they provide an initial example of global model interpretation.

### Logistic Regression

Logistic regression is used for the subscription classification problem.

Features are standardised to improve convergence and make the magnitude of coefficients more comparable.

The coefficients provide a direct interpretation of how individual features influence the model's prediction.

### Decision Tree

A shallow Decision Tree provides a fully transparent rule-based classifier.

Unlike more complex ensemble models, its decision process can be followed directly from the root node to each prediction.

### Random Forest

A Random Forest consisting of **200 trees** is used as the primary black-box model for the interpretability experiments.

Its increased complexity makes it a useful candidate for evaluating different model explanation techniques.

## Interpretability Techniques

### Random Forest Feature Importance

Gini-based feature importance measures how much each feature contributes to reducing impurity across the trees in the Random Forest.

However, impurity-based importance can favour continuous variables because they provide more possible split points.

This motivates the use of additional interpretability techniques.

### Permutation Feature Importance

Permutation importance provides a model-agnostic alternative.

Each feature is randomly shuffled and the resulting decrease in model performance is measured.

A larger performance decrease indicates that the model relies more heavily on that feature.

The analysis demonstrates how permutation importance can produce substantially different rankings from Gini-based Random Forest importance.

### Partial Dependence Plots (PDP)

Partial Dependence Plots visualise how changing a feature affects the model's average prediction.

They provide a useful global view of relationships learned by the model.

A limitation of PDP is its assumption of feature independence, which can result in unrealistic feature combinations when correlated variables are present.

### Accumulated Local Effects (ALE)

Accumulated Local Effects provide an alternative to PDP that evaluates feature effects within the observed data distribution.

This makes ALE particularly useful when features are correlated.

The analysis uses ALE to reveal non-linear relationships that are less apparent using standard feature-importance measures.

### Global Surrogate Model

A shallow Decision Tree is trained to approximate the predictions of the Random Forest.

This creates an interpretable approximation of the more complex ensemble model.

The surrogate achieved an **R² of approximately 0.55**, capturing a substantial portion of the Random Forest's global behaviour.

### LIME

**LIME (Local Interpretable Model-Agnostic Explanations)** is used to explain individual predictions.

Rather than attempting to explain the entire Random Forest, LIME creates a simpler local model around a particular observation.

This allows the features pushing an individual prediction toward or away from subscription to be examined.

### SHAP

**SHAP (SHapley Additive exPlanations)** uses concepts from cooperative game theory to attribute a model prediction across its input features.

SHAP provides a model-faithful way of understanding the contribution of each feature to individual predictions.

The project compares SHAP with LIME to demonstrate how different local explanation techniques can provide complementary views of the same model.

## Key Findings

Several interpretability methods consistently highlighted the importance of a client's previous interactions with the bank.

In particular, **previous campaign success (`poutcome_success`)** emerged as an important predictor across multiple explanation techniques.

The analysis also demonstrates that:

- Gini feature importance and permutation importance can produce substantially different rankings.
- Global and local interpretability techniques answer different questions about model behaviour.
- PDP and ALE can reveal non-linear relationships that simple feature rankings cannot.
- LIME provides intuitive local explanations around individual observations.
- SHAP distributes feature contributions using the structure of the trained model.
- Combining multiple explanation techniques provides a more complete understanding than relying on a single method.

## Project Structure

```text
.
├── README.md
├── Luke_Camilleri.ipynb
├── Luke_Camilleri.pdf
└── form.png
```

### `Luke_Camilleri.ipynb`

Main Jupyter Notebook containing the complete machine learning and interpretability pipeline.

### `Luke_Camilleri.pdf`

PDF version of the project analysis and results.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Random Forests
- LIME
- SHAP
- Partial Dependence Plots
- Accumulated Local Effects
- Machine Learning Interpretability

## Key Concepts Demonstrated

- Explainable Artificial Intelligence (XAI)
- Interpretable Machine Learning
- Feature Importance
- Model-Agnostic Explanations
- Global vs Local Interpretability
- Data Leakage Prevention
- White-Box vs Black-Box Models
- Feature Engineering
- Classification
- Model Evaluation

## Author

**Luke Camilleri**

B.Sc. Artificial Intelligence  
University of Malta
