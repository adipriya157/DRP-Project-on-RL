# Project README
## Overview
This project contains all of the work, analysis, and exploratory modeling performed on two different datasets: one involving a single fish dataset and another involving a three-fish dataset. The purpose of the project is to walk through a complete data science workflow in a way that is understandable to another human, even if they were not involved in the original work. Each notebook is written as a step-by-step narrative that starts with getting familiar with the data and ends with building, evaluating, and interpreting predictive models. The idea is to make the project readable, replicable, and easy to reference later.
## Project Files
### DRP_Final_Project_3Fish_V2.ipynb
This notebook focuses on the dataset containing three fish. It includes an explanation of the raw data, cleaning steps, decisions made during preprocessing, visual exploration of trends or patterns, and the full modeling pipeline. The notebook highlights why certain features were created or removed, what model types were attempted, how performance was measured, and what the results ultimately mean. It is meant to be a polished, narrative walkthrough rather than a messy scratchpad.
### DROP_Final_Project_1Fish.ipynb
This notebook performs a similar analysis but for a single-fish dataset. Because the dataset is simpler, the emphasis is slightly different: more attention is given to understanding how the more limited feature space affects the modeling process. The notebook includes data preparation, visualizations, model experiments, and a discussion comparing the strengths and weaknesses of the one-fish dataset relative to the three-fish version.
## Requirements
To run the notebooks successfully, you will need Python 3.8 or higher along with standard data science libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn. If a requirements.txt file exists, installing everything is as simple as running:

## Mathematical Formulation and Algorithms

This section explains, in plain language, what the model is doing under the hood: how the data is represented mathematically, what the learning algorithm is optimizing, and how we evaluate the results.

### Problem Setup

We assume we have a dataset with \(n\) observations (rows) and \(d\) features (columns). Each observation corresponds to one fish (or one record about a fish), with measurements such as length, height, width, and possibly other variables.

- Let \(X \in \mathbb{R}^{n \times d}\) denote the feature matrix.
  - Each row \(x_i \in \mathbb{R}^d\) is one fish.
  - Each column is a specific feature (e.g., length, height, width, etc.).
- Let \(y \in \mathbb{R}^n\) denote the target variable we want to predict.
  - For example, \(y_i\) might be the weight of fish \(i\).

The goal of supervised learning here is to learn a function \(f\) such that:

\[
\hat{y}_i = f(x_i)
\]

is as close as possible to the true value \(y_i\).

### Notation

- \(x_i = (x_{i1}, x_{i2}, \dots, x_{id})\) are the features for observation \(i\).
- \(y_i\) is the true target value for observation \(i\).
- \(\hat{y}_i\) is the model’s prediction for observation \(i\).
- \(\theta\) represents all of the model parameters that we are trying to learn (for example, the coefficients in a linear model or the split rules in a tree-based model).

### Linear Regression Model (if used)

A common baseline model is **linear regression**, which assumes the target can be approximated as a linear combination of the input features:

\[
\hat{y}_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_d x_{id}
\]

or in vector form:

\[
\hat{y}_i = \beta_0 + x_i^\top \beta
\]

where:
- \(\beta_0\) is the intercept term.
- \(\beta \in \mathbb{R}^d\) is the vector of feature coefficients.

**Interpretation:**  
Each coefficient \(\beta_j\) represents how much the prediction changes on average when feature \(j\) increases by one unit, holding all other features constant.

### Loss Function (What the Model Minimizes)

To find good parameters \(\beta_0, \beta\), the model minimizes a loss function over the training data. For regression, the most common choice is the **Mean Squared Error (MSE)**:

\[
\text{MSE}(\beta_0, \beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

The learning algorithm tries to find \(\beta_0, \beta\) that make the predictions \(\hat{y}_i\) as close as possible to \(y_i\) in the least-squares sense.

### Regularization (if used)

If the model uses regularization (like Ridge or Lasso), a penalty term is added to avoid overfitting:

- **Ridge (L2) regression:**

\[
\text{Loss} = \text{MSE} + \lambda \sum_{j=1}^{d} \beta_j^2
\]

- **Lasso (L1) regression:**

\[
\text{Loss} = \text{MSE} + \lambda \sum_{j=1}^{d} |\beta_j|
\]

Here, \(\lambda \geq 0\) controls the strength of the penalty:
- Higher \(\lambda\) shrinks coefficients more aggressively.
- This can help the model generalize better to new data.

### Tree-Based Models (if used)

If tree-based models like **Decision Trees**, **Random Forests**, or **Gradient Boosted Trees** are used, the underlying idea is different from linear models:

- The feature space is recursively split into regions based on rules like:
  - “If length < 20 cm, go left; else go right.”
- Each final region (leaf) predicts a constant value, usually the average target value of the training points in that region.
- Random Forests build many such trees on different subsamples and average their predictions.
- Gradient Boosting builds trees sequentially, each new tree trying to correct the errors of the previous ones.

Mathematically, a tree-based model can be thought of as:

\[
\hat{y}_i = \sum_{m=1}^{M} f_m(x_i)
\]

where each \(f_m\) is a decision tree and \(M\) is the number of trees.

### Training Procedure

1. **Split the data** into training and testing sets (and sometimes validation sets or use cross-validation).
2. **Fit the model** on the training set:
   - For linear models, this involves solving an optimization problem to minimize the loss function.
   - For tree-based models, this involves building trees by choosing splits that reduce prediction error the most.
3. **Tune hyperparameters** (if applicable):
   - For example, learning rate, number of trees, tree depth, or regularization strength.
   - This is often done using grid search or randomized search with cross-validation.
4. **Evaluate the final model** on the test set to estimate performance on unseen data.

### Evaluation Metrics

For regression tasks (predicting continuous values like weight), typical metrics include:

- **Mean Squared Error (MSE):**

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- **Root Mean Squared Error (RMSE):**

\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

- **Mean Absolute Error (MAE):**

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

- **Coefficient of Determination (\(R^2\)):**

\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\]

where \(\bar{y}\) is the mean of the true target values.  
\(R^2\) measures how much of the variance in the target is explained by the model (1 is perfect, 0 means no better than predicting the mean).

### Interpretation and Use

Once trained and evaluated, the model can be used to:

- Predict fish weight (or another target) for new measurements.
- Understand which features are most important (via coefficients in linear models or feature importance in tree-based models).
- Compare performance between the one-fish and three-fish setups to see how data richness affects predictive power.

The overall goal is not only to make accurate predictions but also to provide a clear, interpretable description of how the model works and how well it performs.
