**Regression Analysis on Iris Dataset with Feature Engineering and Model Tuning**
This project performs regression analysis on the Iris dataset (iris.csv) using various machine learning models. The objective is to predict the **petal_width** based on other features after applying feature engineering techniques and optimizing model hyperparameters.

**Table of Contents**

**Project Overview**
This project aims to build and evaluate regression models to predict `petal_width` using the Iris dataset. It includes preprocessing steps, feature engineering techniques, model selection, hyperparameter tuning, and evaluation based on regression-specific metrics.

**Dataset**
The dataset used in this project is iris.csv, which is a modified version of the classic Iris dataset. It contains the following features:

`sepal_length`: Length of the sepal (numerical)

`sepal_width`: Width of the sepal (numerical)

`petal_length`: Length of the petal (numerical)

`petal_width`: Width of the petal (target variable, numerical)

`species`: Species of the Iris flower (categorical)

**Feature Engineering**

**Missing Values Handling**

**Imputation**: Missing values in `sepal_width` and `petal_width` were imputed using custom methods.

**Feature Interaction**

**Linear Interactions**: Created interactions between petal_length and sepal_width.

**Polynomial Interactions**: Generated polynomial features for petal_length/sepal_width and petal_width/species.

**Feature Selection**

**Selected features**: sepal_length, sepal_width, petal_length, petal_width, species.

**Modeling Approach**

**Regression Models**

**Random Forest Regressor**: Implemented with hyperparameter tuning.

**Gradient Boosted Trees:** Considered for potential implementation.

**Decision Tree Regressor**: Evaluated as a baseline model.

**Hyperparameter Optimization**

**Grid Search**: Used to optimize parameters such as max_depth, min_samples_leaf, and n_estimators for each model.

**Evaluation Metrics**

**Models are evaluated based on:**

**R-squared (R^2):** Measure of how well the model captures the variance in the target variable.

**Mean Squared Error (MSE)**: Indicates the average squared difference between actual and predicted values.

**Other Regression Metrics:** Considered for specific model requirements.

**Results and Insights**

**Best Model:** Identified based on evaluation metrics (e.g., R^2 score).

### Output
Here is the output of the best model:

**Best Parameters for RandomForestRegressor**
max_depth: 20
min_samples_leaf: 10
n_estimators: 10
**Best Score for RandomForestRegressor**
R-squared (R^2): 0.9217

**Insights**: Interpretations and findings from the analysis.

**Recommendations**: Suggestions for improving model performance or exploring alternative approaches.
