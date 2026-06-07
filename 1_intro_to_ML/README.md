# Machine Learning Regression Example with Pandas DataFrames

This project demonstrates how to build and evaluate regression models using Python, Pandas DataFrames, and Scikit-Learn.

The example uses a synthetic housing dataset to predict home prices based on several features:

* Square footage
* Number of bedrooms
* Property age
* Distance to the city center

## Project Structure

```text
.
├── README.md
└── machine_learning_example.py
```

## Overview

The script performs the following machine learning workflow:

1. Generate sample housing data
2. Load data into a Pandas DataFrame
3. Define input features and target variable
4. Split data into training and testing datasets
5. Train multiple regression models
6. Evaluate model performance
7. Compare predictions
8. Analyze feature importance

## Regression Models Included

### Linear Regression

A simple and interpretable model that assumes a linear relationship between input features and the target variable.

Advantages:

* Fast training
* Easy to interpret
* Good baseline model

### Random Forest Regression

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.

Advantages:

* Handles nonlinear relationships
* Captures feature interactions
* Often provides higher predictive performance

## Evaluation Metrics

The example evaluates model performance using the following metrics:

### Root Mean Squared Error (RMSE)

Measures the average magnitude of prediction errors.

Lower values indicate better model performance.

### R² Score

Measures the proportion of variance explained by the model.

Values closer to 1 indicate better performance.

## Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn
```

## Running the Example

Execute the script:

```bash
python machine_learning_example.py
```

## Expected Output

The script displays:

* Sample data from the DataFrame
* Model evaluation metrics
* Prediction comparisons
* Random Forest feature importance rankings

## Learning Objectives

This example demonstrates several core machine learning concepts:

* Working with Pandas DataFrames
* Feature selection
* Train/test splitting
* Supervised learning
* Regression modeling
* Model evaluation
* Feature importance analysis

## Possible Enhancements

You can extend this example by adding:

* Data preprocessing pipelines
* Feature scaling
* Cross-validation
* Hyperparameter tuning
* Categorical variables
* Gradient boosting models
* XGBoost regression
* Model persistence
* Data visualization
* Explainable AI techniques such as SHAP

## References

* Pandas for data manipulation and analysis
* NumPy for numerical computing
* Scikit-Learn for machine learning algorithms and evaluation metrics

This project is intended as a beginner-friendly introduction to regression modeling using modern Python machine learning tools.
