# Heart Attack Machine Learning Project 

This project explores multiple machine learning models to predict the risk of heart attack based on health-related data. It implements and compares model performances through a complete data science pipeline, including data preprocessing, model training, hyperparameter tuning, and evaluation. Key scripts include implementations for various models, cross-validation, and a main execution file that streamlines the training and evaluation process. 

## Files

- **`implementations.py`**: Contains implementations of six machine learning models:
  - Gradient Descent Linear Regression
  - Stochastic Gradient Descent Linear Regression
  - Least Squares Regression
  - Ridge Regression
  - Logistic Regression
  - Regularized Logistic Regression

  Each model returns the optimized weights and final loss, with gradients and losses calculated using helper functions from `utilities.py`.

- **`utilities.py`**: Provides helper functions essential for:
  - **Model Computations**: Loss calculations, gradient calculations, prediction functions, and score calculations.
  - **Data Processing**: Common preprocessing tasks such as filling missing values, feature scaling, upsampling, removing low-variance features, polynomial feature generation, and more. These transformations help ensure the data is prepared before model training.

- **`cross_validation.py`**: Implements cross-validation for model evaluation, helping to select the best hyperparameters for each model. This file includes:
  - `cross_validate_ridge` and `cross_validate_logistic` for ridge regression and logistic regression.
  - **Threshold Optimization**: Helps find the best decision threshold for F1 score maximization in ridge regression.
  - **Hyperparameter Tuning**: Finds optimal combinations of hyperparameters such as lambda (regularization strength) and gamma (learning rate).

- **`run.py`**: The main execution script that coordinates the training and prediction workflow. Key functions include:
  - **Data Loading and Caching**: Loads data and caches it to avoid reloading large files.
  - **Preprocessing Pipeline**: Cleans and preprocesses the data using utilities in `utilities.py`.
  - **Model Selection and Training**: Executes specified models, finds the best hyperparameters through cross-validation, and trains the final model.
  - **Prediction and Submission**: Generates predictions on the test set and saves them to a CSV file for submission.

 
## Project Structure

* **Data Preprocessing**: This includes handling missing values by dropping columns with a high ratio of NaNs, filling in missing values, engineering features through polynomial expansion, removing low-variance features, normalizing features, adding a bias term, and balancing classes with upsampling.
* **Model Training and Evaluation**: Implements regularized logistic regression and ridge regression, using cross-validation to optimize hyperparameters such as lambda, gamma, and decision thresholds for improved prediction accuracy.
* **Performance Metrics**: Models are evaluated and compared using metrics such as accuracy and F1-score to gauge overall performance.



## Usage
**Setup**: Ensure that all the dependencies have been installed:
  * numpy
  * matplotlib
  * os
  * pickle

**Data**: Ensure all data files are located within a folder called "dataset" in the same directory where the script is executed. These data files are:
  * x_test.csv 
  * x_train.csv
  * y_train.csv

**Execution**: Run the main script `run.py` to start the training. To choose between Ridge Regression or Regularized Logistic Regression, modify the variable `method_to_run` with either `ridge_regression` or `logistic_regression`

**Output**: Predictions for the test dataset are saved to a CSV file for submission.



## How It Works

The pipeline in run.py reads the data from the dataset folder. It then preprocesses the data using functions in utilities.py, trains models from implementations.py (which themselves also use helper functions from utilities.py), and optimizes hyperparameters based on cross-validation results from cross_validation.py. The best model is then produces predictions for our test dataset to evalute online.


## Contributors

- Bakiri Ayman
- Ben Mohamed Nizar
- Chahed Ouazzani Adam
