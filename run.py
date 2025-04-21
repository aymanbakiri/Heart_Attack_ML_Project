import numpy as np
import os
import pickle
from helpers import *
from implementations import *
from utilities import *
from cross_validation import *


def main():

    # Path to cache file
    cache_file = "cached_data.pkl"

    # Check if cache exists
    if os.path.exists(cache_file):
        # Load data from cache
        print("Loading data from cache...")
        with open(cache_file, "rb") as f:
            x_train, x_test, y_train, train_ids, test_ids = pickle.load(f)
    else:
        # Load data from CSV (expensive operation)
        print("Loading data from CSV...")
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
            "dataset", sub_sample=False
        )

        # Save the loaded data to cache
        with open(cache_file, "wb") as f:
            pickle.dump((x_train, x_test, y_train, train_ids, test_ids), f)

    print("Shape of x_test:", x_test.shape)
    print("Shape of x_train:", x_train.shape)
    print("Shape of y_train:", y_train.shape)
    # print number of 1 and -1 in y_train
    print("Number of 1 in y_train:", np.sum(y_train == 1))
    print("Number of -1 in y_train:", np.sum(y_train == -1))

    # Preprocess the data

    method_to_run = "ridge_regression"

    # Drop nan columns
    x_train_cleaned = drop_nan_columns(x_train, threshold=0.6)
    x_test_cleaned = drop_nan_columns(x_test, threshold=0.6)
    print("Shape of x_train_cleaned after dropping nan columns:", x_train_cleaned.shape)
    print("Shape of x_test_cleaned after dropping nan columns:", x_test_cleaned.shape)

    if method_to_run == "ridge_regression":
        # Build polynomial features
        x_train_cleaned = build_poly(x_train_cleaned, 3)
        x_test_cleaned = build_poly(x_test_cleaned, 3)

    # Handle missing values (fill with mean)
    x_train_cleaned = fill_missing_values(x_train_cleaned)
    x_test_cleaned = fill_missing_values(x_test_cleaned)

    # Remove features with low variance
    x_train_cleaned, low_variance_mask = remove_low_variance_features(
        x_train_cleaned, threshold=0.01
    )
    x_test_cleaned = x_test_cleaned[:, low_variance_mask]  # Use the same mask
    print(
        "Shape of x_train_cleaned after removing low variance features:",
        x_train_cleaned.shape,
    )
    print(
        "Shape of x_test_cleaned after removing low variance features:",
        x_test_cleaned.shape,
    )

    if method_to_run == "logistic_regression":
        # Remove highly correlated features
        x_train_cleaned, x_test_cleaned = remove_highly_correlated_features(
            x_train_cleaned, x_test_cleaned, threshold=0.80
        )
        print(
            "Shape of x_train_cleaned after removing highly correlated features:",
            x_train_cleaned.shape,
        )
        print(
            "Shape of x_test_cleaned after removing highly correlated features:",
            x_test_cleaned.shape,
        )

    # Standardize x_train_cleaned
    mean_train = np.mean(x_train_cleaned, axis=0)
    std_train = np.std(x_train_cleaned, axis=0)
    x_train_cleaned = (x_train_cleaned - mean_train) / std_train

    # Use train mean and std to standardize x_test_cleaned
    x_test_cleaned = (x_test_cleaned - mean_train) / std_train

    # Add bias term
    x_train_cleaned = np.hstack(
        (np.ones((x_train_cleaned.shape[0], 1)), x_train_cleaned)
    )
    x_test_cleaned = np.hstack((np.ones((x_test_cleaned.shape[0], 1)), x_test_cleaned))

    # Split x_train_cleaned into training and validation sets
    x_train_cleaned, y_train, x_val_cleaned, y_val = split_data(
        x_train_cleaned, y_train, ratio=0.8
    )

    # Upsample the minority class
    x_train_cleaned, y_train = upsample_minority_class(x_train_cleaned, y_train)
    print("Shape of x_train_cleaned after upsampling:", x_train_cleaned.shape)

    print("Shape of x_train_cleaned:", x_train_cleaned.shape)
    print("Shape of x_val_cleaned:", x_val_cleaned.shape)
    print("Data preprocessing completed.")

    if method_to_run == "logistic_regression":

        # Define the lambda and gamma values to test
        lambdas = [0.0001, 0.001, 0.01]
        gammas = [0.001, 0.01, 0.1, 0.5, 1]

        # Find the optimal lambda and gamma
        best_lambda, best_gamma, best_f1_score, results = find_best_lambda_gamma(
            x_train_cleaned, y_train, x_val_cleaned, y_val, lambdas, gammas
        )

        # Change y labels from -1 to 0 for training
        y_train = (y_train + 1) / 2
        y_val = (y_val + 1) / 2

        w, loss = reg_logistic_regression(
            y_train,
            x_train_cleaned,
            lambda_=best_lambda,
            initial_w=np.zeros(x_train_cleaned.shape[1]),
            max_iters=500,
            gamma=best_gamma,
        )
        print(f"Loss: {loss:.4f}")

        # Rechange label from 0 to -1 for evaluation
        y_train = 2 * y_train - 1
        y_val = 2 * y_val - 1

        # Compute accuracy and f1 score on training set
        accuracy = calculate_accuracy(
            y_train, predict_label_logistic_regression(w, x_train_cleaned)
        )
        f1_score = calculate_f1_score(
            y_train, predict_label_logistic_regression(w, x_train_cleaned)
        )
        print(f"Accuracy train: {accuracy:.4f}, F1 score train: {f1_score:.4f}")

        # Compute accuracy and f1 score on validation set
        accuracy = calculate_accuracy(
            y_val, predict_label_logistic_regression(w, x_val_cleaned)
        )
        f1_score = calculate_f1_score(
            y_val, predict_label_logistic_regression(w, x_val_cleaned)
        )
        print(
            f"Accuracy validation: {accuracy:.4f}, F1 score validation: {f1_score:.4f}"
        )

        # Compute predictions on test set
        y_test_pred = predict_label_logistic_regression(w, x_test_cleaned)

        # Save the predictions to a CSV file
        create_csv_submission(test_ids, y_test_pred, "submission_reg_log_regr.csv")

    elif method_to_run == "ridge_regression":

        # First find optimal lambda
        optimal_lambda, best_f1, results = find_optimal_lambda(
            x_train_cleaned,
            y_train,
            x_val_cleaned,
            y_val,
            lambda_range=(1e-6, 1.0),
            num_lambdas=50,
        )

        # Train final model with optimal lambda
        w, loss = ridge_regression(y_train, x_train_cleaned, optimal_lambda)
        print(f"Loss with optimal lambda: {loss:.4f}")

        # Then find optimal threshold using the model trained with optimal lambda
        optimal_threshold, best_f1 = find_optimal_threshold(w, x_val_cleaned, y_val)
        print(
            f"Optimal threshold: {optimal_threshold:.4f}, Best validation F1: {best_f1:.4f}"
        )

        # Make final predictions using both optimal lambda and optimal threshold
        train_accuracy = calculate_accuracy(
            y_train, predict_labels(w, x_train_cleaned, optimal_threshold)
        )
        train_f1 = calculate_f1_score(
            y_train, predict_labels(w, x_train_cleaned, optimal_threshold)
        )
        print(
            f"Final train metrics - Accuracy: {train_accuracy:.4f}, F1 score: {train_f1:.4f}"
        )

        val_accuracy = calculate_accuracy(
            y_val, predict_labels(w, x_val_cleaned, optimal_threshold)
        )
        val_f1 = calculate_f1_score(
            y_val, predict_labels(w, x_val_cleaned, optimal_threshold)
        )
        print(
            f"Final validation metrics - Accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}"
        )

        # Predictions for submission
        y_test_pred = predict_labels(w, x_test_cleaned, optimal_threshold)

        # Save the predictions to a CSV file
        create_csv_submission(
            test_ids, y_test_pred, "submission_ridge_reg_optimized.csv"
        )

    else:
        return


if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    main()
