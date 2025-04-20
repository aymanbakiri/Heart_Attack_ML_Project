from implementations import *
import numpy as np


def cross_validate(x_train, y_train, lambda_values, n_folds=5):
    best_lambda = None
    thres = None
    best_f1_score = -np.inf

    data = np.column_stack((x_train, y_train))
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]

    fold_size = len(x_train) // n_folds
    folds = []

    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            end = len(x_train)
        else:
            end = start + fold_size

        X_val = x[start:end]
        y_val = y[start:end]

        X_train = np.concatenate((x[:start], x[end:]))
        y_train = np.concatenate((y[:start], y[end:]))

        folds.append((X_train, y_train, X_val, y_val))

    for lambda_ in lambda_values:
        f1_scores1 = []
        f1_scores2 = []
        f1_scores3 = []
        f1_scores4 = []

        for x_train_fold, y_train_fold, x_val_fold, y_val_fold in folds:

            # Initialize weights for each fold
            w = np.zeros(x_train_fold.shape[1])

            # Perform logistic regression
            w, loss = ridge_regression(y_train_fold, x_train_fold, lambda_)

            # Calculate F1 score on the validation set
            y_val_pred = predict_labels(w, x_val_fold, threshold=0.0)
            f1 = calculate_f1_score(y_val_fold, y_val_pred)
            f1_scores1.append(f1)

            y_val_pred = predict_labels(w, x_val_fold, threshold=-0.34)
            f1 = calculate_f1_score(y_val_fold, y_val_pred)
            f1_scores2.append(f1)

            y_val_pred = predict_labels(w, x_val_fold, threshold=0.34)
            f1 = calculate_f1_score(y_val_fold, y_val_pred)
            f1_scores3.append(f1)

            y_val_pred = predict_labels(w, x_val_fold, threshold=-0.5)
            f1 = calculate_f1_score(y_val_fold, y_val_pred)
            f1_scores4.append(f1)

        avg_f1_score1 = np.mean(f1_scores1)
        print(f1_scores1)
        print(
            f"Lambda: {lambda_}, Average F1 score1: {avg_f1_score1:.4f}, Threshhold: 0.0"
        )

        avg_f1_score2 = np.mean(f1_scores2)
        print(f1_scores2)
        print(
            f"Lambda: {lambda_}, Average F1 score1: {avg_f1_score2:.4f}, Threshhold: -0.34"
        )

        avg_f1_score3 = np.mean(f1_scores3)
        print(f1_scores3)
        print(
            f"Lambda: {lambda_}, Average F1 score1: {avg_f1_score3:.4f}, Threshhold: 0.34"
        )

        avg_f1_score4 = np.mean(f1_scores4)
        print(f1_scores4)
        print(
            f"Lambda: {lambda_}, Average F1 score1: {avg_f1_score4:.4f}, Threshhold: -0.5"
        )

        # Keep track of the best lambda and threshhold based on F1 score
        if avg_f1_score1 > best_f1_score:
            best_f1_score = avg_f1_score1
            best_lambda = lambda_
            thres = 0.0

        if avg_f1_score2 > best_f1_score:
            best_f1_score = avg_f1_score2
            best_lambda = lambda_
            thres = -0.34

        if avg_f1_score3 > best_f1_score:
            best_f1_score = avg_f1_score3
            best_lambda = lambda_
            thres = 0.34

        if avg_f1_score4 > best_f1_score:
            best_f1_score = avg_f1_score4
            best_lambda = lambda_
            thres = -0.5

    print(
        f"Best lambda: {best_lambda}, Best F1 score: {best_f1_score:.4f}, Best Threshold: {thres}"
    )
    return best_lambda


def cross_validate_logistic(x_train, y_train, lambda_values, gamma_values, n_folds=5):
    best_lambda = None
    best_gamma = None
    best_f1_score = -np.inf

    data = np.column_stack((x_train, y_train))
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]

    fold_size = len(x_train) // n_folds
    folds = []

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(x_train)

        X_val = x[start:end]
        y_val = y[start:end]

        X_train = np.concatenate((x[:start], x[end:]))
        y_train = np.concatenate((y[:start], y[end:]))

        folds.append((X_train, y_train, X_val, y_val))

    for lambda_ in lambda_values:
        for gamma in gamma_values:
            f1_scores = []

            for X_train_fold, y_train_fold, X_val_fold, y_val_fold in folds:
                # Initialize weights for each fold
                w = np.zeros(X_train_fold.shape[1])

                # change labels
                y_train_fold = (y_train_fold + 1) / 2

                # Perform logistic regression with the current lambda and gamma
                w, loss = reg_logistic_regression(
                    y_train_fold,
                    X_train_fold,
                    lambda_,
                    np.zeros(X_train_fold.shape[1]),
                    1000,
                    gamma,
                )

                # rechange labels
                y_train_fold = 2 * y_train_fold - 1

                # Calculate F1 score on the validation set
                y_val_pred = predict_labels(w, X_val_fold)
                f1 = calculate_f1_score(y_val_fold, y_val_pred)
                f1_scores.append(f1)

            avg_f1_score = np.mean(f1_scores)

            if avg_f1_score > best_f1_score:
                best_f1_score = avg_f1_score
                best_lambda = lambda_
                best_gamma = gamma

    print(
        f"Best lambda: {best_lambda}, Best gamma: {best_gamma}, Best F1 score: {best_f1_score:.4f}"
    )
    return best_lambda, best_gamma


def cross_validate_logistic_no_folds(x_train, y_train, lambda_values, gamma_values):
    best_lambda = None
    best_gamma = None
    best_f1_score = -np.inf
    f1_scores_per_gamma = {gamma: [] for gamma in gamma_values}

    for lambda_ in lambda_values:
        for gamma in gamma_values:
            # Initialize weights
            w = np.zeros(x_train.shape[1])
            # change labels
            y_train = (y_train + 1) / 2
            # Perform logistic regression
            w, loss = reg_logistic_regression(y_train, x_train, lambda_, w, 800, gamma)
            # rechange labels
            y_train = 2 * y_train - 1
            # Calculate F1 score on the entire dataset
            y_pred = predict_labels(w, x_train)
            f1 = calculate_f1_score(y_train, y_pred)
            f1_scores_per_gamma[gamma].append(f1)

            if f1 > best_f1_score:
                best_f1_score = f1
                best_lambda = lambda_
                best_gamma = gamma

    print(
        f"Best lambda: {best_lambda}, Best gamma: {best_gamma}, Best F1 score: {best_f1_score:.4f}"
    )
    return best_lambda, best_gamma, f1_scores_per_gamma


def find_optimal_threshold(w, x_val, y_val, num_thresholds=100):
    """
    Find the optimal threshold for ridge regression predictions that maximizes F1 score

    Args:
        w: trained weights
        x_val: validation features
        y_val: validation labels
        num_thresholds: number of threshold values to test

    Returns:
        optimal_threshold: threshold that maximizes F1 score
        best_f1: best F1 score achieved
    """
    # Get raw predictions
    raw_predictions = x_val @ w

    # Define range of thresholds to try
    min_pred = np.min(raw_predictions)
    max_pred = np.max(raw_predictions)
    thresholds = np.linspace(min_pred, max_pred, num_thresholds)

    # Initialize variables to store best results
    best_f1 = 0
    optimal_threshold = 0

    # Try each threshold
    for threshold in thresholds:
        # Convert raw predictions to labels using current threshold
        y_pred = np.where(raw_predictions > threshold, 1, -1)

        # Calculate F1 score for current threshold
        current_f1 = calculate_f1_score(y_val, y_pred)

        # Update best if current is better
        if current_f1 > best_f1:
            best_f1 = current_f1
            optimal_threshold = threshold

    return optimal_threshold, best_f1


def find_optimal_lambda(
    x_train, y_train, x_val, y_val, lambda_range=None, num_lambdas=50
):
    """
    Find optimal lambda for ridge regression using validation set.

    Args:
        x_train: training features
        y_train: training labels
        x_val: validation features
        y_val: validation labels
        lambda_range: tuple of (min_lambda, max_lambda). If None, uses default range
        num_lambdas: number of lambda values to test

    Returns:
        optimal_lambda: lambda that gives best F1 score
        best_f1: best F1 score achieved
        results: dictionary containing lambda values and their corresponding metrics
    """
    if lambda_range is None:
        lambda_range = (1e-6, 1.0)

    # Create array of lambda values to test (log space)
    lambdas = np.logspace(
        np.log10(lambda_range[0]), np.log10(lambda_range[1]), num_lambdas
    )

    # Initialize arrays to store results
    train_f1_scores = []
    val_f1_scores = []
    train_accuracies = []
    val_accuracies = []
    thresholds = []

    # Try each lambda value
    for lambda_ in lambdas:
        # Train model with current lambda
        w, loss = ridge_regression(y_train, x_train, lambda_)

        # Find optimal threshold for this lambda
        optimal_threshold, _ = find_optimal_threshold(w, x_val, y_val)
        thresholds.append(optimal_threshold)

        # Get predictions using optimal threshold
        y_train_pred = predict_labels(w, x_train, optimal_threshold)
        y_val_pred = predict_labels(w, x_val, optimal_threshold)

        # Calculate metrics
        train_f1 = calculate_f1_score(y_train, y_train_pred)
        val_f1 = calculate_f1_score(y_val, y_val_pred)
        train_acc = calculate_accuracy(y_train, y_train_pred)
        val_acc = calculate_accuracy(y_val, y_val_pred)

        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # Find optimal lambda (based on validation F1 score)
    best_idx = np.argmax(val_f1_scores)
    optimal_lambda = lambdas[best_idx]
    best_f1 = val_f1_scores[best_idx]
    optimal_threshold = thresholds[best_idx]

    # Store all results
    results = {
        "lambdas": lambdas,
        "train_f1_scores": train_f1_scores,
        "val_f1_scores": val_f1_scores,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "optimal_threshold": optimal_threshold,
    }

    print(f"Optimal lambda: {optimal_lambda:.6f}")
    print(f"Best validation F1 score: {best_f1:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    return optimal_lambda, best_f1, results


def find_best_lambda_gamma(
    x_train, y_train, x_val, y_val, lambdas, gammas, max_iters=500
):
    """
    Find the best combination of lambda and gamma for logistic regression.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels.
        x_val: Validation feature matrix.
        y_val: Validation labels.
        lambdas: List of lambda values to test.
        gammas: List of gamma values to test.
        max_iters: Maximum iterations for logistic regression.

    Returns:
        best_lambda: Optimal lambda value.
        best_gamma: Optimal gamma value.
        best_f1_score: Best F1 score achieved.
        results: Dictionary of all results for analysis.
    """
    best_f1_score = -np.inf
    best_lambda = None
    best_gamma = None
    results = {"lambdas": [], "gammas": [], "train_f1_scores": [], "val_f1_scores": []}

    # Perform grid search
    for lambda_ in lambdas:
        for gamma in gammas:

            # change y labels from -1 to 0
            y_train = (y_train + 1) / 2
            y_val = (y_val + 1) / 2

            # Train logistic regression with current lambda and gamma
            w, loss = reg_logistic_regression(
                y_train,
                x_train,
                lambda_,
                initial_w=np.zeros(x_train.shape[1]),
                max_iters=max_iters,
                gamma=gamma,
            )

            # rechange label from 0 to -1
            y_train = 2 * y_train - 1
            y_val = 2 * y_val - 1

            # Calculate F1 score on training and validation sets
            y_train_pred = predict_label_logistic_regression(w, x_train)
            train_f1 = calculate_f1_score(y_train, y_train_pred)

            y_val_pred = predict_label_logistic_regression(w, x_val)
            val_f1 = calculate_f1_score(y_val, y_val_pred)

            # Save results for analysis
            results["lambdas"].append(lambda_)
            results["gammas"].append(gamma)
            results["train_f1_scores"].append(train_f1)
            results["val_f1_scores"].append(val_f1)

            # Update best F1 score and parameters if this is the best so far
            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                best_lambda = lambda_
                best_gamma = gamma

            # Print current result for tracking progress
            print(
                f"Lambda: {lambda_}, Gamma: {gamma}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}"
            )

    print(
        f"Best Lambda: {best_lambda}, Best Gamma: {best_gamma}, Best Validation F1 Score: {best_f1_score:.4f}"
    )

    return best_lambda, best_gamma, best_f1_score, results
