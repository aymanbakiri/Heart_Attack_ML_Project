import numpy as np


# Method 2 and 4 mean square sgd and ridge regression


def compute_loss_reg(y, tx, w):
    """
    Calculate the Mean Squared Error (MSE) loss for ridge regression.

    Args:
        y (np.array): Labels for each sample, shape (N,).
        tx (np.array): Feature matrix for each sample, shape (N, D).
        w (np.array): Weight vector, shape (D,).

    Returns:
        float: Computed MSE loss value.
    """
    N = y.shape[0]  # Number of samples
    e = y - tx @ w  # Error term: difference between predicted and actual values
    return (1 / (2 * N)) * (e.T @ e)  # MSE formula


def compute_stoch_gradient_reg(y, tx, w):
    """
    Compute the stochastic gradient for ridge regression using a single data point.

    Args:
        y (float): Label for the current sample.
        tx (np.array): Feature vector for the current sample, shape (D,).
        w (np.array): Weight vector, shape (D,).

    Returns:
        tuple: Gradient and error term:
            - grad (np.array): Stochastic gradient, shape (D,).
            - e (float): Error term for the current sample.
    """
    e = y - np.dot(
        tx, w
    )  # Error term: difference between predicted and actual values for a single sample
    grad = -np.dot(tx.T, e)  # Stochastic gradient calculation
    return (grad, e)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate mini-batches from the dataset for stochastic or mini-batch gradient descent.

    Args:
        y (np.array): Labels for each sample in the dataset, shape (N,).
        tx (np.array): Feature matrix for each sample in the dataset, shape (N, D).
        batch_size (int): Number of samples per batch.
        num_batches (int): Number of batches to generate. If not set, it defaults to 1.
        shuffle (bool): Whether to shuffle the data before creating batches. Default is True.

    Yields:
        tuple: A tuple containing a batch of labels and features:
            - y_batch (np.array): Batch of labels, shape (batch_size,).
            - tx_batch (np.array): Batch of feature samples, shape (batch_size, D).
    """

    data_size = len(y)  # Total number of data points in the dataset.
    batch_size = min(
        data_size, batch_size
    )  # Ensure batch size does not exceed total data points.
    max_batches = (
        data_size // batch_size
    )  # Maximum number of non-overlapping batches possible.
    remainder = (
        data_size % batch_size
    )  # Number of leftover points if data size isn't perfectly divisible.

    if shuffle:
        # Generate random starting indices for each batch.
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Randomly add an offset within the range of remainder for including leftover points.
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # For non-shuffled data, the indices cycle sequentially through available batches.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    # Yield mini-batches of labels and features.
    for start in idxs:
        start_index = start  # Start index of the batch
        end_index = start_index + batch_size  # End index of the batch
        yield y[start_index:end_index], tx[start_index:end_index]


# Method 5 and 6 logistic regression and regularized logistic regression


def sigmoid(t):
    """
    Apply the sigmoid function to the input t, which maps real-valued numbers to the range (0, 1).

    Args:
        t (np.array): Input array, can be a scalar or a vector.

    Returns:
        np.array: Transformed values in the range (0, 1) using the sigmoid function.
    """
    return 1 / (1 + np.exp(-t))  # Sigmoid formula


def calculate_loss_lr(y, tx, w):
    """
    Compute the cost using the negative log likelihood for logistic regression.

    Args:
        y (np.array): Labels for each sample, shape (N,).
        tx (np.array): Feature matrix, shape (N, D).
        w (np.array): Weight vector, shape (D,).

    Returns:
        float: Computed negative log likelihood loss.
    """
    pred = sigmoid(tx @ w)  # Predicted probabilities for each sample
    # Negative log likelihood formula, averaged over all samples
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))


def calculate_gradient_lr(y, tx, w):
    """
    Compute the gradient of the negative log likelihood loss for logistic regression.

    Args:
        y (np.array): Labels for each sample, shape (N,).
        tx (np.array): Feature matrix, shape (N, D).
        w (np.array): Weight vector, shape (D,).

    Returns:
        np.array: Computed gradient vector, shape (D,).
    """
    N = tx.shape[0]  # Number of samples
    predictions = sigmoid(np.dot(tx, w))  # Predicted probabilities, shape (N,)
    error = (
        predictions - y
    )  # Difference between predictions and actual labels, shape (N,)
    gradient = np.dot(tx.T, error) / N  # Gradient calculation, shape (D,)
    return gradient


# Pre-Processing the Data


def split_data(x, y, ratio, seed=42):
    """
    Split the dataset into training and test sets based on the split ratio.

    Args:
        x (numpy array): Feature matrix, shape (N, D).
        y (numpy array): Labels vector, shape (N,).
        ratio (float): Proportion of data to use as the training set (e.g., 0.8 for 80% training).
        seed (int): Random seed for reproducibility of the split.

    Returns:
        (numpy array, numpy array, numpy array, numpy array):
            Four numpy arrays - training features, training labels, test features, test labels.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    indices = np.random.permutation(x.shape[0])  # Shuffle indices for x
    split_index = int(x.shape[0] * ratio)  # Determine split index based on ratio
    # Split indices into training and test sets
    training_idx, test_idx = indices[:split_index], indices[split_index:]
    return x[training_idx], y[training_idx], x[test_idx], y[test_idx]


def build_poly(x, degree):
    """
    Generate polynomial basis functions for input data x up to the specified degree.

    Args:
        x (numpy array): Input feature matrix, shape (N,).
        degree (int): Highest degree of polynomial terms to generate.

    Returns:
        numpy array: Expanded feature matrix with polynomial terms, shape (N, degree + 1).
    """
    # Stack columns of x raised to powers from 0 up to degree
    poly = np.column_stack([x**d for d in range(degree + 1)])
    return poly


def drop_nan_columns(x_train, threshold=0.5):
    """
    Remove columns in the feature matrix where the proportion of NaN values exceeds a specified threshold.

    Args:
        x_train (numpy array): Feature matrix, shape (N, D).
        threshold (float): Maximum allowed fraction of NaN values per column (0-1 range).

    Returns:
        numpy array: Feature matrix with columns containing excess NaN values removed.
    """
    num_samples = x_train.shape[0]  # Total number of samples in the dataset
    nan_counts = np.isnan(x_train).sum(axis=0)  # Count NaNs in each column
    nan_fraction = nan_counts / num_samples  # Calculate the fraction of NaNs per column
    # Create a mask to keep columns below the threshold of NaN values
    cols_to_keep = nan_fraction < threshold
    return x_train[:, cols_to_keep]  # Filter columns based on the mask


def upsample_minority_class(X, y, add_fraction=2, noise_factor=0.05):
    """
    Upsample the minority class by replicating a fraction of its samples with optional noise.

    Args:
        X (numpy array): Feature matrix.
        y (numpy array): Labels.
        add_fraction (float): Factor by which to increase the minority class samples.
        noise_factor (float): Factor for random noise added to new samples to introduce slight variability.

    Returns:
        X (numpy array): Upsampled feature matrix.
        y (numpy array): Upsampled labels.
    """
    # Identify indices of the minority class in labels
    minority_indices = np.where(y == 1)[0]

    # Calculate the number of minority samples to add
    num_to_add = int(len(minority_indices) * add_fraction)

    # Randomly select indices with replacement from the minority class
    indices_to_add = np.random.choice(minority_indices, num_to_add, replace=True)

    # Create new samples by adding random noise to the selected samples
    new_samples = X[indices_to_add] + noise_factor * np.random.randn(
        *X[indices_to_add].shape
    )

    # Add the new samples to the original feature matrix
    X = np.vstack((X, new_samples))
    y = np.hstack((y, y[indices_to_add]))  # Add new labels for the upsampled data

    return X, y


def remove_highly_correlated_features(X_train, X_test, threshold=0.8):
    """
    Remove features from X_train and X_test that are highly correlated in X_train.

    Args:
        X_train (numpy array): Training feature matrix.
        X_test (numpy array): Testing feature matrix.
        threshold (float): Correlation coefficient threshold above which features are removed.

    Returns:
        X_train_reduced (numpy array): Reduced training feature matrix.
        X_test_reduced (numpy array): Reduced testing feature matrix.
    """
    # Compute the correlation matrix for X_train features
    correlation_matrix = np.corrcoef(X_train, rowvar=False)

    # Identify features to remove based on correlation threshold
    to_remove = set()
    for i in range(len(correlation_matrix)):
        for j in range(i):
            # Mark feature 'i' for removal if it has high correlation with feature 'j'
            if abs(correlation_matrix[i, j]) > threshold:
                to_remove.add(i)

    # Remove selected features from the training data
    X_train_reduced = np.delete(X_train, list(to_remove), axis=1)

    # Remove the same features from the test data
    X_test_reduced = np.delete(X_test, list(to_remove), axis=1)

    return X_train_reduced, X_test_reduced


def fill_missing_values(X, discrete_threshold=20, neutral_value=-1):
    """
    Fill missing values in X based on feature type:
    - For discrete features (having fewer unique values than `discrete_threshold`), fill with a neutral category.
    - For continuous features, fill with the mean of the feature.

    Args:
        X (numpy array): Dataset with missing values represented as NaN.
        discrete_threshold (int): Maximum unique values to consider a feature as discrete.
        neutral_value (int or float): Value used to fill missing values in discrete features.

    Returns:
        X (numpy array): Dataset with filled missing values.
    """
    for i in range(X.shape[1]):
        unique_values = np.unique(X[:, i][~np.isnan(X[:, i])])

        # Determine if the feature is discrete or continuous
        if (
            len(unique_values) < discrete_threshold
        ):  # Treat as discrete if unique values are below threshold
            # Fill missing values with the neutral category for discrete features
            X[np.isnan(X[:, i]), i] = neutral_value
        else:
            # Fill missing values with the mean for continuous features
            mean_value = np.nanmean(X[:, i])
            X[np.isnan(X[:, i]), i] = mean_value

    return X


def remove_low_variance_features(X, threshold=0.01):
    """
    Remove features from X that have a variance below the specified threshold.

    Args:
        X (numpy array): Feature matrix.
        threshold (float): Variance threshold; features with variance below this value will be removed.

    Returns:
        X_reduced (numpy array): Feature matrix with low-variance features removed.
        mask (numpy array): Boolean array indicating which features were retained (True for retained features).
    """
    # Calculate the variance for each feature (column) in X
    variances = np.var(X, axis=0)

    # Create a mask indicating which features have variance above the threshold
    mask = variances > threshold

    # Select only features with variance above the threshold
    X_reduced = X[:, mask]

    return X_reduced, mask


# Evaluation Metrics and Predictions


def predict_labels(w, tx, threshold=0.0):
    """
    Generate binary predictions from the ridge regression model.
    Arguments:
    - w: weights (result from ridge regression).
    - tx: feature matrix.
    - threshold: threshold to apply for classification.

    Returns:
    - Binary predictions (-1 and 1 by default).
    """
    # Compute the continuous predictions (Xw)
    y_continuous = tx.dot(w)

    # Apply threshold to generate binary predictions (-1 or 1)
    y_pred = np.where(y_continuous >= threshold, 1, -1)

    return y_pred


def predict_label_logistic_regression(w, tx):
    """
    Generate binary predictions from the logistic regression model.
    Arguments:
    - w: weights (result from logistic regression).
    - tx: feature matrix.

    Returns:
    - Binary predictions (-1 and 1 by default).
    """
    # Compute the continuous predictions (sigmoid(Xw))
    y_pred = sigmoid(tx.dot(w))

    # Apply threshold to generate binary predictions (-1 or 1)
    y_pred = np.where(y_pred >= 0.5, 1, -1)

    return y_pred


def safe_division(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def calculate_f1_score(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    precision = safe_division(tp, (tp + fp))
    recall = safe_division(tp, (tp + fn))
    f1 = safe_division(2 * precision * recall, (precision + recall))

    return f1
