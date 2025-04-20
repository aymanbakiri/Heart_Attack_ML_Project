import numpy as np
from utilities import *


# METHOD 1
# This method performs linear regression using full-batch gradient descent.
# It minimizes the mean squared error by iteratively updating weights based on the gradient.
# The final output is the optimized weights and the mean squared error loss.


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
    """
    # y: target variable, shape (N,)
    # tx: feature matrix, shape (N, D)
    # initial_w: initial weights, shape (D,)
    # max_iters: maximum number of iterations
    # gamma: learning rate
    w = initial_w
    for iter in range(max_iters):
        pred = tx.dot(w)
        error = y - pred
        gradient = -tx.T.dot(error) / y.shape[0]
        w = w - gamma * gradient
    loss = compute_loss_reg(y, tx, w)
    return w, loss


# METHOD 2
# This method performs linear regression using stochastic gradient descent (SGD).
# It uses a mini-batch size of 1 to iteratively update weights based on each sample’s gradient.
# The output is the optimized weights and the mean squared error loss on the full dataset.


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent for linear regression.
    """
    # y: target variable, shape (N,)
    # tx: feature matrix, shape (N, D)
    # initial_w: initial weights, shape (D,)
    # max_iters: maximum number of iterations
    # gamma: learning rate

    w = initial_w  #
    batch_size = 1  # Set batch size for SGD to 1

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size
        ):  # Loop through mini-batches
            gradient = compute_stoch_gradient_reg(
                y_batch, tx_batch, w
            )  # Compute stochastic gradient
            w = (
                w - gamma * gradient[0]
            )  # Update weights with learning rate and gradient

        loss = compute_loss_reg(y, tx, w)
    return w, loss


# METHOD 3
# This method solves linear regression by finding the closed-form least squares solution.
# It calculates the weights that minimize the mean squared error without iterative updates.
# The final output is the optimized weights and the least squares loss.


def least_squares(y, tx):
    """
    Calculate the least squares solution.
    """
    # y: target variable, shape (N,)
    # tx: feature matrix, shape (N, D)

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sum((y - tx.dot(w)) ** 2) / (2 * len(y))
    return w, loss


# METHOD 4
# This method performs ridge regression with L2 regularization to prevent overfitting.
# It minimizes the regularized mean squared error by solving a modified least squares equation.
# The final output is the optimized weights and the ridge regression loss.


def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression.
    """
    # y: target variable, shape (N,)
    # tx: feature matrix, shape (N, D)
    # lambda_: regularization parameter

    d = tx.shape[1]
    N = tx.shape[0]
    I = np.eye(d)
    left_hand_side = tx.T @ tx + I * (2 * N * lambda_)
    right_hand_side = tx.T @ y

    w = np.linalg.solve(left_hand_side, right_hand_side)
    loss = compute_loss_reg(y, tx, w)
    return w, loss


# METHOD 5
# This method performs logistic regression using gradient descent to classify binary targets.
# It iteratively updates weights to minimize logistic loss based on the gradient.
# The final output is the optimized weights and the logistic loss.


def logistic_regression(y, tx, w, max_iters, gamma):
    """
    Logistic regression using gradient descent.
    """
    # y: target variable, shape (N,)
    # tx: feature matrix, shape (N, D)
    # w: initial weights, shape (D,)
    # max_iters: maximum number of iterations
    # gamma: learning rate

    for iter in range(max_iters):
        grad = calculate_gradient_lr(y, tx, w)
        w = w - gamma * grad
    loss = calculate_loss_lr(y, tx, w)
    return w, loss


# METHOD 6
# This method performs regularized logistic regression using gradient descent to classify binary targets.
# It minimizes the regularized logistic loss by iteratively updating weights based on the gradient.
# The final output is the optimized weights and the regularized logistic loss.


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.
    """
    # y: target variable, shape (N,)
    # tx: feature matrix, shape (N, D)
    # lambda_: regularization parameter
    # initial_w: initial weights, shape (D,)
    # max_iters: maximum number of iterations
    # gamma: learning rate

    w = initial_w
    for iter in range(max_iters):
        gradient = calculate_gradient_lr(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = calculate_loss_lr(y, tx, w)
    return w, loss
