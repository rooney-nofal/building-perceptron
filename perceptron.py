"""
Perceptron — a single artificial neuron, implemented from scratch with NumPy.

Rosenblatt's Perceptron (1958) is a supervised binary linear classifier.
It learns one weight per feature plus one bias, and separates the two classes
with a single flat boundary (a line in 2D, a plane in 3D, a hyperplane beyond).

The implementation follows scikit-learn conventions: settings are given to
__init__, learning happens in fit(), and attributes learned during training end
with an underscore (weights_, bias_, errors_).

Project : Building Perceptron — La Plateforme
Author  : Rooney Nofal
"""

import numpy as np  # NumPy gives us fast arrays and the dot product the neuron is built on.


class Perceptron:  # Our model, written as a class so it can be reused like any scikit-learn estimator.
    """Binary linear classifier trained with Rosenblatt's perceptron rule.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Size of each correction step (eta in the equations).
    n_epochs : int, default=50
        Maximum number of complete passes over the training data.
    random_state : int, default=1
        Seed for the random starting weights, so every run is reproducible.

    Attributes learned during fit
    -----------------------------
    weights_ : ndarray of shape (n_features,)
        One weight per feature.
    bias_ : float
        The constant term, learned exactly like a weight.
    errors_ : list of int
        Number of misclassified examples in each epoch (the convergence curve).
    """

    def __init__(self, learning_rate=0.01, n_epochs=50, random_state=1):  # Runs when a Perceptron is created; only stores settings.
        self.learning_rate = learning_rate  # How far the weights move on each correction.
        self.n_epochs = n_epochs            # Upper limit on full passes through the training set.
        self.random_state = random_state    # Fixes the randomness, so the same code gives the same result every time.

    def net_input(self, X):  # Stage 1 of the neuron: combine the features into a single score.
        return np.dot(X, self.weights_) + self.bias_  # z = w . x + b — the weighted sum, plus the bias.

    def activation(self, z):  # Stage 2 of the neuron: turn that score into a class.
        return np.where(z >= 0.0, 1, 0)  # Heaviside step function: 1 when the score reaches zero, otherwise 0.

    def predict(self, X):  # The two stages together — what the model answers for X.
        return self.activation(self.net_input(X))  # Compute the score, then apply the threshold.

    def fit(self, X, y):  # Training: this is where the Perceptron actually learns.
        rng = np.random.RandomState(self.random_state)  # A random generator fixed by our seed.
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])  # Start with one tiny random weight per feature.
        self.bias_ = 0.0  # The bias starts at zero.
        self.errors_ = []  # Will collect the mistake count of each epoch.

        for _ in range(self.n_epochs):  # Repeat the whole training set, up to n_epochs times.
            errors = 0  # Counts the mistakes made during this single pass.
            for xi, target in zip(X, y):  # Visit the training examples one at a time, in order.
                update = self.learning_rate * (target - self.predict(xi))  # eta * (y - y_hat) — exactly zero when the answer was right.
                self.weights_ += update * xi  # Correct every weight, each in proportion to its own feature value.
                self.bias_ += update  # Correct the bias by the same step.
                errors += int(update != 0.0)  # Count a mistake only when an update actually happened.
            self.errors_.append(errors)  # Store this epoch's mistake count, for the convergence curve.
            if errors == 0:  # A full pass with no mistake means the data is now perfectly separated...
                break  # ...so we stop early: no further update could ever occur.
        return self  # Return the trained model, so calls can be chained, e.g. Perceptron().fit(X, y).

    def score(self, X, y):  # Convenience method: how often the model is right.
        return float(np.mean(self.predict(X) == y))  # Share of predictions that match the true labels.
