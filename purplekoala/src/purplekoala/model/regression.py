import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import warnings

class LinearRegression:
    """
    A PyTorch-based Linear Regression implementation for one variable.

    Model: y = w_1 * x + w_0
    Loss: Mean Squared Error

    Acknowledgement:
      Thank you Dr. Bhadani (UAH) for the code basis for this assignment
    """

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000):
        """
        Initialize the Linear Regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # Model parameters
        self.w_1 = nn.Parameter(torch.randn(1, requires_grad=True))  # slope
        self.w_0 = nn.Parameter(torch.randn(1, requires_grad=True))  # intercept

        # Training data storage
        self.X_train = None
        self.y_train = None

        # Model statistics for confidence intervals
        self.n_samples = None
        self.residual_sum_squares = None
        self.X_mean = None
        self.X_var = None
        self.fitted = False

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD([self.w_1, self.w_0], lr=self.learning_rate)

        # Training history
        self.loss_history = []
        self.w0_history = []
        self.w1_history = []
        self.model_states = []



    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.

        Args:
            X: Input tensor of shape (n_samples,)

        Returns:
            Predictions tensor of shape (n_samples,)
        """
        return self.w_1 * X + self.w_0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data.

        Args:
            X_train: Input features of shape (n_samples,)
            y_train: Target values of shape (n_samples,)
            X_test: Input test features of shape (n_samples,)
            y_test: Target test values of shape (n_samples)

        Returns:
            self: Returns the fitted model instance
        """
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        self.n_samples = len(X_train)

        # Store statistics for confidence intervals
        self.X_mean = float(np.mean(X_train))
        self.X_var = float(np.var(X_train, ddof=1))  # Sample variance

        # Training loop
        prev_loss = float('inf')

        for epoch in range(self.max_epochs):
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.forward(self.X_train)

            # Compute loss
            loss = self.criterion(y_pred, self.y_train)

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            #Store current w1 and w0
            self.w1_history.append(self.w_1.clone().detach().numpy())
            self.w0_history.append(self.w_0.clone().detach().numpy())


            # Store loss history
            current_loss = loss.item()
            self.loss_history.append(current_loss)


            prev_loss = current_loss

        # Compute residual sum of squares for confidence intervals
        with torch.no_grad():
            y_pred = self.forward(self.X_train)
            residuals = self.y_train - y_pred
            self.residual_sum_squares = float(torch.sum(residuals ** 2))

        self.fitted = True

        y_mean = float(torch.mean(self.y_train))
        ss_tot = float(torch.sum((self.y_train - y_mean) ** 2))
        sse = 0.0
        test_predictions = self.predict(self.X_test)
        for index in range(0,len(self.X_test)):
          sse += (self.y_train[index] - test_predictions[index]) **2

        test_r_squared = 1 - (sse / ss_tot)
        print(f"Test R^2 = {test_r_squared}")



        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features of shape (n_samples,)

        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.forward(X_tensor)

        return predictions.numpy()

    def analysis_plot(self):

      """
      Displays plots of orginal data, fitted regression line, and w_0, w_1, and loss as traing progressed

      """

      if not self.fitted:
            raise ValueError("Model must be fitted before plotting")

      # Create figure
      fig, ax = plt.subplots(4,figsize=(10, 15))

      # Convert training data to numpy for plotting
      X_np = self.X_train.numpy()
      y_np = self.y_train.numpy()

      ax[0].scatter(x=X_np, y=y_np, label="Original Data")
      ax[0].plot(X_np, self.w_1.item()*X_np +self.w_0.item(), label="Fitted Line", color='red')
      ax[0].set_title("Orginal Data BCR vs Annual Production")
      ax[0].set(xlabel='BCR', ylabel='Annual Production')

      ax[1].plot(range(self.max_epochs), self.w1_history, label='w_1 (weight)', color='blue')
      ax[1].set_title("w_1 as a Function of Epochs")
      ax[1].set(xlabel='Epochs', ylabel='Parameter value')

      ax[2].plot(range(self.max_epochs), self.w0_history, label='w01 (weight)', color='green')
      ax[2].set_title("w_0 as a Function of Epochs")
      ax[2].set(xlabel='Epochs', ylabel='Parameter value')

      ax[3].plot(range(self.max_epochs), self.loss_history, label='loss', color='red')
      ax[3].set_title("Loss as a Function of Epochs")
      ax[3].set(xlabel='Epochs', ylabel='Parameter value')

      fig.tight_layout()
      plt.show()
