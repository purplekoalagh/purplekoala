import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import warnings
from math import log
from pandas.plotting import scatter_matrix
import pandas as pd

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

class CauchyLoss(nn.Module):
  """
  Cauchy Loss Implemented as custom loss function.

  Reference used on how to create a custom loss function: https://www.codecademy.com/resources/docs/pytorch/custom-loss-functions-creation
  """
  def __init__(self, c = 1):
    super(CauchyLoss, self).__init__()
    self.c = c
  def forward(self, y_pred, y_train):
    loss = ((self.c**2)/2)*torch.log(1+((y_train-y_pred)/self.c)**2)
    return torch.mean(loss)

class CauchyRegression:
  """
  A Pytorch-based Multiple Linear Regression with Cauchy Loss for four variable.

  Model: y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4
  Loss: Cauchy
  """

  def __init__(self, learning_rate: float = 0.0001, max_epochs: int = 10000, tolerance: float = 1e-6):
    self.learning_rate = learning_rate
    self.max_epochs = max_epochs
    self.tolerance = tolerance

    # Model parameters
    self.W = torch.randn(4, 1, requires_grad=True, dtype=torch.float32)  # Random initialization for weights
    self.b = torch.randn(1, requires_grad=True, dtype=torch.float32)     # Random initialization for bias

    # Training data storage
    self.X_train = None
    self.y_train = None

    self.fitted = False

    self.loss_history = []

    # Loss function and optimizer
    self.criterion = CauchyLoss()
    self.optimizer = optim.SGD([self.W, self.b], lr=self.learning_rate)

  def forward(self, X: torch.tensor):
    """
    Forward pass of the mutliple linear model.

    Args:
        X: Input tensor of shape (n_samples,)

    Returns:
        Predictions tensor of shape (n_samples,)
    """
    return X @ self.W + self.b

  def fit(self, X_train: torch.tensor, y_train: torch.tensor):
    """
    Fit the Multiple Linear Regression with Cauchy Loss for four variable to the training data.
        
    Args:
        X: Input features of shape (n_samples,)
        y: Target values of shape (n_samples,)
            
    Returns:
        self: Returns the fitted model instance
    """
    self.n_samples = len(X_train)

    self.X_train = X_train
    self.y_train = y_train

    # Training loop
    prev_loss = float('inf')

    for epoch in range(self.max_epochs):
      self.optimizer.zero_grad()

    
      y_pred = self.forward(self.X_train)
      y_pred.requires_grad_(True)

      loss = self.criterion(y_pred, self.y_train)

      loss.backward()

      self.optimizer.step()

      current_loss = loss.item()
      self.loss_history.append(current_loss)

      # Check for convergence
      if abs(prev_loss - current_loss) < self.tolerance:
        print(f"Converged after {epoch + 1} epochs")
        break
            
      prev_loss = current_loss
    
    self.fitted = True
    return self

  def predict(self, X: torch.tensor):
    """
    Make predictions on new data.
        
    Args:
        X: Input features of shape (n_samples,)
            
    Returns:
          Predictions as torch.tensor
    """
    if not self.fitted:
        raise ValueError("Model must be fitted before making predictions")
        
    
        
    with torch.no_grad():
        predictions = self.forward(X)
        
    return predictions


  def get_parameters(self) -> Tuple[float, float, float, float, float]:
    """
    Get the fitted parameters.
    
    Returns:
          Tuple of (W, b) - slope and intercept
    """
    if not self.fitted:
        raise ValueError("Model must be fitted before accessing parameters")
        
    return float(self.W[0]), float(self.W[1]), float(self.W[2]), float(self.W[3]), float(self.b.item())

  def correlatrion_matrix(self, X_y):
    X_y = pd.DataFrame(X_y, columns=['AT', 'V', 'AP', 'RH', 'PE'])
    scatter_matrix(X_y, alpha = 0.2, figsize=(10,10), diagonal='kde')
    plt.show()

  def residual_plot(self):
    fig, axs = plt.subplots(2,2,figsize=(16,8))
    y_pred = self.predict(self.X_train)
    residual = y_pred-self.y_train
    axs[0,0].scatter(residual, self.X_train[:,0])
    axs[0,0].set_title("Residual Against x_1")
    axs[0,0].set_xlabel("x_1")
    axs[0,0].set_ylabel("Residual")
    axs[0,1].scatter(residual, self.X_train[:,1])
    axs[0,1].set_title("Residual Against x_2")
    axs[0,1].set_xlabel("x_2")
    axs[0,1].set_ylabel("Residual")
    axs[1,0].scatter(residual, self.X_train[:,2])
    axs[1,0].set_title("Residual Against x_3")
    axs[1,0].set_xlabel("x_3")
    axs[1,0].set_ylabel("Residual")
    axs[1,1].scatter(residual, self.X_train[:,3])
    axs[1,1].set_title("Residual Against x_4")
    axs[1,1].set_xlabel("x_4")
    axs[1,1].set_ylabel("Residual")
    plt.tight_layout()
    plt.show()
