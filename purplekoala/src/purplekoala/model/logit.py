import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import warnings
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn import metrics #for checking the model accuracy
from sklearn.metrics import confusion_matrix, classification_report



class LogisticsRegression:
  def __init__(self, learning_rate: float = 0.001, max_epochs: int = 100, split=0.1, tolerance: float = 1e-9):
    self.learning_rate = learning_rate
    self.max_epochs = max_epochs
    self.tolerance = tolerance
    self.split = split

    # Model parameters
    self.W = None
    self.b = None

    self.cross_ent = np.zeros(max_epochs)

    self.X_train = None
    self.y_train = None
    
    self.model = None



  def sigmoid(self, X: torch.tensor):
    return 1.0/(1+torch.exp(-(torch.dot(self.W, X) + self.b)))

  def crossent(self, X: torch.tensor, y: torch.tensor):
    epsilon = 1e-9
    y_1 = 0
    y_0 = 0
    for i in range(X[y==1].size(0)):
        y_1 += torch.log(self.sigmoid(X[y==1][i]) + epsilon)
    for i in range(X[y==0].size(0)):
        y_0 +=torch.log(1.0-self.sigmoid(X[y==0][i]) + epsilon)
    ce = y_1 + y_0
    return -ce

  def fit(self, X: torch.tensor, y: torch.tensor):
    i = torch.randperm(len(y))
    X, y = X[i], y[i]
    self.W, self.b = torch.zeros(X.shape[1]), 0
    M = np.floor((1-self.split)*len(y)).astype(int)
    Xtr, ytr, Xva, yva = X[:M], y[:M], X[M:], y[M:]


    for t in range(self.max_epochs):
      for i in torch.randperm(len(ytr)):
        self.W += self.learning_rate*(ytr[i].reshape(-1) - self.sigmoid(Xtr[i].reshape(-1)))*Xtr[i].reshape(-1)
        self.b += self.learning_rate*(ytr[i].reshape(-1) - self.sigmoid(Xtr[i].reshape(-1)))
      self.cross_ent[t] = self.crossent(Xva, yva)
      if abs(self.cross_ent[t] - self.cross_ent[t-1]) <= self.tolerance:
        print(f"Converged at {t} epochs")
        break

    self.fitted = True
    return self

  def predict(self, X: torch.tensor):
    return_list = []
    for i in range(X.size(0)):
      return_list.append(1.0*(self.sigmoid(X[i]) >= 0.5))
    return_tensor= torch.tensor(return_list)
    return return_tensor


  def predict_proba(self, X: torch.tensor):
    return_list = []
    for i in range(X.size(0)):
      return_list.append(self.sigmoid(X))
    return_tensor= torch.tensor(return_list)
    return return_tensor

  def predict_log_proba(self, X: torch.tensor):
    return torch.log(self.predict_proba(X))

  def assessment_metrics(self, X_test_tensor: torch.tensor, y_test_tensor: torch.tensor):
    prediction=self.predict(X_test_tensor)

    print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,y_test_tensor))
    print(confusion_matrix(y_test_tensor, prediction))
    print(classification_report(y_test_tensor, prediction))
