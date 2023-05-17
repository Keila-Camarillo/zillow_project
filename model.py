import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

#sklearn imports
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import wrangle as w
import evaluate as e

def metrics_reg(y, yhat):
    """
    Calculate regression evaluation metrics.

    Parameters:
        y (array-like): Ground truth or actual target values.
        yhat (array-like): Predicted target values.

    Returns:
        tuple: A tuple containing the following regression evaluation metrics:
            - Root Mean Squared Error (RMSE): A measure of the average deviation between the predicted and actual values.
            - R-squared (R2) Score: The proportion of the variance in the target variable that can be explained by the predictor.

    Examples:
        >>> y_true = [3, 5, 7, 9]
        >>> y_pred = [2.5, 5.1, 6.8, 8.9]
        >>> rmse, r2 = metrics_reg(y_true, y_pred)
        >>> print("RMSE:", rmse)
        >>> print("R2 Score:", r2)
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def select_kbest(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    
    return: a list of the selected features from the SelectKBest process
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    mask = kbest.get_support()
    return X.columns[mask]


def rfe(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    
    return: a list of the selected features from the recursive feature elimination process
        & a df of all rankings
    '''
    #MAKE the thing
    rfe = RFE(LinearRegression(), n_features_to_select=k)
    #FIT the thing
    rfe.fit(X, y)
        
    # use the thing
    features_to_use = X.columns[rfe.support_].tolist()
    
    # we need to send show_feature_rankings a trained/fit RFE object
    all_rankings = show_features_rankings(X, rfe)
    
    return features_to_use, all_rankings


def ols_mod(x_train_scaled, x_validate_scaled, y_train, y_validate):
    """
    Performs Ordinary Least Squares (OLS) regression using LinearRegression() from scikit-learn.

    Args:
        x_train_scaled (array-like): Scaled training input features.
        x_validate_scaled (array-like): Scaled validation input features.
        y_train (array-like): Training target variable.
        y_validate (array-like): Validation target variable.

    Returns:
        rmse (float): Root Mean Squared Error (RMSE) between the actual and predicted values on the validation set.
        r2 (float): R-squared value indicating the goodness of fit on the validation set.
    """

    #make it
    lr = LinearRegression()

    #fit it on our RFE features
    lr.fit(x_train_scaled, y_train)

    #use it (make predictions)
    pred_lr = lr.predict(x_train_scaled)

    #use it on validate
    pred_val_lr = lr.predict(x_validate_scaled)
    
    # validates
    rmse, r2 = metrics_reg(y_validate, pred_val_lr)


    return rmse, r2

def lars_mod(x_train_scaled, x_validate_scaled, y_train, y_validate):
    """
    Perform LARS (Least Angle Regression) modeling on the given data.

    Parameters:
        x_train_scaled (array-like): Scaled training features.
        x_validate_scaled (array-like): Scaled validation features.
        y_train (array-like): Training target variable.
        y_validate (array-like): Validation target variable.

    Returns:
        rmse (float): Root Mean Squared Error (RMSE) between the predicted and actual values on the validation set.
        r2 (float): R-squared value indicating the goodness of fit on the validation set.
    """
    #make it
    lars = LassoLars(alpha=1)

    #fit it on our RFE features
    lars.fit(x_train_scaled, y_train)

    #use it (make predictions)
    pred_lars = lars.predict(x_train_scaled)

    #use it on validate
    pred_val_lars = lars.predict(x_validate_scaled)
    
    # validates
    rmse, r2 = metrics_reg(y_validate, pred_val_lars)

    return rmse, r2

def poly_mod(x_train_scaled, x_validate_scaled, x_test_scaled, y_train, y_validate):
    """
    Fits a polynomial regression model of degree 3 using the provided scaled input features and target variables.

    Args:
        x_train_scaled (array-like): Scaled input features for training.
        x_validate_scaled (array-like): Scaled input features for validation.
        x_test_scaled (array-like): Scaled input features for testing.
        y_train (array-like): Target variables for training.
        y_validate (array-like): Target variables for validation.

    Returns:
        tuple: A tuple containing the root mean squared error (rmse) and the coefficient of determination (r2)
               for the validation set predictions.

    """
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    x_train_degree = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_degree = pf.transform(x_validate_scaled)
    x_test_degree = pf.transform(x_test_scaled)
    #make it
    pr = LinearRegression()

    #fit it
    pr.fit(x_train_degree, y_train)

    #use it
    pred_pr = pr.predict(x_train_degree)
    pred_val_pr = pr.predict(x_validate_degree)
    rmse, r2 = metrics_reg(y_validate, pred_val_pr)
    return rmse, r2

def glm_mod(x_train_scaled, x_validate_scaled, y_train, y_validate):
    """
    Fits a Generalized Linear Model (GLM) using the TweedieRegressor and returns the root mean squared error (RMSE)
    and R-squared (R2) for the validation dataset.

    Parameters:
        x_train_scaled (array-like): The scaled feature matrix for the training dataset.
        x_validate_scaled (array-like): The scaled feature matrix for the validation dataset.
        y_train (array-like): The target variable array for the training dataset.
        y_validate (array-like): The target variable array for the validation dataset.

    Returns:
        rmse (float): The root mean squared error (RMSE) for the validation dataset.
        r2 (float): The R-squared (R2) value for the validation dataset.
    """    
    #make it
    glm = TweedieRegressor(power=0, alpha=0)

    #fit it
    glm.fit(x_train_scaled, y_train)

    #use it
    pred_glm = glm.predict(x_train_scaled)
    pred_val_glm = glm.predict(x_validate_scaled)
    
    #validate
    rmse, r2 = metrics_reg(y_validate, pred_val_glm)
    
    return rmse, r2 

def best_model(x_train_scaled, x_test_scaled, y_train, y_test):
     """
    Trains and evaluates a polynomial regression model with the given data.

    Parameters:
        x_train_scaled (array-like): Scaled training input features.
        x_test_scaled (array-like): Scaled testing input features.
        y_train (array-like): Training target values.
        y_test (array-like): Testing target values.

    Returns:
        tuple: A tuple containing the root mean squared error (RMSE) and R-squared (R2) 
               values for the predictions made by the model on the testing data.
    """
    pf = PolynomialFeatures(degree=3)
    # fit and transform X_train_scaled
    x_train_degree = pf.fit_transform(x_train_scaled)
    # transformX_test_scaled
    x_test_degree = pf.transform(x_test_scaled)
    #make it
    pr = LinearRegression()
    #fit it
    pr.fit(x_train_degree, y_train)
    #use it
    pred_test = pr.predict(x_test_degree)

    rmse, r2 = smetrics_reg(y_test, pred_test)
    return rmse, r2