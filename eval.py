import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from math import sqrt

def split_into_xy(train, validate, test, target="property_value"):
    """
    Split the input data into feature variables (X) and target variable (y) for training, validation, and testing sets.

    Parameters:
        train (DataFrame): The training data containing both features and target variable.
        validate (DataFrame): The validation data containing both features and target variable.
        test (DataFrame): The testing data containing both features and target variable.
        target (str, optional): The name of the target variable column. Defaults to "property_value".

    Returns:
        tuple: A tuple containing six DataFrames (x_train, y_train, x_validate, y_validate, x_test, y_test).
            x_train (DataFrame): The feature variables of the training set.
            y_train (DataFrame): The target variable of the training set.
            x_validate (DataFrame): The feature variables of the validation set.
            y_validate (DataFrame): The target variable of the validation set.
            x_test (DataFrame): The feature variables of the testing set.
            y_test (DataFrame): The target variable of the testing set.
    """
    x_train = train.drop(target, axis=1)
    y_train = train[target]

    # Split validate data into X and y
    x_validate = validate.drop(target, axis=1)
    y_validate = validate[target]

    # Split test data into X and y
    x_test = test.drop(target, axis=1)
    y_test = test[target]

    return x_train, y_train, x_validate, y_validate, x_test, y_test


def get_yhat(train, x, y):
    '''
    Will fit your train DataFrame, x, y on LinerRegression()

    Returns:
    pandas.DataFrame with two columns baseline on y and yhat predictions
    '''
    model = LinearRegression().fit(train[[x]], train[y])
    predictions = model.predict(train[[x]])
    train['baseline'] = train[y].mean()
    train['yhat'] = predictions
    return train

def compare_models(y, yhat, y_baseline):
    """
    Compare two models based on the sum of squared errors (SSE) between their predictions and actual values.

    Parameters:
        y (numpy array): The true target values.
        yhat (numpy array): The predicted values from the model being compared.
        y_baseline (numpy array): The predicted values from the baseline model for comparison.

    Returns:
        None: This function doesn't return anything. It only prints the comparison result.

    Example:
        y = np.array([3, 5, 2, 8, 6])
        yhat = np.array([2, 4, 3, 7, 5])
        y_baseline = np.array([4, 3, 2, 6, 5])
        compare_models(y, yhat, y_baseline)
        # Output: Your model performs better than the baseline model.
    """
    # Calculate the sum of squared errors for the model and the baseline
    model_sse = np.sum((y - yhat) ** 2)
    baseline_sse = np.sum((y - y_baseline) ** 2)

    # Compare the SSE values and output the result
    if model_sse < baseline_sse:
        print("Your model performs better than the baseline model.")
    elif model_sse > baseline_sse:
        print("Your model does not perform better than the baseline model.")
    else:
        print("Your model performs equally to the baseline model.")


def plot_residuals(y, yhat):
    """Plot the residuals of a regression model.

    This function calculates the residuals by taking the difference between the true
    target values (y) and the predicted values (yhat) from a regression model. It then
    creates a scatter plot of the residuals against the predicted values, and adds a
    horizontal line at y=0 for reference.

    Parameters:
        y (array-like): The true target values.
        yhat (array-like): The predicted target values from a regression model.

    Returns:
        None: The function only plots the residuals scatter plot and does not return any value.

    Example:
        >>> y = [1, 2, 3, 4, 5]
        >>> yhat = [1.2, 1.8, 2.7, 3.9, 4.5]
        >>> plot_residuals(y, yhat)
    """
    # Calculate residuals
    residuals = y - yhat

    # Create a scatter plot of residuals
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0

    # Set plot labels and title
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    # Show the plot
    plt.show()


def regression_errors(y, yhat):
    """
    Calculate various regression error metrics given the true values and predicted values.

    Parameters:
        y (numpy.ndarray): Array of true target values.
        yhat (numpy.ndarray): Array of predicted target values.

    Returns:
        tuple: A tuple containing the following regression error metrics:
            - sse (float): Sum of Squared Errors (SSE).
            - ess (float): Explained Sum of Squares (ESS).
            - tss (float): Total Sum of Squares (TSS).
            - mse (float): Mean Squared Error (MSE).
            - rmse (float): Root Mean Squared Error (RMSE).
    """
    # Calculate the squared errors
    squared_errors = (y - yhat) ** 2

    # Calculate the sum of squared errors (SSE)
    sse = np.sum(squared_errors)

    # Calculate the explained sum of squares (ESS)
    ess = np.sum((yhat - np.mean(y)) ** 2)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((y - np.mean(y)) ** 2)

    # Calculate the mean squared error (MSE)
    mse = np.mean(squared_errors)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    print("Sum of Squared Errors (SSE):", sse)
    print("Explained Sum of Squares (ESS):", ess)
    print("Total Sum of Squares (TSS):", tss)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    return sse, ess, tss, mse, rmse


def baseline_mean_errors(y):
    '''
    Calculate baseline mean errors for a given dataset.

    This function takes an array or list of numerical values 'y' and calculates the following error metrics:
    - Sum of Squared Errors (SSE): The sum of the squared differences between each value in 'y' and its mean.
    - Mean Squared Error (MSE): The average of squared errors, which is the SSE divided by the number of data points.
    - Root Mean Squared Error (RMSE): The square root of the MSE, providing a measure of the typical error magnitude.

    Parameters:
        y (array-like): A 1-dimensional array or list of numerical values.

    Returns:
        tuple: A tuple containing three error metrics - SSE, MSE, and RMSE.

    Example:
        >>> y = [1, 2, 3, 4, 5]
        >>> sse, mse, rmse = baseline_mean_errors(y)
        >>> sse
        10
        >>> mse
        2.0
        >>> rmse
        1.4142135623730951
    '''
    # Calculate the mean of y
    y_mean = np.mean(y)

    # Calculate the squared errors
    squared_errors = (y - y_mean) ** 2

    # Calculate the sum of squared errors (SSE)
    sse = np.sum(squared_errors)

    # Calculate the mean squared error (MSE)
    mse = np.mean(squared_errors)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    return sse, mse, rmse


def better_than_baseline(y, yhat):
    """
    Determine if a model's performance is better than a baseline.

    This function compares the sum of squared errors (SSE) between the provided model's predictions (yhat) 
    and the actual target values (y) against the SSE of a simple baseline prediction, which is the mean of y.
    If the model's SSE is lower than the baseline's SSE, it is considered better.

    Parameters:
    y (numpy.ndarray): The actual target values.
    yhat (numpy.ndarray): The predicted values from the model.

    Returns:
    bool: True if the model's SSE is lower than the baseline's SSE, otherwise False.

    Example:
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> yhat = np.array([1.2, 1.8, 3.2, 4.1, 5.3])
    >>> better_than_baseline(y, yhat)
    True

    In this example, the model's predictions have a lower SSE than the baseline's SSE, so the function returns True.
    """
    # Calculate the mean of y
    y_mean = np.mean(y)

    # Calculate the sum of squared errors for the model and the baseline
    model_sse = np.sum((y - yhat) ** 2)
    baseline_sse = np.sum((y - y_mean) ** 2)

    # Check if the model's SSE is lower than the baseline's SSE
    if model_sse < baseline_sse:
        return True
    else:
        return False

def compare_sse(df, x, y):
    """
    Compare the performance metrics of a simple linear regression model with a baseline model.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        x (str): The name of the column representing the independent variable (feature).
        y (str): The name of the column representing the dependent variable (target).

    Returns:
        pandas.DataFrame: A DataFrame containing the following metrics for both the linear regression model and the baseline:
            - 'metric': The name of the evaluation metric (SSE, MSE, RMSE, SSE_baseline, MSE_baseline, or RMSE_baseline).
            - 'model_error': The corresponding value of the evaluation metric for the linear regression model and baseline.
    """
    # create baseline
    df['yhat_baseline'] = df[y].mean()
    
    # creating simplae model 
    lr = LinearRegression()
    ols_model = lr.fit(df[[x]], df[y])
    df['yhat'] = ols_model.predict(df[[x]])
    
    # compute SSE
    SSE = mean_squared_error(df[y], df.yhat)*len(df)
    SSE_baseline = mean_squared_error(df[y], df.yhat_baseline)*len(df)
    
    # compute MSE
    MSE = mean_squared_error(df[y], df.yhat)
    MSE_baseline = mean_squared_error(df[y], df.yhat_baseline)
    
    # compute RMSE
    RMSE = sqrt(mean_squared_error(df[y], df.yhat))
    RMSE_baseline = sqrt(mean_squared_error(df[y], df.yhat_baseline))
    
    # compute ESS
    ESS = sum((df.yhat - df[y].mean())**2)
    
    # create dataframe
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE', 'SSE_baseline','MSE_baseline','RMSE_baseline']), columns=['metric'])

    df_eval['model_error'] = np.array([SSE, MSE, RMSE, SSE_baseline, MSE_baseline, RMSE_baseline])
    
    return pd.DataFrame(df_eval)