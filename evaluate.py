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

import model as m

def split_into_xy(train, validate, test, target="property_value"):
    """
    Splits the input data into features (X) and target variable (y) for training, validation, and testing sets.

    Parameters:
        train (DataFrame): The training dataset containing both features and the target variable.
        validate (DataFrame): The validation dataset containing both features and the target variable.
        test (DataFrame): The testing dataset containing both features and the target variable.
        target (str, optional): The name of the target variable column in the datasets. Defaults to "property_value".

    Returns:
        x_train (DataFrame): The features of the training dataset.
        y_train (Series): The target variable of the training dataset.
        x_validate (DataFrame): The features of the validation dataset.
        y_validate (Series): The target variable of the validation dataset.
        x_test (DataFrame): The features of the testing dataset.
        y_test (Series): The target variable of the testing dataset.
    """
    # Split train data into X and y
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
    Compare the performance of a model with a baseline model using sum of squared errors.

    Parameters:
    y (numpy.ndarray): The actual target values.
    yhat (numpy.ndarray): The predicted target values by the model.
    y_baseline (numpy.ndarray): The predicted target values by the baseline model.

    Returns:
    None: This function does not return any value. It prints the comparison result.

    Notes:
    This function calculates the sum of squared errors (SSE) for both the model and the baseline
    predictions. It then compares the SSE values and prints a message indicating the performance
    comparison.

    - If the model's SSE is lower than the baseline's SSE, it prints "Your model performs better
      than the baseline model."
    - If the model's SSE is higher than the baseline's SSE, it prints "Your model does not perform
      better than the baseline model."
    - If the SSE values are equal, it prints "Your model performs equally to the baseline model."
    """
    model_sse = np.sum((y - yhat) ** 2)
    baseline_sse = np.sum((y - y_baseline) ** 2)

    if model_sse < baseline_sse:
        print("Your model performs better than the baseline model.")
    elif model_sse > baseline_sse:
        print("Your model does not perform better than the baseline model.")
    else:
        print("Your model performs equally to the baseline model.")

def plot_residuals(y, yhat):
    """
    Create a residual plot to visualize the differences between the true target values and the predicted values.

    Parameters:
        y (array-like): The true target values.
        yhat (array-like): The predicted target values.

    Returns:
        None

    This function calculates the residuals as the differences between the true target values (y) and the predicted
    values (yhat). It then creates a scatter plot of the residuals against the predicted values. A horizontal line at y=0
    is added to the plot to help visualize the zero residual line. The plot includes labels for the x and y axes, as
    well as a title to describe the plot as the "Residual Plot." The plot is displayed using the plt.show() function.
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
    Calculate various regression error metrics.

    Parameters:
        y (numpy array): The true target values.
        yhat (numpy array): The predicted target values.

    Returns:
        tuple: A tuple containing the following regression error metrics:
            - Sum of Squared Errors (SSE)
            - Explained Sum of Squares (ESS)
            - Total Sum of Squares (TSS)
            - Mean Squared Error (MSE)
            - Root Mean Squared Error (RMSE)
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

    # Print the calculated metrics
    print("Sum of Squared Errors (SSE):", sse)
    print("Explained Sum of Squares (ESS):", ess)
    print("Total Sum of Squares (TSS):", tss)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Return the calculated metrics as a tuple
    return sse, ess, tss, mse, rmse


def baseline_mean_errors(y):
    '''
    Calculate various error metrics for a baseline model.

    This function calculates the sum of squared errors (SSE), mean squared error (MSE), and root mean squared error (RMSE)
    for a given set of observations y, considering the mean of y as the baseline prediction.

    Parameters:
    -----------
    y : array-like
        The observed values.

    Returns:
    --------
    tuple
        A tuple containing three error metrics (sse, mse, rmse) as floats.

    Examples:
    ---------
    >>> y = [2, 3, 4, 5, 6]
    >>> sse, mse, rmse = baseline_mean_errors(y)
    >>> print(sse)
    2.0
    >>> print(mse)
    0.4
    >>> print(rmse)
    0.6324555320336759
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
    Compare the performance of a model to a baseline by checking if the model's sum of squared errors (SSE) is lower than
    the baseline's SSE.

    Parameters:
        y (array-like): The true target values.
        yhat (array-like): The predicted target values by the model.

    Returns:
        bool: True if the model's SSE is lower than the baseline's SSE, otherwise False.
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
    Compare the performance metrics of a simple linear regression model and a baseline model using the Sum of Squared Errors (SSE),
    Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Explained Sum of Squares (ESS).

    Parameters:
        df (DataFrame): The input pandas DataFrame containing the data.
        x (str): The name of the column representing the independent variable (predictor).
        y (str): The name of the column representing the dependent variable (target).

    Returns:
        DataFrame: A pandas DataFrame with performance metrics for the models.

    Note:
        This function fits a simple linear regression model using the provided `x` and `y` columns in the DataFrame `df`.
        It then computes the SSE, MSE, and RMSE for both the linear regression model and a baseline model, where the baseline
        model uses the mean of the `y` values as the prediction for all data points.

    Example:
        import pandas as pd

        # Sample data
        data = {'x': [1, 2, 3, 4, 5],
                'y': [2, 4, 5, 4, 5]}
        df = pd.DataFrame(data)

        # Compare the models
        metrics_df = compare_sse(df, 'x', 'y')
        print(metrics_df)
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
    