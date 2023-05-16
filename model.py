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

def get_baseline(df, y):
    df["yhat_baseline"] = df[y].mean()
    return df

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
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

def show_features_rankings(X_train, rfe):
    """
    Takes in a dataframe and a fit RFE object in order to output the rank of all features
    """
    # Dataframe of rankings
    ranks = pd.DataFrame({'rfe_ranking': rfe.ranking_}
                        ,index = X_train.columns)
    
    ranks = ranks.sort_values(by="rfe_ranking", ascending=True)
    
    return ranks

def get_baseline(y_train):
    x_train_scaled["yhat_baseline"] = y_train.mean()
    return df

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

    rmse, r2 = metrics_reg(y_test, pred_test)
    return rmse, r2