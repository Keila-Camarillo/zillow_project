import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, stats


def plot_variable_pairs(df):
    sns.pairplot(df, kind="reg")
    plt.show()

def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 6))

    # Bar plot
    sns.barplot(x=categorical_var, y=continuous_var, data=df, ax=ax1)
    
    ax1.tick_params(axis='x', rotation=45)

    # Box plot
    sns.boxplot(x=categorical_var, y=continuous_var, data=df, ax=ax2)
    ax2.tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(x=categorical_var, y=continuous_var, data=df, ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.show()

def ind_test(samp1, samp2,  null_hypothesis, alternative_hypothesis, alpha=0.05):
    t, p = stats.ttest_ind(samp1, samp2, equal_var=False)

    if p/2 < alpha and t > 0 :
        print(f'''
    - Reject the null hypothesis: {null_hypothesis}
    - Sufficient evidence to move forward understanding that, {alternative_hypothesis}
        ''')
    else:
        print(f'''
        Fail to reject the null: Insufficient evidence
        ''')
    print(f" p-value: {p} , t: {t}")

def one_test(samp1, samp2,  null_hypothesis, alternative_hypothesis, alpha=0.05):
    t, p = stats.ttest_1samp(samp1, samp2)

    if p/2 < alpha and t > 0 :
        print(f'''
    - Reject the null hypothesis: {null_hypothesis}
    - Sufficient evidence to move forward understanding that, {alternative_hypothesis}
        ''')
    else:
        print(f'''
        Fail to reject the null: Insufficient evidence
        ''')
    print(f" p-value: {p} , t: {t}")

def create_relplot(df, x_var, y_var, title):
    sns.set(style="white")
    # Create scatter plot
    ax = sns.relplot(x=x_var, y=y_var, data=df, color="plum")

    # Add regression line
    sns.regplot(x=x_var, y=y_var, data=df, scatter=False, color='darkred')
    ax.fig.suptitle(title, fontsize=10)
  
    plt.show()
