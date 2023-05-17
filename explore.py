import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, stats


def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    '''
    takes dataset, a categorical variable, continious variable and illustrates box plot, bar graph, and violin
    '''
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

def ind_test(samp1, samp2, alpha=0.05):
    '''
    Completes an sample t-test, based on the null hypothesis less than
    '''
    t, p = stats.ttest_ind(samp1, samp2, equal_var=False)

    if p/2 < alpha and t > 0 :
        print(f'''Reject the null hypothesis: Sufficient''')
    else:
        print(f''' Fail to reject the null: Insufficient evidence''')
    print(f" p-value: {p} , t: {t}")

def one_test(samp1, samp2, alpha=0.05):
    '''
    Completes an independent t-test, based on the null hypothesis less than
    '''
    t, p = stats.ttest_1samp(samp1, samp2)

    if p/2 < alpha and t > 0 :
        print(f'''Reject the null hypothesis: Sufficient''')
    else:
        print(f''' Fail to reject the null: Insufficient evidence''')
    print(f" p-value: {p} , t: {t}")

def create_relplot(df, x_var, y_var):
    '''
    Creates a relplot for input variables within the given dataset
    '''
    sns.set(style="white")
    # Create scatter plot
    sns.set_palette("pastel")
    ax = sns.relplot(x=x_var, y=y_var, data=df)

    # Add regression line
    sns.regplot(x=x_var, y=y_var, data=df, scatter=False, color='DarkSlateBlue')
    # ax.fig.suptitle(title, fontsize=10)
    plt.xlabel("Area (sqft)")
    plt.title("Relationship Between Area and Property Value")
    plt.ylabel("Property Value (USD)")
    plt.show()


def train_heat(train):
    '''
    Creates a heatmap of the variables within the zillow dataset
    '''

    # Increase the figure size to accommodate the heatmap
    plt.figure(figsize=(10, 8))
    # Correlation heat map
    sns.heatmap(train.corr(method='pearson'), cmap='YlGnBu', annot=True, fmt=".2f",
                mask=np.triu(train.corr(method='pearson')))
    # Adjust the font size of the annotations
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    # Show the plot
    
    plt.show()

def bed_bath_barplot(train):
    '''
    The creates a custom bar plot for comparing the avergae property value for 1 bathroom vs a 2 bedroom home
    '''
    one_bath = train[train.bathrooms == 1]
    one_bath["cmp"] = "1 Bathroom"

    two_bed = train[train.bedroom == 2]
    two_bed["cmp"] = "2 Bedrooms"
    tmp = pd.concat([one_bath,two_bed])

    # creat average line

    sns.set_palette("Pastel1")

    plt.figure(figsize=(8, 6))

    sns.barplot(x="cmp", y='property_value', data=tmp)
    # Create bars with different colors
    # create average line
    property_value_average = train.property_value.mean()
    plt.axhline(property_value_average, label="Property Value Average", color='DarkSlateBlue')

    plt.xlabel("Room Type")
    plt.title("One Bathroom Versus Two Bedroom")
    plt.ylabel("Property Value (USD)")
    plt.legend(loc='lower left')
    plt.show()

def bathroom_barplot(df):
    '''
    This function creates a custom bar chart for comparing homes with half bathrooms vs homes with only full bathrooms
    '''
    fig, ax =plt.subplots()

    sns.set_palette("Pastel1")
    plt.title("Single Family Properties with Full Bathrooms Cost Less on Average than Homes with Half Bathrooms")
    sns.barplot(x="full_bath", y="property_value", data=df)
    plt.xlabel("Full Bathrooms")
    plt.ylabel("Property Value (USD)")
    tick_label = ["No", "Yes"]
    ax.set_xticklabels(tick_label)
    # create average line
    property_value_average = df.property_value.mean()
    plt.axhline(property_value_average, label="Property Value Average", color='DarkSlateBlue')
    plt.legend()
    plt.show()

def pool_barplot(df):
    '''
    This function creates a custom bar chart for comparing homes with pools and homes without pools
    '''
    fig, ax =plt.subplots()
    # creat average line
  
    
    plt.title("Single Family Properties with Pools Cost More on Average than Homes without Pools")
    sns.barplot(x="pool", y="property_value", data=df)
    plt.xlabel("Pool")
    plt.ylabel("Property Value (USD)")
    tick_label = ["Without Pool", "With Pool"]
    ax.set_xticklabels(tick_label)
    property_value_average = df.property_value.mean()
    plt.axhline(property_value_average, label="Property Value Average", color='DarkSlateBlue')
    plt.legend()
    plt.show()

def county_barplot(df):
    '''
    This function creates a custom bar chart for comparing the property value for homes in the Orange, LA, and Ventura.
    '''
    fig, ax =plt.subplots()

    plt.title("Average Property Value for Single Family Properties by County")

    colors = ['#D8BFD8', '#66CDAA', '#FFDAB9']
    sns.set_palette(colors)

    sns.barplot(x="fips", y="property_value", data=df)

    plt.xlabel("County")
    plt.ylabel("Property Value (USD)")
    tick_label = ["Los Angeles", "Ventura", "Orange"]
    ax.set_xticklabels(tick_label)
    property_value_average = df.property_value.mean()
    plt.axhline(property_value_average, label="Property Value Average", color='DarkSlateBlue')
    plt.legend()

    plt.show()