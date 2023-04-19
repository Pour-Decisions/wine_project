#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder




def corr_map(df):
    '''
    Creates a correlation matrix heatmap of the top correlated columns in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    '''
    # Encode the 'quality' column using LabelEncoder
    class_tp = LabelEncoder()

    class_ql = {'low':0, 'medium': 1, 'high': 2}
    y_ql = df.quality.map(class_ql)

    # Compute the correlation matrix of the DataFrame
    corr = df.corr()

    # Identify the top correlated columns (excluding 'color')
    top_corr_cols = corr.quality.sort_values(ascending=False).keys()

    # Extract the submatrix of the top correlated columns
    top_corr = corr.loc[top_corr_cols, top_corr_cols]

    # Create a mask to hide the upper triangle of the heatmap
    dropSelf = np.zeros_like(top_corr)
    dropSelf[np.triu_indices_from(dropSelf)] = True

    # Plot the correlation matrix heatmap
    plt.figure(figsize=(18, 10))
    sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)
    sns.set(font_scale=1.5)
    plt.show()
    
    # Delete temporary variables
    del corr, dropSelf, top_corr


def target_dist(df):
    '''
    target_dist takes in a pandas dataframe, then proceeds to plot
    the distribution of the target variable, quality rating.
    '''
    # create histogram
    sns.histplot(x='quality', data=df, color='cyan', binwidth=0.5)
    # calculate mean
    mean_quality = df['quality'].mean()
    # add mean line
    plt.axvline(mean_quality, color='red', linestyle='--', label='Mean Quality')
    # add mean value as text label
    plt.text(mean_quality + .8, 1000, f'Mean Quality = {mean_quality:.2f}', 
             fontsize=10, color='red')
    # add labels and title
    plt.xlabel('Quality Rating')
    plt.title('Visualizing the Target Variable')
    # add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.5, linestyle='--')
    # show plot
    plt.show()

def abv_plots(train):
    '''
    abv_plots takes in train data and produces a histplot comparing the mean ABV of 
    high quality wines versus lower quality wines.
    '''
    # identify low and high quality wines
    high= train[train['quality'] >= 7]
    low= train[train['quality'] <= 5]
    #define figure size
    plt.figure(figsize=(10, 5))
    # set title
    plt.title('High and Low Quality Wine ABV Compared')
    # identify low and high rated means
    mean_low_qual = low['alcohol'].mean()
    mean_high_qual = high['alcohol'].mean()
    # plot data
    sns.histplot(x='alcohol', data=low, alpha=.5, label= 'Lower Quality', color='orange')
    sns.histplot(x='alcohol', data=high, alpha=.3, label='High Quality', color='magenta')
    plt.xlabel(' % Alcohol by Volume')
    # draw low quality mean line
    plt.axvline(x=mean_low_qual, label='Lower Quality Mean ABV', color='orange')
    plt.text(mean_low_qual + 1.8, 70, f'Lower Quality Mean ABV = {mean_low_qual:.1f}%', 
             fontsize=10, color='red')
    # draw high quality mean line
    plt.axvline(x= mean_high_qual, label='High Quality Mean ABV', color='magenta')
    plt.text(mean_high_qual + .2, 80, f'High Quality Mean ABV= {mean_high_qual:.1f}%', 
             fontsize=10, color='red')
    # produce legen
    plt.legend()
    # show gridlines for easy reading
    plt.grid(True, alpha=0.75, linestyle='--')
    # show the vis
    plt.show()
    
def abv_tstat(train):
    '''
    abv_tstat takes in train data, the performs a two-sample t-test 
    comparing high quality and lower quality wines based on ABV.
    '''
    # create two samples
    high_quality = train[train['quality']>= 7]['alcohol']
    low_quality = train[train['quality'] <= 5]['alcohol']

    # perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(high_quality, low_quality, equal_var=True)

    # print results
    print(f"t-statistic: {t_statistic:.2f}")
    print(f"p-value: {p_value:.4f}")    
    

def chlor_plots(train):
    '''
    chlor_plots takes in train data and visualizes chloride levels
    between high wuality wines(7+) and lower quality wines(6 and below)
    '''
    # identify low and high quality wines
    high= train[train['quality'] >= 7]
    low= train[train['quality'] <= 5]
    # identify low and high rated means
    mean_low_chlor = low['chlorides'].mean()
    mean_high_chlor = high['chlorides'].mean()
    # set figure size
    plt.figure(figsize=(10,5))
    # set title
    plt.title('Chloride Levels Compared')
    # plot data
    sns.histplot(x='chlorides', data=high, alpha=1, color = 'cyan', label='High Quality Wines')
    sns.histplot(x='chlorides', data=low, alpha=.4, color = 'magenta', label='Lower Quality Wines')
    # x axis label
    plt.xlabel('Chloride Levels (g / dm^3)')
    # draw mean lines and text
    plt.axvline(x=mean_high_chlor, color='cyan', label='High Quality Mean Chloride')
    plt.text(mean_high_chlor + .058, 125, f'High Quality Mean Chloride= {mean_high_chlor:.2f}%', 
             fontsize=10, color='red')
    # draw mean lines and text
    plt.axvline(x=mean_low_chlor, color='magenta', label='Lower Quality Mean Chloride')
    plt.text(mean_low_chlor + .035, 100, f'Lower Quality Mean Chloride = {mean_low_chlor:.2f}%', 
             fontsize=10, color='red')
    # add legend
    plt.legend()
    # add gridlines
    plt.grid(True, alpha=0.5, linestyle='--')
    # set the x-axis limits to 0 and 0.1
    plt.xlim(0,0.16)  
    plt.show()    


def chlor_stat(train):
    '''
    chlor_stat takes in train data, the performs a two-sample t-test 
    comparing high quality and lower quality wines based on chloride level.
    '''
    # create two samples
    high_quality = train[train['quality']>= 7]['chlorides']
    low_quality = train[train['quality'] <= 5]['chlorides']

    # perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(high_quality, low_quality, equal_var=True)

    # print results
    print(f"t-statistic: {t_statistic:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    
def sugar_plots(train):
    '''
    chlor_plots takes in train data and visualizes residual sugar levels
    between high quality wines(7+) and lower quality wines(6 and below)
    '''
    # identify low and high quality wines
    high= train[train['quality'] >= 7]
    low= train[train['quality'] <= 5]
    # identify low and high rated means
    mean_low_sugar = low['residual sugar'].mean()
    mean_high_sugar = high['residual sugar'].mean()
    # set figure size
    plt.figure(figsize=(10,5))
    # set title
    plt.title('Residual Sugar Levels Compared')
    # plot data
    sns.histplot(x='residual sugar', data=high, alpha=.8, color = 'cyan', label='High Quality Wines')
    sns.histplot(x='residual sugar', data=low, alpha=.4, color = 'magenta', label='Lower Quality Wines')
    # x axis label
    plt.xlabel('Residual Sugar Levels (g / dm^3)')
    # draw mean lines and text
    plt.axvline(x=mean_high_sugar, color='cyan', label='High Quality Mean Residual Sugar')
    plt.text(mean_high_sugar + 7, 220, f'High Quality Mean Residual Sugar= {mean_high_sugar:.2f}%', 
             fontsize=10, color='red')
    # draw mean lines and text
    plt.axvline(x=mean_low_sugar, color='magenta', label='Lower Quality Mean Residual Sugar')
    plt.text(mean_low_sugar + 6, 180, f'Lower Quality Mean Residual Sugar = {mean_low_sugar:.2f}%', 
             fontsize=10, color='red')
    # add legend
    plt.legend()
    # add gridlines
    plt.grid(True, alpha=0.5, linestyle='--')
    # set the x-axis limits to 0 and 0.1
    plt.xlim(0, 20)  
    plt.show()
    
    
def sugar_stat(train):
    '''
    sugar_stat takes in train data, the performs a two-sample t-test 
    comparing high quality and lower quality wines based on residual sugar level.
    '''
    # create two samples
    high_quality = train[train['quality']>= 7]['residual sugar']
    low_quality = train[train['quality'] <= 5]['residual sugar']

    # perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(low_quality, high_quality, equal_var=True)

    # print results
    print(f"t-statistic: {t_statistic:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    
def vol_acid_plots(train):
    '''
    chlor_plots takes in train data and visualizes volatile acidity levels
    between high quality wines(7+) and lower quality wines(6 and below)
    '''
    # identify low and high quality wines
    high= train[train['quality'] >= 7]
    low= train[train['quality'] <= 5]
    # identify low and high rated means
    mean_low_vol = low['volatile acidity'].mean()
    mean_high_vol = high['volatile acidity'].mean()
    # set figure size
    plt.figure(figsize=(10,5))
    # set title
    plt.title('Volatile Acidity Levels Compared')
    # plot data
    sns.histplot(x='volatile acidity', data=high, alpha=.8, color = 'cyan', label='High Quality Wines')
    sns.histplot(x='volatile acidity', data=low, alpha=.4, color = 'magenta', label='Lower Quality Wines')
    # x axis label
    plt.xlabel('Volatile Acidity Levels (g / dm^3)')
    # draw mean lines and text
    plt.axvline(x=mean_high_vol, color='cyan', label='High Quality Mean Volatile Acidity')
    plt.text(mean_high_vol + .3, 85, f'High Quality Mean Volatile Acidity= {mean_high_vol:.2f}%', 
             fontsize=10, color='red')
    # draw mean lines and text
    plt.axvline(x=mean_low_vol, color='magenta', label='Lower Quality Mean Volatile Acidity')
    plt.text(mean_low_vol + .18, 70, f'Lower Quality Mean Volatile Acidity = {mean_low_vol:.2f}%', 
             fontsize=10, color='red')
    # add legend
    plt.legend()
    # add gridlines
    plt.grid(True, alpha=0.5, linestyle='--')
    # set the x-axis limits to 0 and 0.1
    plt.xlim(0, 1.2)  
    plt.show()
    

def vol_stat(train):
    '''
    vol_stat takes in train data, the performs a two-sample t-test 
    comparing high quality and lower quality wines based on volatile acidity.
    '''
    # create two samples
    high_quality = train[train['quality']>= 7]['volatile acidity']
    low_quality = train[train['quality'] <= 5]['volatile acidity']

    # perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(low_quality, high_quality, equal_var=True)

    # print results
    print(f"t-statistic: {t_statistic:.2f}")
    print(f"p-value: {p_value:.4f}")


def col_stat(train):
    '''
    col_stat takes in train data and performs a two-sample t-test 
    comparing high quality and lower quality wines based on each column.
    '''
    # create two samples
    high_quality = train[train['quality']>= 7]
    low_quality = train[train['quality'] <= 5]
    
    # iterate over each column in the dataframe
    for col in train.columns:
        if col == 'quality' or col == 'color':
            continue
        else:
            # perform two-sample t-test
            t_statistic, p_value = stats.ttest_ind(low_quality[col], high_quality[col], equal_var=True)

            # print results
            print(f"{col} - t-statistic: {t_statistic:.2f}")
            print(f"{col} - p-value: {p_value:.4f}") 
            

    
    
def clust_vis(df):
    '''
    clust_vis takes in wine df, makes a copy, then plots a cluster visualization
    '''
    vis_df = df.copy()
    vis_df['quality_label'] = vis_df.quality.apply(lambda q: 'low' if q <= 5 else 'medium' if q <= 7 else 'high')
    #wines.quality_label = pd.Categorical(wines.quality_label, categories=['low', 'medium', 'high'], ordered=True)

    # re-shuffle records just to randomize data points
    vis_df = vis_df.sample(frac=1, random_state=101).reset_index(drop=True)
    # Define a dictionary that maps the quality levels to numerical values
    quality_map = {'low': 0, 'medium': 1, 'high': 2}

    # Create a new column in the DataFrame that maps the quality values to numerical values
    vis_df['quality_num'] = vis_df['quality_label'].map(quality_map)
    
    g = sns.FacetGrid(vis_df, col='color', hue='quality_label', col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],
                  aspect=1.2, size=2.5, palette=sns.light_palette('navy', 3))
    g.map(plt.scatter, 'density', 'residual sugar', alpha=0.9, edgecolor='white', linewidth=0.5)
    fig = g.fig
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle('Wine Type - Density - Residual Sugar', fontsize=14)
    l = g.add_legend(title='Wine Quality Class')

    g = sns.FacetGrid(vis_df, col='color', hue='quality_label', col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],
                      aspect=1.2, size=2.5, palette=sns.light_palette('green', 3))
    g.map(plt.scatter, "density", "volatile acidity", alpha=0.9, edgecolor='white', linewidth=0.5)
    fig = g.fig 
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle('Wine Type - Density - Volitile Acidity', fontsize=14)
    l = g.add_legend(title='Wine Quality Class')

    g = sns.FacetGrid(vis_df, col='color', hue='quality_label', col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],
                      aspect=1.2, size=2.5, palette=sns.light_palette('lightcoral', 3))
    g.map(plt.scatter, "total sulfur dioxide", "residual sugar", alpha=0.9, edgecolor='white', linewidth=0.5)
    fig = g.fig 
    fig.subplots_adjust(top=0.8, wspace=0.3)
    fig.suptitle('Wine Type - Residual Sugar - Total Sulfur Dioxide', fontsize=14)
    l = g.add_legend(title='Wine Quality Class')
    
    plt.show()
    
