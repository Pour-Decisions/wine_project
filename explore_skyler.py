#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


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


    

