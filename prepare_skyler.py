#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######### IMPORTS ##############

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

####### AQUIRE / PREP ###########

def unique_rows(df):
    """
    Returns the input dataframe with only unique rows.
    """
    return df.drop_duplicates()


def get_df():
    '''
    get_df pulls red and white wine df's from data.world url's,
    creates a new column for wine color, concats them and returns 
    them as a single pandas df with only unique rows.
    '''
    # Pull in red and white wine df's
    df_white = pd.read_csv(
        'https://query.data.world/s/lmm2oc73ncl233bsk4m4pergggesuf?dws=00000')
    df_red = pd.read_csv(
        'https://query.data.world/s/xvy3biopsnfrfxfgialtawp6v477mk?dws=00000')
    # add column for wine color before merging
    df_white['color']= 'white'
    df_red['color']= 'red'
    #concat them and return new df
    df= pd.concat([df_white, df_red], ignore_index=True)
    # remove duplicates and return only unique rows 
    df = unique_rows(df)
    
    return df

############ SPLIT ###############

def train_val_test_split(df):
    '''
    takes in a pandas df, then splits between train, validate,
    and test subsets. 56/24/20 split
    '''
    train_validate, test=train_test_split(df, 
                                 train_size=.8, 
                                 random_state=828)
    
    train, validate =train_test_split(train_validate, 
                                      test_size=.3, 
                                      random_state=828)
    return train, validate, test

############## SCALE #############

def scale_wine(df):
    #identify scaler
    scaler=sklearn.preprocessing.MinMaxScaler()
    #identify cols to scale
    col=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
          'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    #fit cols 
    sclaer.fit(df[col])
    df[col]=scaler.transform(df[col])
    return df

