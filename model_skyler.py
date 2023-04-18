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
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt 

def combined_df(df, f1, f2):
    '''
    This function calls another function in explore.py 
    and merges a column to the original dataset
    '''
    
    X = clustering(df, f1, f2)
    
    scaled_clusters = X['scaled_clusters']
    df = pd.merge(df, scaled_clusters, left_index=True, right_index=True)
    
    return df


def cluster_relplot(df, f1, f2):
    '''
    this functions creates a relplot of the clusters
    '''
    
    sns.set(style = "whitegrid")
    
    X = clustering(df, f1, f2)
    
    sns.relplot(data = X, x = f1, y = f2, hue = 'scaled_clusters')
    
    plt.title('Clusters')
    
    return plt.show()



def clustering(train, f1, f2):
    '''
    clustering fits and predicts Kmeans clustering to the input featues of the
    dataset
    '''
    # set random seed
    seed = 828
    # define 'X'
    X = train[[f1, f2]]
    # fit the thing
    kmeans = KMeans(n_clusters = 3, random_state= seed)
    kmeans.fit(X)
    kmeans.predict(X)
    # scale features
    mm_scaler = MinMaxScaler()
    X[[f1, f2]] = mm_scaler.fit_transform(X[[f1, f2]])
    # predict
    kmeans_scale = KMeans(n_clusters = 3, random_state = 828)
    kmeans_scale.fit(X[[f1, f2]])
    kmeans_scale.predict(X[[f1, f2]])
    # add predictions to a new column
    X['scaled_clusters'] = kmeans_scale.predict(X[[f1, f2]])
    
    return X  

def best_cluster(train, f1, f2):
    '''
    best_cluster takes in the data set, and the two feautures to cluster,
    then makes a graph to show the most optimal cluster number.
    '''
    # define 'X'
    X = clustering(train, f1, f2)
    # empty list
    inertia = []
    seed = 828 
    # for loop to test different number of clusters
    for n in range(1,11):

        kmeans = KMeans(n_clusters = n, random_state = seed)

        kmeans.fit(X[[f1, f2]])

        inertia.append(kmeans.inertia_)
        
    # append reults to new df    
    results_df = pd.DataFrame({'n_clusters': list(range(1,11)),
                               'inertia': inertia})   
    # plot the reults
    sns.set_style("whitegrid")
    sns.relplot(data = results_df, x='n_clusters', y = 'inertia', kind = 'line')
    plt.xticks(np.arange(0, 11, step=1))
    point = (3, 62) # specify the x and y values of the point to annotate
    plt.annotate("optimal cluster", xy=point, xytext=(3.2, 75), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    # title
    plt.title('Clusters Versus Inertia')
    
    return plt.show()
    
    return plt.show()


def find_best_features(df, k_min, k_max):
    '''
    This function takes in a dataframe,  a minimum number 
    of clusters (k_min), and a maximum number 
    of clusters (k_max). It returns a list of the best features 
    to use for clustering based on the Silhouette score.
    '''
    # Remove the target column from the dataframe
    X = df.drop(['quality', 'color'], axis=1)
    
    # Create an empty list to store the best features
    best_features = []
    
    # Loop through the range of k values
    for k in range(k_min, k_max+1):
        # Fit a KMeans clustering model
        kmeans = KMeans(n_clusters=k, random_state=828).fit(X)
        
        # Calculate the Silhouette score for the clustering
        score = silhouette_score(X, kmeans.labels_)
        
        # If this is the first iteration, add all features to the best_features list
        if k == k_min:
            best_features = list(X.columns)
        else:
            # For each subsequent iteration, compare the Silhouette score to the previous iteration
            # If the score has improved, update the best_features list with the current set of features
            if score > prev_score:
                best_features = list(X.columns)
        
        # Set the previous score to the current score for the next iteration
        prev_score = score
    
    # Return the list of best features
    return best_features



def scale_data(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    columns_scale = train.iloc[:, :11]
    columns_to_scale = columns_scale.columns
    
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    mms = sklearn.preprocessing.MinMaxScaler()
    #     fit the thing
    mms.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(mms.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(mms.transform(validate[columns_to_scale]), 
                                                     columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mms.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


def splitting_subsets(train, train_scaled, validate_scaled, test_scaled):
    '''
    This function splits our train, validate, and test scaled datasets into X/y train,
    validate, and test subsets
    '''
    
    
    X_train = train_scaled.drop(columns = ['quality'])
    X_train = pd.get_dummies(X_train, columns = ['color', 'scaled_clusters'])
    y_train = train_scaled['quality']


    X_validate = validate_scaled.drop(columns = ['quality'])
    X_validate = pd.get_dummies(X_validate, columns = ['color', 'scaled_clusters'])
    y_validate = validate_scaled['quality']


    X_test = test_scaled.drop(columns = ['quality'])
    X_test = pd.get_dummies(X_test, columns = ['color', 'scaled_clusters'])
    y_test = test_scaled['quality']

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_baseline(y_train):
    '''
    This function takes in y_train, then calculates the baseline RMSE
    '''
    
    preds_df = pd.DataFrame({'actual': y_train})
    
    preds_df['baseline'] = y_train.mean()
    
    baseline_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.baseline))

    return baseline_rmse


def linear_model(X_train, y_train):
    '''
    This function makes a linear regression model, fits, and predicts the output values.
    Giving us a dataframe of predicted linear and actual values
    '''
    
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_preds = lm.predict(X_train)
    
    preds_df = pd.DataFrame({'actual': y_train,'lm_preds': lm_preds})
    
    lm_rmse = sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    
    df = pd.DataFrame({'model': 'linear', 'linear_rmse': lm_rmse},index=['0']) 
                      
    return df


def lasso_lars(X_train, y_train):
    
    '''
        This function is used to run a for loop on lasso lars. We will use the best preforming model,
        and use it on the validate datasets.
    '''
    
    metrics = []

    for i in np.arange(0.05, 1, .05):
    
        lasso = LassoLars(alpha = i )
    
        lasso.fit(X_train, y_train)
    
        lasso_preds = lasso.predict(X_train)
        
        preds_df = pd.DataFrame({'actual': y_train})
    
        preds_df['lasso_preds'] = lasso_preds

        lasso_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    
        output = {
                'alpha': i,
                'lasso_rmse': lasso_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('lasso_rmse')


def tweedie_models(X_train, y_train):
    
    '''
    This function is used to run a for loop on tweedie model. We will use the best preforming model,
    and use it on the validate datasets.
    '''
    
    metrics = []

    for i in range(0, 4, 1):
    
        tweedie = TweedieRegressor(power = i)
    
        tweedie.fit(X_train, y_train)
    
        tweedie_preds = tweedie.predict(X_train)
        
        preds_df = pd.DataFrame({'actual': y_train})
    
        preds_df['tweedie_preds'] = tweedie_preds
    
        tweedie_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.tweedie_preds))
    
        output = {
                'power': i,
                'tweedie_rmse': tweedie_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('tweedie_rmse') 


def linear_poly(X_train, y_train):
    
    '''
        This function is used to run a for loop on liner poly. We will use the best preforming model,
        and use it on the validate datasets.
    '''
    
    metrics = []

    for i in range(2,4):

        pf = PolynomialFeatures(degree = i)

        pf.fit(X_train, y_train)

        X_polynomial = pf.transform(X_train)

        lm2 = LinearRegression()

        lm2.fit(X_polynomial, y_train)
        
        preds_df = pd.DataFrame({'actual': y_train})

        preds_df['poly_preds'] = lm2.predict(X_polynomial)

        poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))

        output = {
                'degree': i,
                'poly_rmse': poly_rmse
                 }

        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('poly_rmse') 


def validate_models(X_train, y_train, X_validate, y_validate):
    '''
    This model is used to test our models on the validate datasets and then return the results. 
    These results will then be used to find our best model.
    '''
       
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_val = lm.predict(X_validate)
    
    val_preds_df = pd.DataFrame({'actual_val': y_validate})
    
    val_preds_df['lm_preds'] = lm_val

    lm_rmse_val = sqrt(mean_squared_error(val_preds_df['actual_val'], val_preds_df['lm_preds']))

    #tweedie model
    
    tweedie = TweedieRegressor(power = 1)
    
    tweedie.fit(X_train, y_train)
    
    tweedie_val = tweedie.predict(X_validate)
    
    val_preds_df['tweedie_preds'] = tweedie_val
    
    tweedie_rmse_val = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df.tweedie_preds))
    
    #polynomial model
    
    pf = PolynomialFeatures(degree = 2)
    
    pf.fit(X_train, y_train)
    
    X_train = pf.transform(X_train)
    X_validate = pf.transform(X_validate)
    
    lm2 = LinearRegression()
    
    lm2.fit(X_train, y_train)
    
    val_preds_df['poly_vals'] = lm2.predict(X_validate)
    
    poly_validate_rmse = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['poly_vals']))

    #lasso_lars model
    
    lasso = LassoLars(alpha = .05 )
    
    lasso.fit(X_train, y_train)
    
    lasso_val = lasso.predict(X_validate)
    
    val_preds_df['lasso_preds'] = lasso_val

    lasso_rmse_val = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['lasso_preds']))
    
    
    return lm_rmse_val, tweedie_rmse_val, lasso_rmse_val, poly_validate_rmse


def best_models(X_train, y_train, X_validate, y_validate):
    
    '''
    This function uses the train and validate datasets and returns the results of the best preforming model 
    for each algorithm. The results are returned as a dataframe.
    '''
    
    lm_rmse = linear_model(X_train, y_train).iloc[0,1]
    
    lasso_rmse = lasso_lars(X_train, y_train).iloc[0,1]
    
    tweedie_rmse = tweedie_models(X_train, y_train).iloc[0,1]
        
    poly_rmse = linear_poly(X_train, y_train).iloc[1,1]
    
    baseline_rmse = get_baseline(y_train)
    
    lm_rmse_val, tweedie_rmse_val, lasso_rmse_val, poly_validate_rmse = validate_models(X_train, y_train, X_validate, y_validate)
    
    df = pd.DataFrame({'model': ['linear', 'tweedie', 'lasso_lars','linear_poly', 'baseline'],
                      'train_rmse': [lm_rmse, tweedie_rmse, lasso_rmse, poly_rmse,  baseline_rmse],
                      'validate_rmse': [lm_rmse_val, tweedie_rmse_val, lasso_rmse_val, poly_validate_rmse, baseline_rmse]})
    
    df['difference'] = df['train_rmse'] - df['validate_rmse']
    
    return df.sort_values('difference').reset_index().drop(columns = ('index'))



def test_model(X_train, y_train, X_test, y_test):
    '''
    This function is used to test our best model and use it 
    on the test datasets to get our final results.
    '''
    
    # Step 1: Create a PolynomialFeatures object with degree=2
    pf = PolynomialFeatures(degree=2)
    
    # Step 2: Fit the PolynomialFeatures object to the training data
    pf.fit(X_train)
    
    # Step 3: Transform the training and test data using the fitted PolynomialFeatures object
    X_train_poly = pf.transform(X_train)
    X_test_poly = pf.transform(X_test)
    
    # Step 4: Create a LinearRegression object and fit it to the transformed training data
    lm2 = LinearRegression()
    lm2.fit(X_train_poly, y_train)
    
    # Step 5: Use the fitted LinearRegression object to predict the target variable for the test data
    lm2_preds = lm2.predict(X_test_poly)
    
    # Step 6: Create a dataframe to store the actual test values and the predicted values
    test_preds_df = pd.DataFrame({'actual_test': y_test})
    test_preds_df['poly_test'] = lm2_preds
    
    # Step 7: Calculate the root mean squared error (RMSE) between the actual and predicted test values
    poly_test_rmse = sqrt(mean_squared_error(test_preds_df.actual_test, test_preds_df['poly_test']))
    
    # Step 8: Return the RMSE value
    print(f' The RMSE score on test data is: {poly_test_rmse}')






