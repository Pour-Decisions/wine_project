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
import plotly as py
import plotly.graph_objs as go

def combined_df(df, f1, f2, f3, f4):
    '''
    This function calls another function in explore.py 
    and merges a column to the original dataset
    '''
    
    X1 = clustering(df, f1, f2)
    X2 = clustering(df, f1, f3)
    X3 = clustering(df, f2, f4)
    
    scaled_clusters1 = X1['scaled_clusters']
    scaled_clusters2 = X2['scaled_clusters']
    scaled_clusters3 = X3['scaled_clusters']
    
    df = pd.merge(df, scaled_clusters1, left_index=True, right_index=True)
    df = pd.merge(df, scaled_clusters2, left_index=True, right_index=True)
    df = pd.merge(df, scaled_clusters3, left_index=True, right_index=True)
    
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
    kmeans = KMeans(n_clusters = 4, random_state= seed)
    kmeans.fit(X)
    kmeans.predict(X)
    # scale features
    mm_scaler = MinMaxScaler()
    X[[f1, f2]] = mm_scaler.fit_transform(X[[f1, f2]])
    # predict
    kmeans_scale = KMeans(n_clusters = 4, random_state = 828)
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
    point = (4, 62) # specify the x and y values of the point to annotate
    plt.annotate("optimal cluster", xy=point, xytext=(4.2, 75), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    # title
    plt.title('Clusters Versus Inertia')
    
    return plt.show()
    


def find_best_features(df, k_min, k_max):
    '''
    This function takes in a dataframe, a minimum number 
    of clusters (k_min), and a maximum number 
    of clusters (k_max). It returns a list of the top 6 features 
    to use for clustering based on the mean Silhouette score.
    '''
    # Remove the target column from the dataframe
    X = df.drop(['quality', 'color'], axis=1)

    # Create an empty dictionary to store the mean Silhouette scores for each feature
    feature_scores = {}

    # Loop through each feature
    for feature in X.columns:
        # Create a new dataframe with just this feature
        X_single = X[[feature]]

        # Create an empty list to store the Silhouette scores for this feature
        scores = []

        # Loop through the range of k values
        for k in range(k_min, k_max+1):
            # Fit a KMeans clustering model
            kmeans = KMeans(n_clusters=k, random_state=828).fit(X_single)

            # Calculate the Silhouette score for the clustering
            score = silhouette_score(X_single, kmeans.labels_)

            # Add the score for this iteration of k to the list of scores for this feature
            scores.append(score)

        # Calculate the mean Silhouette score for this feature
        mean_score = sum(scores) / len(scores)

        # Add the mean Silhouette score for this feature to the dictionary
        feature_scores[feature] = mean_score

    # Sort the dictionary by value (descending) and get the top 6 features
    top_features = sorted(feature_scores, key=feature_scores.get, reverse=True)[:6]

    # Print the top 6 features with their mean Silhouette scores
    print("Top 6 features:")
    for feature in top_features:
        print(f"{feature}: {feature_scores[feature]:.3f}")
    


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
    
    
    X_train = train_scaled.drop(columns = ['quality', 'free sulfur dioxide'])
    X_train = pd.get_dummies(X_train, columns = ['color', 'scaled_clusters'])
    y_train = train_scaled['quality']


    X_validate = validate_scaled.drop(columns = ['quality', 'free sulfur dioxide'])
    X_validate = pd.get_dummies(X_validate, columns = ['color', 'scaled_clusters'])
    y_validate = validate_scaled['quality']


    X_test = test_scaled.drop(columns = ['quality', 'free sulfur dioxide'])
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



    
def get_vis_df(df):
    '''
    takes in df, makes a copy, then preps for cluster model visualization
    '''
    vis_df = df.sample(frac=1, random_state=101).reset_index(drop=True)
    # Define a dictionary that maps the quality levels to numerical values
    quality_map = {'low': 0, 'medium': 1, 'high': 2}
    #
    vis_df['quality_label'] = vis_df.quality.apply(lambda q: 'low' if q <= 5 else 'medium' if q <= 7 else 'high')
    # Create a new column in the DataFrame that maps the quality values to numerical values
    vis_df['quality_num'] = vis_df['quality_label'].map(quality_map)
    # drop target variable
    vis_df = vis_df.drop(columns=['quality_label', 'quality'])
    
    return vis_df
    
def get_3d_vis1(df):
    '''
    renders 3d cluster vis
    '''
    data = df
    
    trace1 = go.Scatter3d(
        x= data['density'],
        y= data['quality_num'],
        z= data['residual sugar'],
        mode='markers',
         marker=dict(
            color = data['residual sugar'], 
            size= 10,
            line=dict(
                color= data['residual sugar'],
                width= 12
            ),
            opacity=0.8
         )
    )
    data1 = [trace1]
    layout = go.Layout(
        title= 'Clusters with Density and Residual Sugar',
        scene = dict(
                xaxis = dict(title  = 'Density'),
                yaxis = dict(title  = 'Quality'),
                zaxis = dict(title  = 'Residual Sugar')
            )
    )
    fig = go.Figure(data=data1, layout=layout)
    py.offline.iplot(fig)
    
    
def get_3d_vis2(df):
    '''
    renders 3d cluster vis
    '''
    data = df
    
    trace1 = go.Scatter3d(
        x= data['density'],
        y= data['quality_num'],
        z= data['volatile acidity'],
        mode='markers',
         marker=dict(
            color = data['volatile acidity'], 
            size= 10,
            line=dict(
                color= data['volatile acidity'],
                width= 12
            ),
            opacity=0.8
         )
    )
    data1 = [trace1]
    layout = go.Layout(
        title= 'Clusters with Density and Volatile Acidity',
        scene = dict(
                xaxis = dict(title  = 'Density'),
                yaxis = dict(title  = 'Quality'),
                zaxis = dict(title  = 'Volatile Acidity')
            )
    )
    fig = go.Figure(data=data1, layout=layout)
    py.offline.iplot(fig)

    
def get_3d_vis3(df):
    '''
    renders 3d cluster vis
    '''
    data = df
    
    trace1 = go.Scatter3d(
        x= data['total sulfur dioxide'],
        y= data['quality_num'],
        z= data['residual sugar'],
        mode='markers',
         marker=dict(
            color = data['residual sugar'], 
            size= 10,
            line=dict(
                color= data['residual sugar'],
                width= 12
            ),
            opacity=0.8
         )
    )
    data1 = [trace1]
    layout = go.Layout(
        title= 'Clusters with Total Sulfur Dioxide and Residual Sugar',
        scene = dict(
                xaxis = dict(title  = 'Total Sulfur Dioxide'),
                yaxis = dict(title  = 'Quality'),
                zaxis = dict(title  = 'Residual Sugar')
            )
    )
    fig = go.Figure(data=data1, layout=layout)
    py.offline.iplot(fig)



def splitting_subsets2(train, train_scaled, validate_scaled, test_scaled):
    '''
    This function splits our train, validate, and test scaled datasets into X/y train,
    validate, and test subsets
    '''
    X_train2 = train_scaled.drop(columns = ['quality', 'free sulfur dioxide', 'scaled_clusters'])
    X_train2 = pd.get_dummies(X_train2, columns = ['color'])
    y_train2 = train_scaled['quality']


    X_validate2 = validate_scaled.drop(columns = ['quality', 'free sulfur dioxide', 'scaled_clusters'])
    X_validate2 = pd.get_dummies(X_validate2, columns = ['color'])
    y_validate2 = validate_scaled['quality']


    X_test2 = test_scaled.drop(columns = ['quality', 'free sulfur dioxide', 'scaled_clusters'])
    X_test2 = pd.get_dummies(X_test2, columns = ['color'])
    y_test2 = test_scaled['quality']

    return X_train2, y_train2, X_validate2, y_validate2, X_test2, y_test2    

    