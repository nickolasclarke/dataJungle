#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import scale
import matplotlib.pyplot as plt

import glob
import os
import datetime
import zipfile


# In[2]:


pd.set_option('display.max_columns', 500)


# ## Loading data

# In[3]:


df_beac = pd.read_csv("Beacon_Data_SinglePointModel.csv")


# In[4]:


df_epa = pd.read_csv("EPA_Data_SinglePointModel.csv")


# ## Merging & cleaning EPA and beacon dataframes //  <font color='red'> ** Still need to normalize before using Ridge </font>

# In[5]:


bigdf = df_epa.merge(df_beac, how='left', on='datetime').reset_index(drop=True)


# In[7]:


bigdf.drop(columns=['Laney College'], inplace=True) #laney college is screwed up (from visual check)


# ### Which rows have NAs?

# In[8]:


#now we gotta see how many rows to drop
nullcount = bigdf.isnull().sum(axis=1)


# In[9]:


bigdf0 = bigdf[nullcount == 0].reset_index(drop=True)


# ## Starting actual analysis!

# In[18]:


#here we go
y = bigdf0['epa_meas']
X = bigdf0.drop(columns=['epa_meas','datetime','latitude','longitude','name'])


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# In[20]:


#Getting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[21]:


def fit_model(Model, X_train, X_test, y_train, y_test, alpha = 1):
    """
    This function fits a model of type Model to the data in the training set of X and y, and finds the MSE on the test set
    Inputs: 
        Model (sklearn model): the type of sklearn model with which to fit the data - LinearRegression, Ridge, or Lasso
        X_train: the set of features used to train the model
        y_train: the set of response variable observations used to train the model
        X_test: the set of features used to test the model
        y_test: the set of response variable observations used to test the model
        alpha: the penalty parameter, to be used with Ridge and Lasso models only
    Returns:
        mse: a scalar containing the mean squared error of the test data
        coeff: a list of model coefficients
    """
    
    # initialize model
    if (Model == Ridge) | (Model == Lasso):
        model = Model(alpha = alpha)
    elif Model == LinearRegression:
        model = Model() #these parens were key. still figuring out how to pass this as argument
        
    # fit model
    model.fit(X_train, y_train)
    
    # get mean squared error of test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    
    # get coefficients of model
    coef = model.coef_
    
    return mse, coef


# In[22]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[23]:


mses_all


# In[24]:


coefs_all #This seems weird...tho I guess lasso has a big MSE. Maybe just because we didn't set alpha yet


# In[25]:


from sklearn.model_selection import KFold


# In[26]:


def model_cv_mse(Model, X, y, alphas, k = 3):
    """
    This function calculates the MSE resulting from k-fold CV using lasso regression performed on a training subset of 
    X and y for different values of alpha.
    Inputs: 
        Model (sklearn model): the type of sklearn model with which to fit the data - Ridge or Lasso
        X: the set of features used to fit the model
        y: the set of response variable observations
        alphas: a list of penalty parameters
        k: number of folds in k-fold cross-validation
    Returns:
        average_mses: a list containing the mean squared cross-validation error corresponding to each value of alpha
    """
    mses = np.zeros((k,len(alphas))) # initialize array of MSEs to contain MSE for each fold and each value of alpha
        
    kf = KFold(k, shuffle=True, random_state=0) # get kfold split
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    fold = 0
    for train_i, val_i in kf.split(X):
        # get training and validation values
        X_f_train = X.iloc[train_i,:]
        X_f_val = X.iloc[val_i,:]
        y_f_train = y[train_i]
        y_f_val = y[val_i]
        
        for i in range(len(alphas)): # loop through alpha values
            model = Model(alpha=alphas[i]) # initialize model

            model.fit(X_f_train,y_f_train) # fit model
            
            y_pred = model.predict(X_f_val) # get predictions
            
            mse = mean_squared_error(y_f_val,y_pred)
            mses[fold,i] = mse # save MSE for this fold and alpha value
        
        fold += 1
    
    average_mses = np.mean(mses, axis=0) # get average MSE for each alpha value across folds
    
    return average_mses


# In[27]:


alphas_ridge = [0.01, 0.1, 1, 10, 100, 1000, 10000, 10**5, 10**6, 10**7]
alphas_lasso = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
mses_ridge = model_cv_mse(Ridge, X_train, y_train, alphas_ridge)
mses_lasso = model_cv_mse(Lasso, X_train, y_train, alphas_lasso)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplot(121)
plt.plot(np.log10(alphas_ridge),mses_ridge)
plt.title("Model Selection w/ Ridge Regression")
plt.xlabel("Log of alpha")
plt.ylabel("Cross-validated MSE")

plt.subplot(122)
plt.plot(np.log10(alphas_lasso),mses_lasso)
plt.title("Model Selection w/ Lasso Regression")
plt.xlabel("Log of alpha")
plt.ylabel("Cross-validated MSE")

plt.tight_layout()


# In[28]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:


mses_all


# In[30]:


coefs_all


# ### Using statsmodels to get more info about the OLS model

# In[31]:


import statsmodels.api as sm
X = X_train
y = y_train
X = sm.add_constant(X)


# In[32]:


lm = sm.OLS(y, X, missing='drop')
results = lm.fit()
results.summary()

