#!/usr/bin/env python
# coding: utf-8

# # <font color='yellow'>How can we predict not just the hourly PM2.5 concentration at the site of one EPA sensor, but predict the hourly PM2.5 concentration anywhere?</font>
# 
# Here, you build a new model for any given hour on any given day. This will leverage readings across all ~120 EPA sensors, and only PA data

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
from sklearn.preprocessing import StandardScaler
import glob
import os
import datetime
from datetime import timezone
import zipfile
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', 500)


# ## <font color='yellow'>Loading data</font>
# 
# We'll load only Purpleair data


# In[]
df_pa = pd.read_csv("pa_full.csv")
df_pa = df_pa.rename(columns={"lat":"latitude","lon":"longitude"})
df_pa['name'] = 'none'
cols = ['datetime', 'latitude', 'longitude','name','PM1.0 (ATM)','PM2.5 (ATM)','PM10.0 (ATM)','PM2.5 (CF=1)','id']
df_pa = df_pa[cols]
df_pa.head(10)

df_epa = pd.read_csv("EPA_Data_MultiPointModel.csv")
df_epa.head(1)


# ## <font color='yellow'>Selecting a date and time</font>
# 
# This function subsets a dataframe to only contain data from a certain date and hour.

# In[8]:


def select_datetime(df, month, day, hour, year=2018):
    """
    Inputs: dataframe with a column called 'datetime' containing *UTC* strings formatted as datetime objects; 
        month, day, and hour desired (year is assumed 2018)
    Outputs: rows of the original dataframe that match that date/hour
    """
    
    if pd.to_datetime(df['datetime'][0]).tzinfo is None:
        desired_datetime = datetime.datetime(month=month, day=day, year=2018, hour=hour)
    else:
        desired_datetime = datetime.datetime(month=month, day=day, year=2018, hour=hour).replace(tzinfo=timezone.utc)

    idx = pd.to_datetime(df['datetime']) == desired_datetime
    
    return df[idx].reset_index(drop=True)


 
# In[10]:


def get_distance_matrix(df_epa, df_other):
    """
    Inputs: df of EPA sensors and df of dataset with coordinates in it.
        Should have 'latitude' and 'longitude' columns
    Output: data frame of Euclidean distance from each EPA sensor to each point in the other dataframe
        (Assumes the earth is flat)
    """
    
    dists = np.full((df_epa.shape[0], df_other.shape[0]), np.inf)
    
    for i in range(df_epa.shape[0]):
        for j in range(df_other.shape[0]):
            dists[i,j] = ((df_epa['latitude'][i] - df_other['latitude'][j])**2 
                         + (df_epa['longitude'][i] - df_other['longitude'][j])**2)**.5
            
    return dists


# In[11]:


def get_KNN_avgs(df_epa, df_other, k=5):
    """
    Inputs: two dataframes (one EPA data, one other data) with cols ['datetime', 'latitude',
        'longitude','name','data1...'...],
        and a value for k (the number of neighbors that will be averaged for a given point)
    Outputs: the k closest points from the other dataset
    """
    
    dists = get_distance_matrix(df_epa, df_other)
    order = np.argsort(dists, axis=1)
    
    df_output = df_epa[['epa_meas','latitude','longitude']]
    cols = df_other.columns.tolist()[4:]
    for i in range(dists.shape[0]):
        for col in cols:
            sorted_data = df_other[col][order[i]]
            x = np.nanmean(sorted_data[0:k])
            df_output.loc[i,col] = x
            
    return df_output


# In[12]:


def get_final_timevarying_dataframe(df_epa, other_dfs, month, day, hour, k=5):
    """
    Input: EPA dataframe, list of any other dataframes to be added in via KNN, and desired day/time
    Other df format must be ['datetime', 'latitude','longitude','name','data1','data2'..]
    Output: a nearly-analysis-ready df (ie, just y and X)...still need to add in
        non-time-varying data
    """
        
    epa_subset = select_datetime(df_epa, month=month, day=day, hour=hour)
    df0_subset = select_datetime(other_dfs[0], month=month, day=day, hour=hour)
    output = get_KNN_avgs(epa_subset, df0_subset)
    
    for df in other_dfs[1:]:
        df_subset = select_datetime(df, month=month, day=day, hour=hour)
        x = get_KNN_avgs(epa_subset, df_subset)
        output = output.merge(x, on='epa_meas',how='left')
        
    return output


# In[13]:
#First Run on May 2 at 8 am
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=5,day=2,hour=8)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[27]:


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
        model = Model(max_iter = 1000, alpha = alpha, normalize=True)
    elif Model == LinearRegression:
        model = Model(normalize=True) 
        
    # fit model
    model.fit(X_train, y_train)
    
    # get mean squared error of test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    
    # get coefficients of model
    coef = model.coef_
    
    return mse, coef


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:


mses_all


# In[30]:


coefs_all


# In[31]:


from sklearn.model_selection import KFold
def model_cv_mse(Model, X, y, alphas, k = 5):
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
            model = Model(alpha=alphas[i], normalize=True, max_iter=100000) #<<<lots of iterations

            model.fit(X_f_train,y_f_train) # fit model
            
            y_pred = model.predict(X_f_val) # get predictions
            
            mse = mean_squared_error(y_f_val,y_pred)
            mses[fold,i] = mse # save MSE for this fold and alpha value
        
        fold += 1
    
    average_mses = np.mean(mses, axis=0) # get average MSE for each alpha value across folds
    
    return average_mses


# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all


# In[13]:
#Second Run on June 7 at 8 am
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=6,day=7,hour=8)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:

mses_all

# In[30]:
coefs_all

# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all

# In[13]:
#Third Run on May 14 at 6 am
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=5,day=14,hour=6)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:

mses_all

# In[30]:
coefs_all

# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all


# In[13]:
#Fourth Run on Jan 3 at 12 pm
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=1,day=3,hour=12)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:

mses_all

# In[30]:
coefs_all

# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all

# In[13]:
#Fifth Run on Dec 22 at 4 pm
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=12,day=22,hour=16)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:

mses_all

# In[30]:
coefs_all

# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all
# In[13]:
#Sixth Run on Nov 8 at 7 am
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=11,day=8,hour=7)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:

mses_all

# In[30]:
coefs_all

# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all
# In[13]:
#Seventh Run on Nov 11 at 7 am
df_analysis = get_final_timevarying_dataframe(df_epa, [df_pa], month=11,day=11,hour=7)
df_analysis = df_analysis.drop(columns=['id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)


# In[24]:


y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)


# In[26]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)


# In[28]:


mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[29]:

mses_all

# In[30]:
coefs_all

# In[32]:


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
plt.show()


# In[33]:


#Redoing three models, w optimized alphas
mses_all = {} #new dict approach. weird i can call w/ just the name Lasso when stored as longer term
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef


# In[34]:


mses_all


# In[35]:


coefs_all