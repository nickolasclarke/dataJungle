#!/usr/bin/env python
# coding: utf-8

# # <font color='yellow'>Exploring & processing NOAA data</font>
# 
# 
# Sections:
# * Loading and exploring data (NEED TO EXPLORE)
# * Processing data to fit single-point and multi-point models

# ### <font color='yellow'>Loading data (still need to do EDA!!)</font>

# In[6]:


import json
import csv
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import scale
import matplotlib.pyplot as plt

import glob
import os
import datetime

pd.set_option('display.max_columns', 500)


# In[7]:


path = r'/Users/margaretmccall/Downloads/final'
all_files = glob.glob(path + "/*.csv")


# In[8]:


dfs = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)


# In[9]:


df = pd.concat(dfs, ignore_index=True)


# In[10]:


df.head()


# In[11]:


#Picking the most relevant-seeming columns
df = df[['Date_Time','Station_ID','wind_direction_set_1','wind_speed_set_1','pressure_set_1d','precip_accum_one_hour_set_1','air_temp_set_1']]


# In[12]:


df = df.rename(columns={"Date_Time":"datetime"})


# In[13]:


df.head()


# In[14]:


df.to_csv("NOAA_Data_Complete.csv", index=False)


# # <font color='yellow'>Processing NOAA data</font>

# ## <font color='yellow'>Getting station coordinates</font>

# In[15]:


#Processing station coords
coords = pd.read_csv("noaa_ca_stations.csv")


# In[16]:


coords.head()


# In[17]:


import shapely.wkt
lons, lats = [], []

for i in range(len(coords['geometry'])):
    lon, lat = shapely.wkt.loads(coords['geometry'][i]).xy
    lons.append(lon[0])
    lats.append(lat[0])


# In[18]:


noaa_coords = coords[['stationIdentifier']]


# In[19]:


noaa_coords['latitude'] = lats
noaa_coords['longitude'] = lons


# In[20]:


noaa_coords.to_csv("NOAA_Coordinates.csv", index=False)


<<<<<<< HEAD
# %% [markdown]
# ### Trying to fix datetimes to align with EPA/Beacon sensors
# 

# %%
df_noaa['dt'] = pd.to_datetime(df_noaa['datetime'])


# %%
#adding a date+hour column like EPA/Beacon have
dts = []
for d in df_noaa['dt']:
    date = datetime.date(year=d.year ,day=d.day, month=d.month)
    dts.append(datetime.datetime.combine(date, datetime.time(d.hour)))


# %%
df_noaa['datetime'] = dts
df_noaa = df_noaa.drop(columns=['dt'])

=======
# ### <font color='yellow'>Shaping data for single-point model</font>
>>>>>>> 63e88c2c7e9882b1343bf52a000dc514f460b77a

# In[21]:

# %% [markdown]
# ### <font color='yellow'>Shaping data for single-point model</font>

<<<<<<< HEAD
# %%
df_noaa_wlonlat = df_noaa
df_noaa = df_noaa.drop(columns=['latitude','longitude'])
=======
#TBD
>>>>>>> 63e88c2c7e9882b1343bf52a000dc514f460b77a


# ### <font color='yellow'>Shaping data for multi-point model</font>

# In[22]:


#Adding coords
noaa_coords.rename(columns={'stationIdentifier':'Station_ID'}, inplace=True)


# In[23]:


df_noaa = df
df_noaa = df_noaa.merge(noaa_coords, on='Station_ID', how='left')


# In[24]:


<<<<<<< HEAD
# %%
#df_noaa_output = pd.concat(DataFrameDict, axis=1) -- this would work beautifully but some indexing problem--maybe from datetimes--is generating all NaNs


# %%
#so instead, we will dumbly merge by hand, bc out of time for this


# %%
x = DataFrameDict['wind_dir'].merge(DataFrameDict['wind_speed'], on='datetime', how='left')
x = x.merge(DataFrameDict['pressure'], on='datetime', how='left')
x = x.merge(DataFrameDict['precip'], on='datetime', how='left')
x = x.merge(DataFrameDict['temp'], on='datetime', how='left')
x.shape


# %%
df_noaa_output = x
df_noaa_output.head()


# %%
df_noaa_output.to_csv("NOAA_Data_SinglePointModel.csv")
=======
df_noaa.head()
>>>>>>> 63e88c2c7e9882b1343bf52a000dc514f460b77a


# In[25]:


<<<<<<< HEAD
# %%
df_noaa = df_noaa_wlonlat
df_noaa.head()


# %%
#We want every dataframe's columns in the following order: datetime, lat, lon, name, measurement1, measurement2, ...
cols = ['datetime', 'latitude','longitude','name',
 'wind_dir',
 'wind_speed',
 'pressure',
 'precip',
 'temp',
 ]
df_noaa = df_noaa[cols]
=======
#We want every dataframe's columns in the following order: datetime, lat, lon, name, measurement1, measurement2, ...
cols = ['datetime',  'latitude', 'longitude', 'Station_ID',
 'wind_direction_set_1',
 'wind_speed_set_1',
 'pressure_set_1d',
 'precip_accum_one_hour_set_1',
 'air_temp_set_1',
]
df_noaa = df_noaa[cols].rename(columns={'Station_ID':'name'})


# In[26]:


df_noaa.head(1)
>>>>>>> 63e88c2c7e9882b1343bf52a000dc514f460b77a


# In[28]:


df_noaa.to_csv("NOAA_Data_MultiPointModel.csv", index=False)


# In[ ]:




