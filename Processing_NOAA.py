# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # <font color='yellow'>Exploring & processing NOAA data</font>
# 
# 
# Sections:
# * Loading and exploring data (NEED TO EXPLORE)
# * Processing data to fit single-point and multi-point models
# %% [markdown]
# ### <font color='yellow'>Loading data (still need to do EDA!!)</font>

# %%
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


# %%
path = r'/Users/margaretmccall/Downloads/final'
all_files = glob.glob(path + "/*.csv")


# %%
dfs = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)


# %%
df = pd.concat(dfs, ignore_index=True)


# %%
df.head()


# %%
#Picking the most relevant-seeming columns
df = df[['Date_Time','Station_ID','wind_direction_set_1','wind_speed_set_1','pressure_set_1d','precip_accum_one_hour_set_1','air_temp_set_1']]


# %%
df = df.rename(columns={"Date_Time":"datetime"})


# %%
df.head()


# %%
df.to_csv("NOAA_Data_Complete.csv", index=False)

# %% [markdown]
# # <font color='yellow'>Processing NOAA data (run from here down to reformat data)</font>

# %%
df = pd.read_csv("NOAA_Data_Complete.csv")

# %% [markdown]
# ## <font color='yellow'>Getting station coordinates</font>

# %%
#Processing station coords
coords = pd.read_csv("noaa_ca_stations.csv")


# %%
coords.head(1)


# %%
import shapely.wkt
lons, lats = [], []

for i in range(len(coords['geometry'])):
    lon, lat = shapely.wkt.loads(coords['geometry'][i]).xy
    lons.append(lon[0])
    lats.append(lat[0])


# %%
noaa_coords = coords[['stationIdentifier']]


# %%
noaa_coords.loc[:,'latitude'] = lats
noaa_coords.loc[:,'longitude'] = lons


# %%
noaa_coords.to_csv("NOAA_Coordinates.csv", index=False)


# %%
#Adding coords
noaa_coords.rename(columns={'stationIdentifier':'Station_ID'}, inplace=True)


# %%
df_noaa = df
df_noaa = df_noaa.merge(noaa_coords, on='Station_ID', how='left')

# %% [markdown]
# ### Relabeling columns
# 

# %%
df_noaa.head(1)


# %%
df_noaa = df_noaa.rename(columns={'Station_ID':'name', 'wind_direction_set_1':'wind_dir', 'wind_speed_set_1':'wind_speed','pressure_set_1d':'pressure','precip_accum_one_hour_set_1':'precip', 'air_temp_set_1':'temp'})

# %% [markdown]
# ### <font color='yellow'>Shaping data for single-point model</font>

# %%
df_noaa.head()


# %%
df_noaa = df_noaa.drop(columns=['latitude','longitude'])


# %%
data_types = df_noaa.columns.tolist()[2:7]

DataFrameDict = {elem : pd.DataFrame for elem in data_types}


# %%
for key in DataFrameDict.keys():
    DataFrameDict[key] = df_noaa[[key,'datetime','name']]


# %%
for key in DataFrameDict.keys():
    DataFrameDict[key] = DataFrameDict[key].pivot_table(index='datetime',columns='name',values=key)


# %%
for key in DataFrameDict.keys():
    DataFrameDict[key] = DataFrameDict[key].add_suffix("_"+key)


# %%
DataFrameDict['wind_dir'].head(1)
#wow, this worked like a charm


# %%
df_noaa_output = pd.concat(DataFrameDict, axis=1)
df_noaa_output.to_csv("NOAA_Data_SinglePointModel.csv")


# %%
#Outputting each data type into separate CSV, if we want that
#for key in DataFrameDict.keys():
#    DataFrameDict[key].to_csv("NOAA"+key+".csv")

# %% [markdown]
# ### <font color='yellow'>Shaping data for multi-point model</font>

# %%
df_noaa.head()


# %%
df_noaa.columns.tolist()


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


# %%
df_noaa.head(1)


# %%
df_noaa.to_csv("NOAA_Data_MultiPointModel.csv", index=False)


# %%


