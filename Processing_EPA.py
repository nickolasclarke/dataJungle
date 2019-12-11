#!/usr/bin/env python
# coding: utf-8

# # <font color='yellow'>Exploring & processing EPA data</font>
# 
# 
# Sections:
# * Loading and exploring data
# * Processing data to fit single-point and multi-point models

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


# In[2]:


pd.set_option('display.max_columns', 500)


# ## <font color='yellow'>Loading and exploring data</font>

# In[3]:


with open("EPA Data/jan.json") as json_file:
    dat = json.load(json_file)


# In[4]:


json_normalize(dat['Data']).head(2) #phew


# In[5]:


path = r'/Users/margaretmccall/Documents/2019 Fall/ER 131- Data Enviro Soc/Data/EPA Data/'
all_files = glob.glob(path + "*.json")


# In[6]:


dfs = []

for file in all_files:
    with open(file) as json_file:
        dat = json.load(json_file)
    df = json_normalize(dat['Data'])
    dfs.append(df)


# In[7]:


df = pd.concat(dfs, ignore_index=True)


# In[8]:


df.columns


# In[9]:


df_backup = df


# In[10]:


df.sort_values(by=['date_local','time_local','site_number'], inplace=True)


# In[11]:


df.head()


# In[12]:


df.columns


# In[13]:


len(df['latitude'].unique())
#odd...doesn't match either the number of sites nor the number of monitors...


# In[14]:


len(df['longitude'].unique())
#well, at least this matches


# In[15]:


print("There are this many sites:", len(df['site_number'].unique()))
#That's weird...I thought there were a lot more sites...


# In[16]:


#test = pd.read_csv("EPA Data/California 2018 data.csv")


# In[17]:


#len(test['Site ID'].unique())
#Yeah, the other CA data I downloaded had 157 sites...


# In[18]:


#print("Amalgamated data sites:", test['Site ID'].unique())
print("JSON detailed data sites:", df['site_number'].unique())


# In[19]:


with open("EPA Data/Other info/MonitorList.json") as json_file:
    dat = json.load(json_file)
monitors = json_normalize(dat['Data'])


# In[20]:


monitors.head()
#got a list of the monitors--there seem to be multiple per site


# In[21]:


len(monitors['site_number'].unique())
#ok, this ALMOST matches the site_number from the big dataframe...


# In[22]:


monitors.sort_values(by=['site_number'], inplace=True)


# In[23]:


monitors.head(10)
#what the heck?! monitors that have the same site number have wildly different counties that they belong to...
#maybe we just go with the lat/lon of the sample data...


# In[24]:


df.head()


# In[25]:


print(df['sample_frequency'].unique())
print(df['sample_duration'].unique())
print(df['uncertainty'].unique())
print(df['detection_limit'].unique()) #not sure what this is...granularity?


# In[26]:


df.to_csv("EPA_Data_Complete.csv", index=False)


# In[27]:


df['site_number'].groupby([df['site_number'], df['sample_frequency']]).size()


# ## <font color='yellow'>Processing data to fit single-point and multi-point models</font>

# ### <font color='yellow'>Saving EPA coordinates</font>

# In[28]:


df_epa = pd.read_csv("EPA_Data_Complete.csv")


# In[29]:


coords_epa = df_epa[['county','latitude','longitude']].drop_duplicates(keep='first').reset_index(drop=True)
coords_epa.to_csv("EPA_Coordinates.csv", index=False)


# ### <font color='yellow'>Simplifying EPA data for multi-point model</font>

# In[30]:


df_epa.head(1)


# In[31]:


def pare_down_epa(epa1):
    """
    Takes in a dataframe of EPA data (epa1), raw, [[but selected for a single EPA sensor (so far tested w/ hourly data)]]
    Outputs a dataframe with a column containing the date&time, and another column containing the hourly reading
    """
    
    #Paring down to the essential columns: date, time, measurement
    epa1 = epa1[['date_gmt','time_gmt','sample_measurement','latitude','longitude']]
    
    #Getting date & hour in datetime format
    epa1['datetime'] = pd.to_datetime(epa1['date_gmt'])
    epa1['Hour'] = epa1['time_gmt'].apply(lambda dt: datetime.datetime.strptime(dt, "%H:%M").hour)
    
    #Combining date & hour into a single column
    dts = []
    for i in range(len(epa1['datetime'])):
        dts.append(datetime.datetime.combine(epa1['datetime'][i], datetime.time(epa1['Hour'][i])))
    epa1['datetime'] = dts

    #Dropping unnecessary cols
    epa1.drop(columns=['Hour','date_gmt','time_gmt'],inplace=True)
    
    return epa1


# In[32]:


epa_simplified = pare_down_epa(df_epa)


# In[35]:


epa_simplified.head(1)


# In[36]:


epa_simplified['name'] = '?'


# In[37]:


cols = ['datetime','latitude','longitude','name','sample_measurement']
epa_simplified = epa_simplified[cols].rename(columns={'sample_measurement':'epa_meas'})


# In[38]:


epa_simplified.to_csv("EPA_Data_MultiPointModel.csv", index=False)


# ### <font color='yellow'>Simplifying EPA data for single-point model. Consider doing more EDA</font>
# Currently targeting single sensor in Beacon area

# In[39]:


beac_df = pd.read_csv("Beacon_Coordinates.csv")
beac_coords = list(set(list(zip(beac_df['latitude'],beac_df['longitude']))))


# In[40]:


epa_coords = list(set(list(zip(df_epa['latitude'],df_epa['longitude']))))


# In[41]:


beac_lons, beac_lats, epa_lons, epa_lats = [], [], [], []

for i in range(len(beac_coords)):
    beac_lats.append(beac_coords[i][0])
    beac_lons.append(beac_coords[i][1])
    
for i in range(len(epa_coords)):
    epa_lats.append(epa_coords[i][0])
    epa_lons.append(epa_coords[i][1])


# In[42]:


plt.plot(epa_lons, epa_lats, 'or')
plt.plot(beac_lons, beac_lats, 'ob')
plt.xlim(-123,-121)
plt.ylim(37,39.5)


# In[43]:


beac_pts = pd.Series(beac_coords).apply(lambda coord: Point(coord)) #success
epa_pts = pd.Series(epa_coords).apply(lambda coord: Point(coord))
#would love to make a buffer around relevant points, but instead just going to eyeball it


# In[44]:


y1 = -122.75 #-- lon
y2 = -122
x1 = 37.5 #-- lat
x2 = 38.5
bbox = Polygon([[x1,y1],[x1,y2],[x2,y2],[x2,y1],[x1,y1]])


# In[45]:


inbox = epa_pts.apply(lambda x: x.intersects(bbox))


# In[46]:


epa_lons_include = np.array(epa_lons)[np.array(inbox)]
epa_lats_include = np.array(epa_lats)[np.array(inbox)]


# In[47]:


df_epa_subset = df_epa[df_epa['latitude'].isin(epa_lats_include) & df_epa['longitude'].isin(epa_lons_include)].reset_index(drop=True)


# In[48]:


target_lat = df_epa_subset['latitude'][0]
target_lon = df_epa_subset['longitude'][0]


# In[49]:


epa_singlepoint = epa_simplified[(epa_simplified['latitude']==target_lat) & (epa_simplified['longitude']==target_lon)]
epa_singlepoint.to_csv("EPA_Data_SinglePointModel.csv", index=False)


# In[ ]:




