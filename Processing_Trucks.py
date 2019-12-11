#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
import pickle

pd.set_option('display.max_columns', 500)


# In[18]:


infile = open("/Users/margaretmccall/Downloads/final.pickle","rb")
noaa = pickle.load(infile)
infile.close()


# In[19]:


df = gpd.read_file("/Users/margaretmccall/Downloads/faf4_esri_arcgis/FAF4.shp")


# In[20]:


pip install dbfread


# In[21]:


import dbfread


# In[22]:


from dbfread import DBF


# In[23]:


tab = DBF("/Users/margaretmccall/Downloads/Assignment Result 4/FAF4DATA_V43.DBF")


# In[27]:


frame= pd.DataFrame(iter(tab))


# In[28]:


frame.head()


# In[30]:


trucks = frame.merge(df, on='FAF4_ID', how="left")


# In[31]:


trucks.shape


# In[33]:


trucks = trucks[trucks['STATE_x']=='CA']


# In[34]:


trucks.shape


# In[35]:


trucks.head()


# In[36]:


trucks_ca_allData = trucks


# In[68]:


trucks = trucks[['AADT12','FAF12','NONFAF12','YKTON12','KTONMILE12','LENGTH','geometry']].reset_index(drop=True)


# In[69]:


trucks.head()


# In[42]:


epa_coords = pd.read_csv("EPA_Station_Coordinates.csv")


# In[47]:


epa_points = []
for i in range(len(epa_coords['0'])):
    epa_points.append(Point(epa_coords['1'][i], epa_coords['0'][i]))


# In[48]:


epa_coords['geometry'] = epa_points


# In[50]:


epa_coords.drop(columns='Unnamed: 0', inplace=True)


# In[54]:


plt.hist(trucks['LENGTH']) 
#ok, we're just gonna assume that if a road segment intersects the buffer, it's assigned to it for now


# In[106]:


#we have an oval issue
km_per_deg = 111.1
epa_coords['Buffer1000m'] = epa_coords['geometry'].apply(lambda pt: pt.buffer(1/km_per_deg)) 
epa_coords['Buffer100m'] = epa_coords['geometry'].apply(lambda pt: pt.buffer(1/km_per_deg/10))
epa_coords['Buffer5000m'] = epa_coords['geometry'].apply(lambda pt: pt.buffer(1/km_per_deg*5)) 


# In[107]:


epa_coords.head(5)


# In[111]:


for i in range(len(epa_coords['Buffer5000m'])):
    trucks['5000m-'+str(i)] = trucks['geometry'].apply(lambda road: road.intersects(epa_coords['Buffer5000m'][i]))


# In[112]:


#getting results for 100m buffer
#dammit just realized these will not vary in time...is that ok? i guess so

for i in range(len(epa_coords['Buffer5000m'])):
    for fig in ['AADT12','FAF12','NONFAF12','YKTON12','KTONMILE12']:
        epa_trucks.loc[i,'5000m'+fig] = np.nansum(trucks[fig][trucks['5000m-'+str(i)]])


# In[113]:


#can use union.length if results above are weird

epa_trucks.head(50)


# In[114]:


epa_trucks.to_csv("TruckTrafficData.csv")


# In[ ]:




