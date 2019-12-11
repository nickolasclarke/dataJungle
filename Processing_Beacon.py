#!/usr/bin/env python
# coding: utf-8

# # <font color='yellow'>Exploring & processing Beacon data</font>
# 
# 
# Sections:
# * Loading and exploring data
# * Processing data to fit single-point and multi-point models

# In[2]:


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


# ## <font color='yellow'>Loading and exploring data</font>

# In[3]:


path = r'/Users/margaretmccall/Documents/2019 Fall/ER 131- Data Enviro Soc/Data/BEACON Data'
all_files = glob.glob(path + "/*.csv")


# In[4]:


all_files = all_files[1:len(all_files)]


# In[5]:


dfs = []
bad_files = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        name = file.replace('/Users/margaretmccall/Documents/2019 Fall/ER 131- Data Enviro Soc/Data/BEACON Data/',"")
        name = name.replace("_start_2017-12-25 16_00_00_end_2018-12-31 16_00_00_measurements(1).csv","")
        x['AnnoyingName'] = name
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)


# In[6]:


df = pd.concat(dfs, ignore_index=True)


# In[7]:


df.head(1)


# In[8]:


df_nodes = pd.read_csv("BEACON Data/All_nodes.csv")


# In[9]:


df_nodes[df_nodes['id']==80]
#So--the 'id' in the nodes table is in fact the same as node_id in the data tables


# In[10]:


nodenames = df_nodes[['id', 'node_name_long','lat','lng','height_above_ground','height_above_sea']]


# In[11]:


nodenames.rename(columns={'id':'node_id'}, inplace=True)


# In[12]:


#Now combining the dataframe with the nodenames dataframe
#that has all the identifying characteristics

df_final = df.merge(nodenames, how='left', on='node_id').reset_index(drop=True)


# In[13]:


df_final.drop(columns=['AnnoyingName','julian_day','CO2_QC_level', 'CO2_ppm'], inplace=True)


# In[14]:



df_final.head()
#Local_timestamp: Pacific Time
#Datetime = UTC
#Averages have been taken for a whole hour, and assigned to beginning of the hour
#pm_pct_fs = raw instrument measurement (% of time scattering is detected by unit)
#PM_ug/m3 = converted raw measurement to estimate of ug/m3
#PM_QC_level = 0, 1, 1a, 2 (raw, raw*18.9, L1 w/ offset, L1a but inspected for quality) 
##-- confused what this is referring to, since pm_pct_fs and PM_ug/m3 seem to have own columns
##-- also confused bc -999 is another number, but appears when there's still raw data...


# In[15]:


#separating dates and times
date = datetime.datetime.strptime(df_final['datetime'][0], "%Y-%m-%d  %H:%M:%S")
date.year


# In[16]:


df_final['Year'] = df_final['datetime'].apply(lambda dt: datetime.datetime.strptime(dt, "%Y-%m-%d  %H:%M:%S").year)


# In[17]:


df_final = df_final[df_final['Year']==2018]


# In[18]:


df_final.shape


# In[19]:


df_final['dt'] = pd.to_datetime(df_final['datetime'])


# In[20]:


df_final['dt'][0].hour


# In[21]:


df_final['Month'] = df_final['dt'].apply(lambda dt: dt.month)
df_final['Day'] = df_final['dt'].apply(lambda dt: dt.day)
df_final['HourStarting'] = df_final['dt'].apply(lambda dt: dt.hour)


# In[22]:


#periodically saving in case I screw something up 
###
df_BACKUP1 = df_final


# In[23]:


df_final.head()


# In[24]:


df_final.drop(columns=['local_timestamp','datetime','dt'], inplace=True)


# In[25]:


kaiser = df_final[df_final['node_name_long']=='Kaiser Building']


# In[26]:


kaiser.shape


# In[27]:


plt.plot(kaiser.index, kaiser['pm_pct_fs'],'ob')


# In[28]:


df_final['PM_QC_level'].unique()


# In[29]:


#TAKEAWAY: -999, -111 are both unexpected / undefined, and quite numerous
df_final['PM_QC_level'].groupby(df_final['PM_QC_level']).size()


# In[30]:


#What do they seem to be? (-999, -111, that is)
#Don't obviously correspond with NaNs...seem to correspond with actual readings. I vote we ignore them
df_final[df_final['PM_QC_level']==-999].head()


# In[31]:


#While most NaNs go with -999 or -111, some are labeled 1 or even 2...
df_final[df_final['pm_pct_fs'].isna()].head()


# In[32]:


####Making dataframe to capture key features of each sensor:
#Earliest reading in year
#Last reading in year
#Total number of hours with readings (out of 8760)
#Total # of hou...


# In[33]:


stations = df_final['node_name_long'].unique()


# In[34]:


summary = pd.DataFrame()
summary['#readings at sensor'] = df_final['pm_pct_fs'].groupby(df_final['node_name_long']).size()
summary['maxreading'] = df_final['pm_pct_fs'].groupby(df_final['node_name_long']).max()
##seems like there's a cap-out at 100...
#summary['#nans'] = df_final['pm_pct_fs'].groupby(df_final['node_name_long']).apply(lambda x: x.isna()).size()


# In[35]:


summary


# In[36]:


#Fine...I'll just graph 'em
#9 rows, 6 cols would do it

plt.figure(figsize=(15,20))

for i in range(len(stations)):
    dat = df_final[df_final['node_name_long']==stations[i]]
    timeidx = dat['Month']+dat['Day']/30+dat['HourStarting']/(30*24)
    plt.subplot(9,6,i+1)
    plt.plot(timeidx,dat['pm_pct_fs'],'ob')
    plt.title(stations[i])
    plt.xlim(1,13)
    plt.ylim(0,40)
    
plt.savefig('beacon_cleaning.png', bbox_inches='tight')


# In[37]:


#Does the raw data reading always correspond to the ug conversion?
#Yes, seems like 1, 1a, 2 all indicate conversion. So maybe -999 just means raw data only...fine
df_final[df_final['PM_QC_level']=='2'].head()


# In[38]:


df_final.to_csv("Beacon_Data_Complete.csv", index=False)


# ## <font color='yellow'>Processing data to fit single-point and multi-point models</font>

# ### <font color='yellow'>Saving Beacon coordinates</font>

# In[39]:


df_final.head(1)


# In[40]:


df_beac = df_final


# In[41]:


beac_coords = df_beac[['node_name_long','lat','lng']].rename(columns={'lat':'latitude', 'lng':'longitude'}).drop_duplicates(keep='first').reset_index(drop=True)
beac_coords.to_csv("Beacon_Coordinates.csv", index=False)


# ### <font color='yellow'>General cleanup</font>
# Dropping NaNs, and subsetting data to sites that have relatively decent data (8 of 54 sites, roughly)

# In[42]:


#dropping nans
df_beac = df_beac[~df_beac['pm_pct_fs'].isna()].reset_index(drop=True)


# In[43]:


#subsetting beacon data to just have sites with >7000 observations for a given year
cutoff = 7000
most_data = df_beac['pm_pct_fs'].groupby(df_beac['node_name_long']).size().sort_values(ascending=False)
best_sites = list(most_data.index[most_data>cutoff])
df_beac_subset = df_beac[df_beac['node_name_long'].isin(best_sites)].reset_index(drop=True)


# In[44]:


#adding a date+hour column
dts = []
for i in range(len(df_beac_subset['Year'])):
    date = datetime.date(year=df_beac_subset['Year'][i],day=df_beac_subset['Day'][i],month=df_beac_subset['Month'][i])
    dts.append(datetime.datetime.combine(date, datetime.time(df_beac_subset['HourStarting'][i])))
df_beac_subset['datetime'] = dts


# In[45]:


df_beac_subset = df_beac_subset[['pm_pct_fs','node_name_long','datetime']]


# In[46]:


df_beac_subset.head(1)


# ### <font color='yellow'>Shaping data for single-point model</font>

# In[47]:


#Reshaping beacon data to have each predictor in a separate column
pivoted1 = df_beac_subset.pivot_table(index='datetime', columns='node_name_long', values='pm_pct_fs', aggfunc=np.max)
pivoted2 = df_beac_subset.pivot_table(index='datetime', columns='node_name_long', values='pm_pct_fs', aggfunc=np.min)


# In[48]:


#investigating the problem entries...looks like a duplicate in embarcadero
(pivoted1-pivoted2).sum(skipna=True)


# In[49]:


#smoothing over embarcadero issue; small potatoes
beac_piv = df_beac_subset.pivot_table(index='datetime', columns='node_name_long', values='pm_pct_fs', aggfunc=np.mean)


# In[50]:


beac_piv.head(1)


# In[51]:


#Saving dataframe formatted for single-point model
beac_piv.to_csv("Beacon_Data_SinglePointModel.csv")


# ### <font color='yellow'>Shaping data for multi-point model</font>

# In[52]:


df_beac = pd.read_csv("Beacon_Data_SinglePointModel.csv")


# In[53]:


df_beac.head(1)


# In[54]:


#Beacon data needs unpivoted
df_beac_unpiv = df_beac.melt(id_vars = ['datetime'], var_name='node_name_long', value_name='poll_conc')
df_beac = df_beac_unpiv.merge(beac_coords, on='node_name_long', how='left')


# In[55]:


df_beac.head(1)


# In[56]:


#Dataframe structure needs made consistent with others
cols = ['datetime','latitude', 'longitude', 'node_name_long', 'poll_conc']
df_beac = df_beac[cols].rename(columns={'node_name_long':'name', 'poll_conc':'beac_data'})


# In[57]:


df_beac.head(1)


# In[58]:


df_beac.to_csv("Beacon_Data_MultiPointModel.csv", index=False)


# In[ ]:




