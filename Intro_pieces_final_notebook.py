# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
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

pd.set_option('display.max_columns', 500)


# %%
#Setting directory where csv files live, to make this document easier to pass between us
filehome = "/Users/margaretmccall/Documents/2019 Fall/ER 131- Data Enviro Soc/Data"

# %% [markdown]
# 
# 
# ## Input Data Description (5 points)
# Here you will provide an initial description of your data sets, including:
# 1. The origins of your data.  Where did you get the data?  How were the data collected from the original sources?
# 2. The structure, granularity, scope, temporality and faithfulness (SGSTF) of your data.  To discuss these attributes you should load the data into one or more data frames (so you'll start building code cells for the first time).  At a minimum, use some basic methods (`.head`, `.loc`, and so on) to provide support for the descriptions you provide for SGSTF. 
# 
# [Chapter 5](https://www.textbook.ds100.org/ch/05/eda_intro.html) of the DS100 textbook might be helpful for you in this section.
# 
# 
# %% [markdown]
# For our project we used the following data sets:
# 
# 1. Purple Air Data
# 2. Berkeley Enviro Air-Quality and CO2 Observation Network (BEACON)
# 3. EPA
# 4. NOAA Weather Data
# 5. Trucking Data from the Freight Analysis Framework
# %% [markdown]
# ### Purple Air:
# 
# We obtained historic Purple Air data through two workflows. First, we used a python script to create a scraped list (using a REST API) of every purple air sensor including their Channel IDs, read key numbers, lat/lon, and device location (outside/inside). We filtered for outside devices and ported the data into R where we performed a spatial subset to isolate only sensors in California. We ported the data back into python and used another script with a REST API to query the ThingSpeak database which hosts all of the historic purple air data. The data was initially collected from discrete sensors - each purple air sensor has two channels (A and B). Purple Air takes readings from these channels every 120 seconds and stores them via the ThingSpeak API, a platform used to host large amounts of data. Below we read in our data:

# %%
df_pa = pd.read_csv("pa_full.csv")
df_pa.rename(columns={"lat":"latitude","lon":"longitude"})
df_pa.head(10)

df_pa['PM2.5 (ATM)'].plot.hist(bins=50,range=(0,100))

# %% [markdown]
# Let's first consider the granularity of our data. We have 8 columns which represent a datetime, PM1.0, PM2.5, PM10.0, PM2.5 with a correction factor, lat, lon, and id numbers.
#     Each cell of our datetime column contains the datatime in ___ format (UNX/UTC?). Each row in this dataframe represent a measurement from a CA sensor
#     at a given hour throughout the year. Each cell of our lat and lon columns provide the discrete latitude and longitude for where the specific sensor is
#     located. The ID numbers provide the Purple Air identification for that particular sensor. The readings for the air quality are in the standard ug/m3 format.
# 
#   The structure of this data set is an 8 column by 4424491 row dataframe which we read in from a CSV file. 
# 
# With respect to scope, because we subset the entire universe of PA sensors by just those in CA, we know that the scope is adequate for our purposes.
# 
# The temporality of the purple air data set is also adequate for our purposes. Because 2018 was the last full calendar year in which we have access to 12 
#     months of data, we were sure to query the ThingSpeak API for all 2018 data from the subset of California sensors. 
#     The datatime column provides the year, month, day, and hour of when the specific row (reading) occured. 
# 
# As for faithfulness, we do believe it captures reality. Given how much information is collected from these sensors every 120 seconds, it's unlikely that any
#     of the values we have here are hand-entered or falsified. Additionally, the actual cells themselves seem to reflect realistic or accurate findings.
# 
# Our histogram plotted above is meant to illustrate the range of values for one of our features of interest. In this case, an expected long tailed distribution
# is created. Again, based on the results of the visualization, it appears as though the data has not been tampered with and represents reality. 
# %% [markdown]
# ### BEACON:
# 
# 1. We obtained Beacon data from http://www.beacon.berkeley.edu/. Beacon is a small Berkeley-run network of lower-quality air pollution sensors that seems to have some of the same benefits as PurpleAir, in that they make a tradeoff between sensor quality and density. Most sensors are in the Bay Area, but some are in NYC and Houston. <p> Beacon has the feeling of being a much smaller operation than any of the other data sources we used--in addition to having a small sensor network in the first place, we had to download data (in CSV form) from each of the ~80 sensors by hand--there was no API or preformatted data download.
# 
# 2. In terms of SGSTF, the Beacon data have some drawbacks, which we discovered when we first started looking through the data.<p>
#     <b>Structure:</b> The data were downloadable as CSV files that had air pollution readings in a certain timeframe for an individual sensor. Each row is an hourly average value of air pollution. The air pollution readings given are for particulate matter (PM) and CO2. To combine the data into the raw structure shown below, we read in each individual CSV file (75 total), omitting those that threw an error for being empty (21 in total). We concatenated the remaining 56 into a dataframe, which we then merged with a metadata df containing lat/lon information and the names of different sites. This merged df, with irrelevant columns dropped, is shown below. <p> The columns represent the following data: <p>
#     * Year, Month, Day, HourStarting: columns representing date and time in UTC, pulled from original datetime column. Format: integer
#     * node_id: a unique ID for each sensor. Format: integer
#     * pm_pct_fs: the raw measurement from the sensor instrument--this number is proportional to the percentage of time scattering is detected by the unit. Format: float
#     * PM_QC_level: the quality control level of the PM record (0 = raw data; 1 = raw data*18.9 as rough conversion to ug/m3; 1a = level 1 with an offset; 2 = 1a but inspected for quality. -999 and -111 seem to reflect raw data as well as we explored, but are not explicitly stated to do so). Format: integer OR string (fixed!)
#     * PM_ug/m3: if the raw data has been converted to units of micrograms per cubic meter (standard air pollution units). -999 value implies no conversion. Format: integer OR string (fixed!)
#     * node_name_long: name of the sensor. Matches with node_id. Format: string
#     * lat: latitude. Format: float
#     * lng: longitude. Format: float
#     * height_above_ground: how far sensor is above ground (units unknown). Not ultimately used in analysis. Format: float
#     * height_above_sea: how far sensor is above sea (units unknown). Not ultimately used in analysis. Format: float
# </p><p><p>
#     
# <b>Granularity:</b> Each record in our dataset represents a particular hour at which an hourly air pollution average is given at a particular sensor. The original data were aggregated into the outputs shown in this table according to this process: "Averages have been calculated by taking measurements for the whole hour, then assigning them to beginning of the hour. So 12 AM will include measurements from 12:00:00-12:59:59." ***do they capture it at same level??***<p>
# <b>Scope:</b> The scope of the data is where the trouble starts. Geographically, they only cover the Bay Area, whereas we're interested in the whole of California; however, we knew that would be a limitation going in. Temporally, however, we discover that huge chunks of data are missing for most of the sensors. There should be 8760 hourly observations (at least for raw data) in a complete year, but only 8 of the 54 sensors had more than 7000 observations. (We chose this level of data completeness pretty arbitarily, so that we could at least keep some sensors but not lose too much data.) See below for a rough depiction of the air pollution levels recorded over time at each sensor--this was enough to encourage us to ditch most of them.<p>
# <b>Temporality:</b> As mentioned, we were working with UTC data to begin with, which represent air pollution measurements from a given hour. We subsetted the data to include 2018 only (since the data pulled had some stragglers from surrounding years). As shown below, no values are out of the ordinary.<p>
# <b>Faithfulness:</b> The notable faithfulness issue is that even though raw data (PM_QC_level = 0, or -999 or -111) should be easily converted to PM_ug/m3 (and a PM_QC_level of 1) by multiplying by 18.9, this wasn't always done; there are fewer missing raw data points than there are points that have failed to be converted to PM_ug/m3. As such, we will use the raw data in our model. We also spot-checked level-1 PM_ug/m3 readings against pm_pct_fs readings to see if they were roughly 18.9x bigger, as expected.

# %%
#Structure
df_beac_raw = pd.read_csv(filehome+"/Beacon_Data_RAW.csv")
df_beac_raw.head(3)


# %%
#Scope
plt.figure(figsize=(15,20))
stations = df_beac_raw['node_name_long'].unique()

for i in range(len(stations)):
    dat = df_beac_raw[df_beac_raw['node_name_long']==stations[i]]
    timeidx = dat['Month']+dat['Day']/30+dat['HourStarting']/(30*24)
    plt.subplot(9,6,i+1)
    plt.plot(timeidx,dat['pm_pct_fs'],'ob')
    plt.title(stations[i])
    plt.xlim(1,13)
    plt.ylim(0,50)


# %%
#Temporality
for i in ['Month','Day','Year','HourStarting']:
    print(df_beac_raw[i].unique())


# %%
#Faithfulness
print("Number of missing raw data points:", sum(df_beac_raw['pm_pct_fs'].isna()))
df_beac_raw['PM_QC_level'].groupby(df_beac_raw['PM_QC_level']).size()

# %% [markdown]
# ### EPA:
# 
# 1. We collected data from the EPA via a REST API. The API is used to collected data from the EPA's Air Quality System (AQS) database. From the EPA's website: "AQS contains ambient air sample data collected by state, local, tribal, and federal air pollution control agencies from thousands of monitors around the nation. It also contains meteorological data, descriptive information about each monitoring station (including its geographic location and its operator), and information about the quality of the samples."
# 2. SGSTF: <p>
# **Structure**: The data was originally obtained as a single JSON file for each month. In order to work with the data more easily, we used a function called json_normalize to convert to a dataframe as we read in each file, which we then appended together. After this simple conversion, the records were neatly organized in rows. **The fields of relevance are described further in the data cleaning section, but are date_gmt, time_gmt, sample_measurement, sample_frequency, latitude, and longitude.** (GOTTA DESCRIBE)<p>
# **Granularity**: Each record represents a time period at a specific sensor site at which an air pollution sample was taken. However, the raw data contains two columns--sample_duration and sample_frequency--that differ across different observations. We are interested in samples taken every hour, and it seems that if we select observations with sample_duration=='1 HOUR', that corresponds almost exclusively to data points with sample_frequency=='HOURLY' except for one point that appears to be mislabeled 'SEASONAL'.<p>
# **Scope**: <p>
# **Temporality**: <p>
# **Faithfulness**:
# 

# %%
#Structure
df_epa_raw = pd.read_csv(filehome+"/EPA_Data_Complete.csv")
df_epa_raw.head(3)


# %%
#Granularity
print(df_epa_raw['sample_frequency'].unique())
print(df_epa_raw['sample_duration'].unique())
print(df_epa_raw['sample_frequency'][df_epa_raw['sample_duration']=='1 HOUR'].unique())

# %% [markdown]
# Our dataframe consists of 5 columns and 67801 rows pulled from a CSV file. Because the data was extracted via a REST API, we were able to pull only 2018 data from California. The data set includes the latitude and longitude from each of the 119 distinct EPA sensors across CA. Each observation (row) represents a reading from a sensor at a specific hour throughout 2018. The PM2.5 reading is given in the column "epa_meas". That column is what we are ultimately going to be trying to forecast. Our histogram of the EPA data shows another right tailed distribution. Interestingly, there appears to be one bin in particular that is much more frequent than the rest.
# %% [markdown]
# ### NOAA
# 
# Our NOAA data (like our other data) was collected through a REST API. NOAA maintains historic information regarding weather, including wind direction, wind speets, air pressure,
# temperature.  
# 
# more
# 
# info
# 
# here
# 
# !
# 
# !

# %%


# %% [markdown]
# ### Trucking data
# 1. We obtained truck data from the Freight Analysis Framework (FAF) model from the Bureau of Transportation Statistics. FAF data is suitable for geospatial analysis, and contains historical and predicted figures relevant to freight movement (e.g., annual number of freight trucks, non-freight trucks, tons of goods by truck, etc) for each highway segment. We ultimately only used this trucking data in our multi-point model--since it doesn't vary in time (the values presented are annual averages), we realized that it has no explanatory power when included in the single-point model.
# 
# 2. SGSTF: <p>
#     **Structure**: The data was initially read in as a shapefile, and then a separate DBF dataframe containing attributes associated with each linestring in the shapefile was read in and merged. The DBF file was tabular already (I think) but was transformed into a dataframe to facilitate the merge. The relevant columns are addressed in the data cleaning section. <p>
#     **Granularity**: Each row of the truck data represents data for a specific linestring--that is, a geometric representation of a certain highway segment. While all rows are at this level of granularity, the length of each segment of highway may be different, so the features are not comparable on their face.<p>
#     **Scope**: The initial dataset is country-wide and extends into some parts of Canada as well. <p>
#     **Temporality**: This data is aggregated data (in fact, estimated population data based on sample data) representing truck traffic in 2012. Forward-looking estimates for 2045 were also available in this dataset <p>
#     **Faithfulness**: The main faithfulness check that was feasible to do was to determine whether the length of the highway segment given in the 'LENGTH' column corresponded to the difference between the BEGMP and ENDMP (start and end mile marker) columns, which it did.

# %%
#Structure
df_trucks_raw = pd.read_csv(filehome+"/Truck_Data_RAW.csv")
df_trucks_raw.head(3)

# %% [markdown]
# ## Data Cleaning (10 points)
# In this section you will walk through the data cleaning and merging process.  Explain how you make decisions to clean and merge the data.  Explain how you convince yourself that the data don't contain problems that will limit your ability to produce a meaningful analysis from them.  
# 
# [Chapter 4](https://www.textbook.ds100.org/ch/04/cleaning_intro.html) of the DS100 textbook might be helpful to you in this section.  
# 
# 

# %%

#Jake note: After re reading this section's expectations and skimming the python scripts that clean the data, it seems like all we
#need here is a written explanation of how we cleaned the data? I can't imagine Duncan actually wants us to throw our cleaning code here.



# %% [markdown]
# Below we describe the separate data cleaning process for each of our five datasets. The PurpleAir and the Beacon datasets definitely required the most cleaning/processing compared to the three government-sourced datasets. Beacon in particular had a lot of missing data.
# 
# The data merging process was straightforward and consistent across each of the following datasets: we merged the data in two different ways to accommodate two different models. For the single-point prediction model, we had a row for each unique hour in 2018, and merged each dataframe on that datetime column such that each row was a unique hour and each column was a unique sensor. For the multi-point prediction model, we merged the datasets on which EPA sensor they were associated with, obtaining a dataframe with a row for each EPA sensor and a column for the value of each other variable at that sensor's location.
# 
# 
# **Formatting**:
# 
# For each dataset that was going to be fed into both the single-point and the multi-point model, we formatted it in two different ways. For the single-point model, we created a "pivoted" dataframe, with a column for each individual sensor and type of data (e.g., a column for each NOAA sensor with precipitation data), and a row for each hour. For the multi-point model, we created a "melted" dataframe, which was easier to pass into a KNN function to get the value of each variable at each EPA sensor point.
# 
# The format of each pivoted dataframe was to have the same first four columns to feed into the multi-point model functions: a column for datetime, latitude, longitude, and the sensor name (for readability). The subsequent columns should each contain a unique dataset. For the single-point model, each dataframe had a datetime column first, and then a column for each unique data type at each unique sensor.

# %%
sampl = pd.read_csv(filehome+"/NOAA_Data_SinglePointModel.csv")
sampl.head()

# %% [markdown]
# ### PurpleAir
# ((Nick / Jake))
# %% [markdown]
# ### BEACON
# 
# Much of the Beacon data-cleaning process was described in the first section: essentially, we
# * Downloaded and concatenated all CSV files by hand, leaving us with 54 sensors with data in 2018; 
# * Merged dataframe with metadata to attach lat/lon data to each observation;
# * Dropped data from sensors that had <7000 hourly observations for the year, leaving us with 8 sensors;
# * Looked at a timeseries of the 8 sensors to see if anything looked amiss, and dropped another sensor (Laney College) that seemed to have a lot of observations with no variation (suggesting a faulty sensor)
# 
# After finishing our initial data exploration, our main concern was in getting sensors with decent data quality, which we believe to have accomplished with this process.
# %% [markdown]
# ### EPA
# The principal cleaning that we did to the EPA data was: 
# * Subset it to contain observations where the sample_duration was one hour;
# * Reformat the datetime column to be in a consistent format with other data frames;
# * Drop all unnecessary columns, keeping only datetime, lat, lon, name, and the pollution measurement
# 
# For the single-point model, the goal of which is to try to predict pollution at the location of a single EPA sensor, we chose a sensor in the Bay Area in order to be able to leverage the Beacon sensors in the model. To do this, we plotted the location of the Beacon sensors and EPA sensors, and visually designated a box from within which we selected an EPA sensor, using geopandas operations and then taking the first EPA sensor in the list. See graph below for some of the process.

# %%
beac_coords = pd.read_csv(filehome+"/Beacon_Coordinates.csv")
beac_coords = list(set(list(zip(beac_coords['latitude'],beac_coords['longitude']))))
epa_coords = list(set(list(zip(df_epa_raw['latitude'],df_epa_raw['longitude']))))

beac_lons, beac_lats, epa_lons, epa_lats = [], [], [], []
for i in range(len(beac_coords)):
    beac_lats.append(beac_coords[i][0])
    beac_lons.append(beac_coords[i][1]) 
for i in range(len(epa_coords)):
    epa_lats.append(epa_coords[i][0])
    epa_lons.append(epa_coords[i][1])


# %%
plt.plot(epa_lons, epa_lats, 'oy', label='EPA Sensors')
plt.plot(beac_lons, beac_lats, 'ob', label='Beacon Sensors')
plt.plot(-122.520004, 37.97231, '*r', label='Selected EPA Sensor')
plt.xlim(-123,-121)
plt.ylim(37,39.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# %% [markdown]
# ### NOAA
# 
# 
# ((Nick/Jake))
# 
# %% [markdown]
# ### Trucking data
# 
# The main data cleaning and manipulation we did on the truck data was as follows:
# * We merged the geodataframe containing the highway linestrings with the data table containing information about truck traffic at those locations based on the column FAF4_ID (as instructed via the FAF website); 
# * We subsetted the data to be specific to California;
# * We chose a few columns with data that seemed particularly relevant and maybe not too repetitive (all numbers are averages for 2012): AADT12 (average annual daily traffic on this segment), FAF12 (the number of FAF freight trucks on this segment daily), NONFAF12 (same, but non-freight trucks), YKTON12 (thousands of tons of freight daily), KTONMILE12 (thousands of ton-miles of freight daily).
# * We then created three buffers--100m, 1000m, and 5000m--and summed the value for each relevant trucking variable within those buffers at each EPA sensor site.
# %% [markdown]
# ## Data Summary and Exploratory Data Analysis (10 points)
# 
# In this section you should provide a tour through some of the basic trends and patterns in your data.  This includes providing initial plots to summarize the data, such as box plots, histograms, trends over time, scatter plots relating one variable or another.  
# 
# [Chapter 6](https://www.textbook.ds100.org/ch/06/viz_intro.html) of the DS100 textbook might be helpful for providing ideas for visualizations that describe your data.  
# %% [markdown]
# ### PurpleAir

# %%
df_pa.head(10)


# %%
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(df_pa["PM2.5 (ATM)"],df_pa["PM2.5 (CF=1)"])
plt.title("PM2.5 with no CF relative to PM2.5 with CF")
plt.xlabel("PM2.5 with no correction factor")
plt.ylabel("PM2.5 with correction factor")

plt.subplot(1,3,2)
plt.plot(df_pa["PM2.5 (ATM)"],df_pa["PM1.0 (ATM)"])
plt.title("Visualization of PM2.5 relative to PM1.0")
plt.xlabel("PM2.5")
plt.ylabel("PM1.0")

plt.subplot(1,3,3)
plt.plot(df_pa["PM2.5 (ATM)"],df_pa["PM10.0 (ATM)"])
plt.title("Visualization of PM2.5 relative to PM10.0")
plt.xlabel("PM2.5")
plt.ylabel("PM10.0")

# %% [markdown]
# Our visualizations reveal a few notable insights. Looking at our comparison of PM2.5 to the PM2.5 with a correction factor,
# it appears as though a constant correction factor is applied across the board (i.e., the slope of the line is constant). 
# This might mean that purple air sensors are all initially biased in the same direction. Our second visualization comparing PM2.5
# to PM1.0 shows that for PM2.5 values above 2000, the relationship to PM1.0 is constant. Interestingly se see a lot of noise below 
# 2000 on the x axis, meaning that there might be some threshold of PM2.5 that we'd need to see before we see a constant linear relationship
# between the two variables. Our last comparison of PM2.5 to PM10.0 reveals a similar trend. Let's view these relationships in a different format:

# %%
pa_sample = df_pa.sample(n=60,random_state=1)

plt.scatter(pa_sample["datetime"], pa_sample["PM1.0 (ATM)"], s = 5, color = 'c')
plt.title("2018 PM 1.0 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

plt.scatter(pa_sample["datetime"], pa_sample["PM10.0 (ATM)"], s = 5, color = 'c')
plt.title("2018 PM 10.0 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

plt.scatter(pa_sample["datetime"], pa_sample["PM2.5 (CF=1)"], s = 5, color = 'c')
plt.title("2018 PM 2.5 with CF readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

# %% [markdown]
# The scatter plots above show us the distribution of our random sample of Purple Air PM readings over 2018. Let's look at the raw scatter plot of PM2.5 readings with no correction factor included:

# %%
plt.scatter(pa_sample["datetime"], pa_sample["PM2.5 (ATM)"], s = 5, color = 'c')
plt.title("2018 PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

# %% [markdown]
# Based on our scatterplot, we see that most of our readings from this sample fall between 0 and 20 throughout the year. There are
# values that exceed 20 here and there, but they seem to be on the higher side of the sample. If we plot another histogram of our sample
# we can see this:

# %%
pa_sample['PM2.5 (ATM)'].plot.hist(bins=60,range=(0,70))

# %% [markdown]
# ### Beacon

# %%
# The structure of this data set is a 5 column and 70080 row dataframe, which we read in from a CSV file. We have a lat, lon, name of location, datetime, and reading from the individual node. Each row represents a reading at a certain location at a certain hour during 2018. The Lat and Lon data correspond to where the node is located. The scope of this data almost perfectly matches our projct given that each CA sensor is within the bay area. It would have been better if there were BEACON nodes across the entire state, but this may help us getting an accurate forecast for some of our areas nearby. Much like the PurpleAir data, because the data is collected from sensors, there appears to be little reason to doubt the reliability of what we've collected. The lat and lons seem to make sense,and the actual readings themselves match with what we'd expect as well.The temporality is also adequate, the data contained within is for 2018. Our histogram of the beacon data also shows a similar long-tailed distribution of our feature of interest. 


# %%
df_beac = pd.read_csv("Beacon_Data_MultiPointModel.csv")
df_beac.head(10)
df_beac['beac_data'].plot.hist(bins=50,range=(0,10))

# %% [markdown]
# ### EPA
# 

# %%
df_epa = pd.read_csv("EPA_Data_MultiPointModel.csv")
df_epa.head(10)
df_epa['epa_meas'].plot.hist(bins=100,range=(0,120))

# %% [markdown]
# We included this histogram above, but we include it again to higlight the distribution of our EPA readings that we are interested in forecasting. 
# 
# 
# 
# 

# %%
epa_sample = df_epa.sample(n=60,random_state=1)

plt.scatter(epa_sample["datetime"], epa_sample["epa_meas"], s = 5, color = 'c')
plt.title("2018 EPA PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

# %% [markdown]
# Because our data set is so large, we needed to take a random sample to actually plot the readings over the time of the year. It appears as though the readings are mostly consistent throughout the year with one outlier in the middle of the year, and a few higher
# points scattered throughout.
# %% [markdown]
# ### NOAA

# %%
# Our dataframe consists of 9 columns and 1033680 observations spread across CA in 2018. The datetimes in this particular frame are for each hour at different weather 
#stations throughout CA. Each row (observation) is a measurement taken at a specific station for a given hour during the year.


# %%
df_noaa = pd.read_csv("NOAA_Data_MultiPointModel.csv")
df_noaa.head(10)
df_noaa['wind_speed_set_1'].plot.hist(bins=50,range=(0,20))
df_noaa['air_temp_set_1'].plot.hist(bins=50,range=(0,20))

# %% [markdown]
# ### Trucking

# %%
df_truck = pd.read_csv("Truck_Data_MultiPointModel.csv")
df_truck.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)
df_truck = df_truck.rename(columns={'0':'latitude','1':'longitude'})
df_truck['datetime'] = 'none'
df_truck['name'] = 'none'
cols = ['datetime', 'latitude', 'longitude', 'name', '100mAADT12', '100mFAF12', '100mNONFAF12',
       '100mYKTON12', '100mKTONMILE12', '1000mAADT12', '1000mFAF12',
       '1000mNONFAF12', '1000mYKTON12', '1000mKTONMILE12', '5000mAADT12',
       '5000mFAF12', '5000mNONFAF12', '5000mYKTON12', '5000mKTONMILE12']
df_truck = df_truck[cols]
df_truck.head(10)

