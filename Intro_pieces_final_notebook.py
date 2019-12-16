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
#  ## Input Data Description (5 points)
#  Here you will provide an initial description of your data sets, including:
#  1. The origins of your data.  Where did you get the data?w  How were the data collected from the original sources?
#  2. The structure, granularity, scope, temporality and faithfulness (SGSTF) of your data.  To discuss these attributes you should load the data into one or more data frames (so you'll start building code cells for the first time).  At a minimum, use some basic methods (`.head`, `.loc`, and so on) to provide support for the descriptions you provide for SGSTF.
# 
#  [Chapter 5](https://www.textbook.ds100.org/ch/05/eda_intro.html) of the DS100 textbook might be helpful for you in this section.
# 
# 
# %% [markdown]
#  For our project we used the following data sets:
# 
#  1. Purple Air Data
#  2. Berkeley Enviro Air-Quality and CO2 Observation Network (BEACON)
#  3. EPA
#  4. NOAA Weather Data
#  5. Trucking Data from the Freight Analysis Framework
# %% [markdown]
#  ### Purple Air:
# 
#  We obtained historic Purple Air data through two workflows. First, we used a python script to create a scraped list (using a REST API) of every purple air sensor including their Channel IDs, read key numbers, lat/lon, and device location (outside/inside). We filtered for outside devices and ported the data into R where we performed a spatial subset to isolate only sensors in California. We ported the data back into python and used another script with a REST API to query the ThingSpeak database which hosts all of the historic purple air data. The data was initially collected from discrete sensors - each purple air sensor has two channels (A and B). Purple Air takes readings from these channels every 120 seconds and stores them via the ThingSpeak API, a platform used to host large amounts of data. Below we read in our data:

# %%
df_pa = pd.read_csv("pa_full.csv")
df_pa.rename(columns={"lat":"latitude","lon":"longitude"})
df_pa.head(10)

df_pa['PM2.5 (ATM)'].plot.hist(bins=50,range=(0,100))

# %% [markdown]
#  Let's first consider the granularity of our data. We have 8 columns which represent a datetime, PM1.0, PM2.5, PM10.0, PM2.5 with a correction factor, lat, lon, and id numbers.
#      Each cell of our datetime column contains the datatime in ___ format (UNX/UTC?). Each row in this dataframe represent a measurement from a CA sensor
#      at a given hour throughout the year. Each cell of our lat and lon columns provide the discrete latitude and longitude for where the specific sensor is
#      located. The ID numbers provide the Purple Air identification for that particular sensor. The readings for the air quality are in the standard ug/m3 format.
# 
#    The structure of this data set is an 8 column by 4424491 row dataframe which we read in from a CSV file.
# 
#  With respect to scope, because we subset the entire universe of PA sensors by just those in CA, we know that the scope is adequate for our purposes.
# 
#  The temporality of the purple air data set is also adequate for our purposes. Because 2018 was the last full calendar year in which we have access to 12
#      months of data, we were sure to query the ThingSpeak API for all 2018 data from the subset of California sensors.
#      The datatime column provides the year, month, day, and hour of when the specific row (reading) occured.
# 
#  As for faithfulness, we do believe it captures reality. Given how much information is collected from these sensors every 120 seconds, it's unlikely that any
#      of the values we have here are hand-entered or falsified. Additionally, the actual cells themselves seem to reflect realistic or accurate findings.
# 
#  Our histogram plotted above is meant to illustrate the range of values for one of our features of interest. In this case, an expected long tailed distribution
#  is created. Again, based on the results of the visualization, it appears as though the data has not been tampered with and represents reality.
# %% [markdown]
#  ### BEACON:
# 
#  1. We obtained Beacon data from http://www.beacon.berkeley.edu/. Beacon is a small Berkeley-run network of lower-quality air pollution sensors that seems to have some of the same benefits as PurpleAir, in that they make a tradeoff between sensor quality and density. Most sensors are in the Bay Area, but some are in NYC and Houston. <p> Beacon has the feeling of being a much smaller operation than any of the other data sources we used--in addition to having a small sensor network in the first place, we had to download data (in CSV form) from each of the ~80 sensors by hand--there was no API or preformatted data download.
# 
#  2. In terms of SGSTF, the Beacon data have some drawbacks, which we discovered when we first started looking through the data.<p>
#      <b>Structure:</b> The data were downloadable as CSV files that had air pollution readings in a certain timeframe for an individual sensor. Each row is an hourly average value of air pollution. The air pollution readings given are for particulate matter (PM) and CO2. To combine the data into the raw structure shown below, we read in each individual CSV file (75 total), omitting those that threw an error for being empty (21 in total). We concatenated the remaining 56 into a dataframe, which we then merged with a metadata df containing lat/lon information and the names of different sites. This merged df, with irrelevant columns dropped, is shown below. <p> The columns represent the following data: <p>
#      * Year, Month, Day, HourStarting: columns representing date and time in UTC, pulled from original datetime column. Format: integer
#      * node_id: a unique ID for each sensor. Format: integer
#      * pm_pct_fs: the raw measurement from the sensor instrument--this number is proportional to the percentage of time scattering is detected by the unit. Format: float
#      * PM_QC_level: the quality control level of the PM record (0 = raw data; 1 = raw data*18.9 as rough conversion to ug/m3; 1a = level 1 with an offset; 2 = 1a but inspected for quality. -999 and -111 seem to reflect raw data as well as we explored, but are not explicitly stated to do so). Format: integer OR string (fixed!)
#      * PM_ug/m3: if the raw data has been converted to units of micrograms per cubic meter (standard air pollution units). -999 value implies no conversion. Format: integer OR string (fixed!)
#      * node_name_long: name of the sensor. Matches with node_id. Format: string
#      * lat: latitude. Format: float
#      * lng: longitude. Format: float
#      * height_above_ground: how far sensor is above ground (units unknown). Not ultimately used in analysis. Format: float
#      * height_above_sea: how far sensor is above sea (units unknown). Not ultimately used in analysis. Format: float
#  </p><p><p>
# 
#  <b>Granularity:</b> Each record in our dataset represents a particular hour at which an hourly air pollution average is given at a particular sensor. The original data were aggregated into the outputs shown in this table according to this process: "Averages have been calculated by taking measurements for the whole hour, then assigning them to beginning of the hour. So 12 AM will include measurements from 12:00:00-12:59:59." ***do they capture it at same level??***<p>
#  <b>Scope:</b> The scope of the data is where the trouble starts. Geographically, they only cover the Bay Area, whereas we're interested in the whole of California; however, we knew that would be a limitation going in. Temporally, however, we discover that huge chunks of data are missing for most of the sensors. There should be 8760 hourly observations (at least for raw data) in a complete year, but only 8 of the 54 sensors had more than 7000 observations. (We chose this level of data completeness pretty arbitarily, so that we could at least keep some sensors but not lose too much data.) See below for a rough depiction of the air pollution levels recorded over time at each sensor--this was enough to encourage us to ditch most of them.<p>
#  <b>Temporality:</b> As mentioned, we were working with UTC data to begin with, which represent air pollution measurements from a given hour. We subsetted the data to include 2018 only (since the data pulled had some stragglers from surrounding years). As shown below, no values are out of the ordinary.<p>
#  <b>Faithfulness:</b> The notable faithfulness issue is that even though raw data (PM_QC_level = 0, or -999 or -111) should be easily converted to PM_ug/m3 (and a PM_QC_level of 1) by multiplying by 18.9, this wasn't always done; there are fewer missing raw data points than there are points that have failed to be converted to PM_ug/m3. As such, we will use the raw data in our model. We also spot-checked level-1 PM_ug/m3 readings against pm_pct_fs readings to see if they were roughly 18.9x bigger, as expected.

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
#  ### EPA:
# 
#  1. We collected data from the EPA via a REST API. The API is used to collected data from the EPA's Air Quality System (AQS) database. From the EPA's website: "AQS contains ambient air sample data collected by state, local, tribal, and federal air pollution control agencies from thousands of monitors around the nation. It also contains meteorological data, descriptive information about each monitoring station (including its geographic location and its operator), and information about the quality of the samples."
#  2. SGSTF: <p>
#  **Structure**: The data was originally obtained as a single JSON file for each month. In order to work with the data more easily, we used a function called json_normalize to convert to a dataframe as we read in each file, which we then appended together. After this simple conversion, the records were neatly organized in rows. **The fields of relevance are described further in the data cleaning section, but are date_gmt, time_gmt, sample_measurement, sample_frequency, latitude, and longitude.** (GOTTA DESCRIBE)<p>
#  **Granularity**: Each record represents a time period at a specific sensor site at which an air pollution sample was taken. However, the raw data contains two columns--sample_duration and sample_frequency--that differ across different observations. We are interested in samples taken every hour, and it seems that if we select observations with sample_duration=='1 HOUR', that corresponds almost exclusively to data points with sample_frequency=='HOURLY' except for one point that appears to be mislabeled 'SEASONAL'.<p>
#  **Scope**: <p>
#  **Temporality**: <p>
#  **Faithfulness**:
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
#  Our dataframe consists of 5 columns and 67801 rows pulled from a CSV file. Because the data was extracted via a REST API, we were able to pull only 2018 data from California. The data set includes the latitude and longitude from each of the 119 distinct EPA sensors across CA. Each observation (row) represents a reading from a sensor at a specific hour throughout 2018. The PM2.5 reading is given in the column "epa_meas". That column is what we are ultimately going to be trying to forecast. Our histogram of the EPA data shows another right tailed distribution. Interestingly, there appears to be one bin in particular that is much more frequent than the rest.
# %% [markdown]
#  ### NOAA
# 
#  Our NOAA data (like our other data) was collected through a REST API. NOAA maintains historic information regarding weather, including wind direction, wind speets, air pressure,
#  temperature.
# 
#  more
# 
#  info
# 
#  here
# 
#  !
# 
#  !

# %%


# %% [markdown]
#  ### Trucking data
#  1. We obtained truck data from the Freight Analysis Framework (FAF) model from the Bureau of Transportation Statistics. FAF data is suitable for geospatial analysis, and contains historical and predicted figures relevant to freight movement (e.g., annual number of freight trucks, non-freight trucks, tons of goods by truck, etc) for each highway segment. We ultimately only used this trucking data in our multi-point model--since it doesn't vary in time (the values presented are annual averages), we realized that it has no explanatory power when included in the single-point model.
# 
#  2. SGSTF: <p>
#      **Structure**: The data was initially read in as a shapefile, and then a separate DBF dataframe containing attributes associated with each linestring in the shapefile was read in and merged. The DBF file was tabular already (I think) but was transformed into a dataframe to facilitate the merge. The relevant columns are addressed in the data cleaning section. <p>
#      **Granularity**: Each row of the truck data represents data for a specific linestring--that is, a geometric representation of a certain highway segment. While all rows are at this level of granularity, the length of each segment of highway may be different, so the features are not comparable on their face.<p>
#      **Scope**: The initial dataset is country-wide and extends into some parts of Canada as well. <p>
#      **Temporality**: This data is aggregated data (in fact, estimated population data based on sample data) representing truck traffic in 2012. Forward-looking estimates for 2045 were also available in this dataset <p>
#      **Faithfulness**: The main faithfulness check that was feasible to do was to determine whether the length of the highway segment given in the 'LENGTH' column corresponded to the difference between the BEGMP and ENDMP (start and end mile marker) columns, which it did.

# %%
#Structure
df_trucks_raw = pd.read_csv(filehome+"/Truck_Data_RAW.csv")
df_trucks_raw.head(3)

# %% [markdown]
#  ## Data Cleaning (10 points)
#  In this section you will walk through the data cleaning and merging process.  Explain how you make decisions to clean and merge the data.  Explain how you convince yourself that the data don't contain problems that will limit your ability to produce a meaningful analysis from them.
# 
#  [Chapter 4](https://www.textbook.ds100.org/ch/04/cleaning_intro.html) of the DS100 textbook might be helpful to you in this section.
# 
# 

# %%

#Jake note: After re reading this section's expectations and skimming the python scripts that clean the data, it seems like all we
#need here is a written explanation of how we cleaned the data? I can't imagine Duncan actually wants us to throw our cleaning code here.


# %% [markdown]
#  Below we describe the separate data cleaning process for each of our five datasets. The PurpleAir and the Beacon datasets definitely required the most cleaning/processing compared to the three government-sourced datasets. Beacon in particular had a lot of missing data.
# 
#  The data merging process was straightforward and consistent across each of the following datasets: we merged the data in two different ways to accommodate two different models. For the single-point prediction model, we had a row for each unique hour in 2018, and merged each dataframe on that datetime column such that each row was a unique hour and each column was a unique sensor. For the multi-point prediction model, we merged the datasets on which EPA sensor they were associated with, obtaining a dataframe with a row for each EPA sensor and a column for the value of each other variable at that sensor's location.
# 
# 
#  **Formatting**:
# 
#  For each dataset that was going to be fed into both the single-point and the multi-point model, we formatted it in two different ways. For the single-point model, we created a "pivoted" dataframe, with a column for each individual sensor and type of data (e.g., a column for each NOAA sensor with precipitation data), and a row for each hour. For the multi-point model, we created a "melted" dataframe, which was easier to pass into a KNN function to get the value of each variable at each EPA sensor point.
# 
#  The format of each pivoted dataframe was to have the same first four columns to feed into the multi-point model functions: a column for datetime, latitude, longitude, and the sensor name (for readability). The subsequent columns should each contain a unique dataset. For the single-point model, each dataframe had a datetime column first, and then a column for each unique data type at each unique sensor.

# %%
sampl = pd.read_csv(filehome+"/NOAA_Data_SinglePointModel.csv")
sampl.head()

# %% [markdown]
#  ### PurpleAir
#  ((Nick / Jake))
# %% [markdown]
#  ### BEACON
# 
#  Much of the Beacon data-cleaning process was described in the first section: essentially, we
#  * Downloaded and concatenated all CSV files by hand, leaving us with 54 sensors with data in 2018;
#  * Merged dataframe with metadata to attach lat/lon data to each observation;
#  * Dropped data from sensors that had <7000 hourly observations for the year, leaving us with 8 sensors;
#  * Looked at a timeseries of the 8 sensors to see if anything looked amiss, and dropped another sensor (Laney College) that seemed to have a lot of observations with no variation (suggesting a faulty sensor)
# 
#  After finishing our initial data exploration, our main concern was in getting sensors with decent data quality, which we believe to have accomplished with this process.
# %% [markdown]
#  ### EPA
#  The principal cleaning that we did to the EPA data was:
#  * Subset it to contain observations where the sample_duration was one hour;
#  * Reformat the datetime column to be in a consistent format with other data frames;
#  * Drop all unnecessary columns, keeping only datetime, lat, lon, name, and the pollution measurement
# 
#  For the single-point model, the goal of which is to try to predict pollution at the location of a single EPA sensor, we chose a sensor in the Bay Area in order to be able to leverage the Beacon sensors in the model. To do this, we plotted the location of the Beacon sensors and EPA sensors, and visually designated a box from within which we selected an EPA sensor, using geopandas operations and then taking the first EPA sensor in the list. See graph below for some of the process.

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
#  ### NOAA
# 
# 
#  ((Nick/Jake))
# 
# %% [markdown]
#  ### Trucking data
# 
#  The main data cleaning and manipulation we did on the truck data was as follows:
#  * We merged the geodataframe containing the highway linestrings with the data table containing information about truck traffic at those locations based on the column FAF4_ID (as instructed via the FAF website);
#  * We subsetted the data to be specific to California;
#  * We chose a few columns with data that seemed particularly relevant and maybe not too repetitive (all numbers are averages for 2012): AADT12 (average annual daily traffic on this segment), FAF12 (the number of FAF freight trucks on this segment daily), NONFAF12 (same, but non-freight trucks), YKTON12 (thousands of tons of freight daily), KTONMILE12 (thousands of ton-miles of freight daily).
#  * We then created three buffers--100m, 1000m, and 5000m--and summed the value for each relevant trucking variable within those buffers at each EPA sensor site.
# %% [markdown]
#  ## Data Summary and Exploratory Data Analysis (10 points)
# 
#  In this section you should provide a tour through some of the basic trends and patterns in your data.  This includes providing initial plots to summarize the data, such as box plots, histograms, trends over time, scatter plots relating one variable or another.
# 
#  [Chapter 6](https://www.textbook.ds100.org/ch/06/viz_intro.html) of the DS100 textbook might be helpful for providing ideas for visualizations that describe your data.
# %% [markdown]
#  ### PurpleAir

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
#  Our visualizations reveal a few notable insights. Looking at our comparison of PM2.5 to the PM2.5 with a correction factor,
#  it appears as though a constant correction factor is applied across the board (i.e., the slope of the line is constant).
#  This might mean that purple air sensors are all initially biased in the same direction. Our second visualization comparing PM2.5
#  to PM1.0 shows that for PM2.5 values above 2000, the relationship to PM1.0 is constant. Interestingly se see a lot of noise below
#  2000 on the x axis, meaning that there might be some threshold of PM2.5 that we'd need to see before we see a constant linear relationship
#  between the two variables. Our last comparison of PM2.5 to PM10.0 reveals a similar trend. Let's view these relationships in a different format:

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
#  The scatter plots above show us the distribution of our random sample of Purple Air PM readings over 2018. Let's look at the raw scatter plot of PM2.5 readings with no correction factor included:

# %%
plt.scatter(pa_sample["datetime"], pa_sample["PM2.5 (ATM)"], s = 5, color = 'c')
plt.title("2018 PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

# %% [markdown]
#  Based on our scatterplot, we see that most of our readings from this sample fall between 0 and 20 throughout the year. There are
#  values that exceed 20 here and there, but they seem to be on the higher side of the sample. If we plot another histogram of our sample
#  we can see this:

# %%
pa_sample['PM2.5 (ATM)'].plot.hist(bins=60,range=(0,70))

# %% [markdown]
#  ### Beacon

# %%
# The structure of this data set is a 5 column and 70080 row dataframe, which we read in from a CSV file. We have a lat, lon, name of location, datetime, and reading from the individual node. Each row represents a reading at a certain location at a certain hour during 2018. The Lat and Lon data correspond to where the node is located. The scope of this data almost perfectly matches our projct given that each CA sensor is within the bay area. It would have been better if there were BEACON nodes across the entire state, but this may help us getting an accurate forecast for some of our areas nearby. Much like the PurpleAir data, because the data is collected from sensors, there appears to be little reason to doubt the reliability of what we've collected. The lat and lons seem to make sense,and the actual readings themselves match with what we'd expect as well.The temporality is also adequate, the data contained within is for 2018. Our histogram of the beacon data also shows a similar long-tailed distribution of our feature of interest. 


# %%
df_beac = pd.read_csv("Beacon_Data_MultiPointModel.csv")
df_beac.head(10)
df_beac['beac_data'].plot.hist(bins=50,range=(0,10))

# %% [markdown]
#  ### EPA
# 

# %%
df_epa = pd.read_csv("EPA_Data_MultiPointModel.csv")
df_epa.head(10)
df_epa['epa_meas'].plot.hist(bins=100,range=(0,120))

# %% [markdown]
#  We included this histogram above, but we include it again to higlight the distribution of our EPA readings that we are interested in forecasting.
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
#  Because our data set is so large, we needed to take a random sample to actually plot the readings over the time of the year. It appears as though the readings are mostly consistent throughout the year with one outlier in the middle of the year, and a few higher
#  points scattered throughout.
# %% [markdown]
#  ### NOAA

# %%
# Our dataframe consists of 9 columns and 1033680 observations spread across CA in 2018. The datetimes in this particular frame are for each hour at different weather 
#stations throughout CA. Each row (observation) is a measurement taken at a specific station for a given hour during the year.


# %%
df_noaa = pd.read_csv("NOAA_Data_MultiPointModel.csv")
df_noaa.head(10)
df_noaa['wind_speed_set_1'].plot.hist(bins=50,range=(0,20))
df_noaa['air_temp_set_1'].plot.hist(bins=50,range=(0,20))

# %% [markdown]
#  ### Trucking

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

# %% [markdown]
#  ## Forecasting and Prediction Modeling (25 points)
# 
#  This section is where the rubber meets the road.  In it you must:
#  1. Explore at least 3 prediction modeling approaches, ranging from the simple (e.g. linear regression, KNN) to the complex (e.g. SVM, random forests, Lasso).
#  2. Motivate all your modeling decisions.  This includes parameter choices (e.g., how many folds in k-fold cross validation, what time window you use for averaging your data) as well as model form (e.g., If you use regression trees, why?  If you include nonlinear features in a regression model, why?).
#  1. Carefully describe your cross validation and model selection process.  You should partition your data into training, testing and *validation* sets.
#  3. Evaluate your models' performance in terms of testing and validation error.  Do you see evidence of bias?  Where do you see evidence of variance?
#  4. Very carefully document your workflow.  We will be reading a lot of projects, so we need you to explain each basic step in your analysis.
#  5. Seek opportunities to write functions allow you to avoid doing things over and over, and that make your code more succinct and readable.
# 
# %% [markdown]
#  ### Single-point prediction model
#  In summary, this model attempts to predict PM2.5 levels at the location of a specific EPA sensor (as described in the data cleaning section) in the Bay Area. It combines EPA data, Beacon data, NOAA data (across 5 data types), and PurpleAir PM2.5 data.
# 
#  In terms of modeling approaches, this single-point model draws on OLS, lasso, ridge, and the elastic net. We ran into two principal issues in our data, which motivated our investigation of sub-questions:
#  * **Data quality:** Many of our predictors had incomplete data for the year (ie, less than 8760 hours' worth of data). We explored various "cutoffs" designating the maximum number of missing data points that any given column could have in order to remain in the dataset. After setting the cutoff, we removed any observations that had any NaNs remaining--thus, increasing the cutoff led to fewer observations being used for the model.
#  * **Collinearity:** Because collinearity was a big issue in our data for both PurpleAir and NOAA (ie, nearby sensors had highly correlated readings), both lasso and elastic net often failed to converge during gradient descent. As such, we leaned on ridge and OLS predominantly. However, we also tried to directly address collinearity per the suggestion of ISLR, by building a function to purge a dataframe of variables exceeding a certain correlation threshold.
# 

# %%
#Loading dependencies for models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


# %%
#Loading four dataframes
df_beac = pd.read_csv(filehome+"/Beacon_Data_SinglePointModel.csv")
df_epa = pd.read_csv(filehome+"/EPA_Data_SinglePointModel.csv")
df_noaa = pd.read_csv(filehome+"/NOAA_Data_SinglePointModel.csv")
df_pa = pd.read_csv(filehome+"/PA_Data_SinglePointModel.csv")

# %% [markdown]
# In the course of building the model, we noticed that the lasso model wasn't converging. When features are highly correlated, as we learned in class, lasso can become unstable. As such, we decided to A) investigate what correlation was present in the NOAA and PurpleAir datasets (which are contributing the bulk of the predictors and are very likely correlated), and B) see if we could address it by removing some of the semi-redundant variables. (This is one of the methods suggested by ISLR, although they don't give much detail on how to do it--so we did a few sensitivities.) After doing all this work, we determined that the converge issue could be solved by increasing the number of iterations allowed for gradient descent in our main functions, but we still find the results interesting!
# 
# We wrote a function to identify which variables in a dataframe exceed a certain correlation limit and to drop the redundant rows (e.g., if one variable is 98% correlated with another, it's not adding much value to the model).

# %%
def drop_correlated_cols(df, limit=.95):
    """
    Takes a dataframe with many correlated features, and drops all but one that have a correlation above a certain cutoff
    Returns the dataframe subsetted to only include features not correlated above a certain cutoff with anything else
    """

    cor = df.corr().abs()

    to_drop = []
    dont_drop = []

    for column in cor.columns:
        correlated_columns = cor.index[cor.loc[:,column]>limit].tolist()
        x = correlated_columns[1:] #want to get rid of all but one
        if len(correlated_columns) > 0:
            if correlated_columns[0] not in to_drop:
                dont_drop.append(correlated_columns[0]) #making sure we don't put this in the to-drop list later
        for i in range(len(x)):
            to_drop.append(x[i])
    

    all_cols = set(cor.columns)
    drop_cols = set(to_drop) - set(dont_drop)
    keep_cols = all_cols - drop_cols
    keep_cols.add('datetime')

    return df.loc[:,keep_cols]

# %% [markdown]
#  We used this function to "purge" the NOAA and PurpleAir datasets of a certain amount of correlated data. See the visual representation of the correlation matrices for each below. Many of the nearly-1 columns are gone, decreasing the amount of bright yellow you see, and the overall number of features is reduced.

# %%
lim = .85  #max correlation coefficient that will be allowed before throwing out a feature

#Creating 'purged' NOAA dataframe
ncor1 = df_noaa.corr()
df_noaa_new = drop_correlated_cols(df_noaa, limit=lim)
ncor2 = df_noaa_new.corr()

#Creating 'purged' PurpleAir dataframe
pcor1 = df_pa.corr()
df_pa_new = drop_correlated_cols(df_pa, limit=lim)
pcor2 = df_pa_new.corr()


# %%
plt.figure(figsize=(7,4))

plt.subplot(1,2,1)
plt.imshow(ncor1)
plt.title("NOAA data, all predictors")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(ncor2)
plt.title("NOAA data, predictors with \n  >"+str(lim)+" corr. coeff. dropped")
plt.colorbar()

plt.tight_layout()
plt.savefig("NOAA--75.png")


# %%
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(pcor1)
plt.title("PurpleAir data, all predictors")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(pcor2)
plt.title("PurpleAir data, predictors with >"+str(lim)+" \n correlation coeff. dropped")
plt.colorbar()
plt.savefig("PA--85.png")

# %% [markdown]
#  Next we decide how many missing values we are willing to tolerate, and designate a "cutoff" we use to drop columns with too many NaNs. We then remove each row with any NaNs in it, so that we can analyze a clean, complete dataframe. We keep track of the cutoff as well as the resulting number of observations and predictors we're working with, in case it affords any insight into our model's functioning...
# 
#  We then formatted our dataframe into X's and y's to be able to feed into sklearn models. Importantly, we standardized our data, because the scale of the data makes a dfiference when using ridge.
# 
#  We packaged this into two functions to be able to more easily manipulate the cutoff and produce X and y.

# %%



# %%
def set_cutoff(bigdf, cutoff):
    """
    Takes cutoff for maximum missing data points to subset dataframe to get ready
    to turn into X/y dataframes
    """
    #How many NaNs are there in each column?
    nullcols = bigdf.isnull().sum(axis=0)
    
    #Subsetting data to just have predictors with <X missing observations for a given year
    c = cutoff
    best_sites = list(nullcols.index[nullcols<c])
    bigdf_subset = bigdf[best_sites].reset_index(drop=True)

    #Subsetting data to just have observations with no NaNs
    nullrows = bigdf_subset.isnull().sum(axis=1)
    bigdf_subset = bigdf_subset[nullrows == 0].reset_index(drop=True)
    
    return bigdf_subset


# %%
def process_XY(bigdf_subset):
    """
    Takes dataframe containing amalgamated data, subsetted to get rid
    of NaNs, and outputs training & testing X and y
    """
    #Creating X and y
    y = bigdf_subset['epa_meas']
    X_raw = bigdf_subset.drop(columns=['epa_meas','datetime','latitude','longitude','name'])

    #Standardizing data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)

    #Getting training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)
    return X_train, X_test, y_train, y_test

# %% [markdown]
#  Finally, we defined the two functions that we used to generate and tune our candidate models:
#  * **fit_model**, which fits a model (either OLS, ridge, lasso, or elastic net) to the training data and returns the MSE and a list of the coefficients of the model.
#  * **model_cv_mse**, which conducts a k-fold cross validation using the training dataset. This optimizes the hyperparamter alpha (or lambda) in the ridge/lasso/elastic net models, but doesn't make sense for OLS (which doesn't have a hyperparamter like that). We defaulted to using <font color='red'>k=5  on the recommendation of ISLR.</font>
# 
#  <font color='red'>To help encourage convergence, we increased the tolerance</font>

# %%
def fit_model(Model, X_train, X_test, y_train, y_test, alpha = 1):
    """
    This function fits a model of type Model to the data in the training set of X and y, and finds the MSE on the test set
    Inputs: 
        Model (sklearn model): the type of sklearn model with which to fit the data - LinearRegression, Ridge, or Lasso
        alpha: the penalty parameter, to be used with Ridge and Lasso models only
    Returns:
        mse: a scalar containing the mean squared error of the test data
        coeff: a list of model coefficients
    """

    if (Model == Ridge) | (Model == Lasso) | (Model == ElasticNet):    # initialize model
        model = Model(max_iter = 10**9, alpha = alpha, tol=.1)
    elif Model == LinearRegression:
        model = Model() 
    
    model.fit(X_train, y_train)    # fit model
    
    y_pred = model.predict(X_test)  # get mean squared error of test data
    mse = mean_squared_error(y_pred,y_test)
    
    coef = model.coef_     # get coefficients of model
    
    return mse, coef


# %%
from sklearn.model_selection import KFold
def model_cv_mse(Model, X, y, alphas, k = 3):
    """
    This function calculates the MSE resulting from k-fold CV performed on a training subset of 
    X and y for different values of alpha.
    Inputs: 
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
            if Model == LinearRegression:
                model = Model()
            else:
                model = Model(max_iter = 10**8, tol=.1, alpha=alphas[i]) # initialize model

            model.fit(X_f_train,y_f_train) # fit model
            
            y_pred = model.predict(X_f_val) # get predictions
            
            mse = mean_squared_error(y_f_val,y_pred)
            mses[fold,i] = mse # save MSE for this fold and alpha value
        
        fold += 1
    
    average_mses = np.mean(mses, axis=0) # get average MSE for each alpha value across folds
    
    return average_mses

# %% [markdown]
#  Finally, we write a function that puts it all together: it takes in the original dataframe, the data missingness cutoff parameter, and outputs a list of MSEs. We use the model_cv_mse function to fit the lasso/ridge/elasticnet models and tune their hyperparameters via cross-validation (using the validation data partitioned out of the training set), and then call fit_model for all four models, including OLS (using the test data partitioned out of the full dataset at the beginning).

# %%
def master_model(bigdf, cut):
    """ etc """
    
    #Setting data cutoff
    bigdf_subset = set_cutoff(bigdf, cutoff=cut)
    X_train, X_test, y_train, y_test = process_XY(bigdf_subset)
    
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 10**5, 10**6, 10**7]
    models = [Ridge, Lasso, ElasticNet, LinearRegression]
    
    #Optimizing alpha for each model with a hyperparameter
    optimal_alpha = {}
    for model in models:
        mses = model_cv_mse(model, X_train, y_train, alphas)
        optimal_alpha[model] = alphas[np.argmin(mses)]
    
    #Fitting models with tuned hyperparameters
    mses_all = []
    coefs_all = []
    for mod in models:
        mse, coef = fit_model(mod, X_train, X_test, y_train, y_test, optimal_alpha[mod])
        mses_all.append(mse)
        coefs_all.append(coef)
        
    #Output
    out = [cut] + [bigdf_subset.shape[0]] + [bigdf_subset.shape[1]] + mses_all
    return out

# %% [markdown]
# Finally, we merged all of our data (waiting until this late in the notebook to do so to make it easier to change the correlation-purge inputs) and ran the models across a series of data quality cutoffs, making the limit for missing values for a predictor range from 175 to 975.

# %%
#Merging dataframes
bigdf = df_epa.merge(df_beac, how='left', on='datetime').reset_index(drop=True)
bigdf.drop(columns=['Laney College'], inplace=True) #laney college is messed up
bigdf = bigdf.merge(df_noaa, on='datetime', how='left')
bigdf = bigdf.merge(df_pa, on='datetime', how='left')


# %%
#Exploring a range of different data quality cutoffs for the dataset with no dropped correlations
cuts = np.arange(175,975,15)


# %%
df_original = pd.DataFrame(['Cutoff', '# Observations', '# Predictors', 'Ridge MSE', 'Lasso MSE', 'ElasticNet MSE', 'LinearRegression MSE'])
for c in cuts:
    x = pd.Series(master_model(bigdf, cut=c))
    df_original = pd.concat([df_original,x], axis=1)

# %% [markdown]
#  We then reran the above code three more times--once each for 95%, 85%, and 75% purges for the correlation matrices.

# %%
lim=.95
df_noaa_new = drop_correlated_cols(df_noaa, limit=lim)
df_pa_new = drop_correlated_cols(df_pa, limit=lim)

#Merging dataframes
bigdf = df_epa.merge(df_beac, how='left', on='datetime').reset_index(drop=True)
bigdf.drop(columns=['Laney College'], inplace=True) #laney college is messed up
bigdf = bigdf.merge(df_noaa_new, on='datetime', how='left')
bigdf = bigdf.merge(df_pa_new, on='datetime', how='left')

df_95 = pd.DataFrame(['Cutoff', '# Observations', '# Predictors', 'Ridge MSE', 'Lasso MSE', 'ElasticNet MSE', 'LinearRegression MSE'])
for c in cuts:
    x = pd.Series(master_model(bigdf, cut=c))
    df_95 = pd.concat([df_95,x], axis=1)


# %%
lim=.85
df_noaa_new = drop_correlated_cols(df_noaa, limit=lim)
df_pa_new = drop_correlated_cols(df_pa, limit=lim)

#Merging dataframes
bigdf = df_epa.merge(df_beac, how='left', on='datetime').reset_index(drop=True)
bigdf.drop(columns=['Laney College'], inplace=True) #laney college is messed up
bigdf = bigdf.merge(df_noaa_new, on='datetime', how='left')
bigdf = bigdf.merge(df_pa_new, on='datetime', how='left')

df_85 = pd.DataFrame(['Cutoff', '# Observations', '# Predictors', 'Ridge MSE', 'Lasso MSE', 'ElasticNet MSE', 'LinearRegression MSE'])
for c in cuts:
    x = pd.Series(master_model(bigdf, cut=c))
    df_85 = pd.concat([df_85,x], axis=1)


# %%
lim=.75
df_noaa_new = drop_correlated_cols(df_noaa, limit=lim)
df_pa_new = drop_correlated_cols(df_pa, limit=lim)

#Merging dataframes
bigdf = df_epa.merge(df_beac, how='left', on='datetime').reset_index(drop=True)
bigdf.drop(columns=['Laney College'], inplace=True) #laney college is messed up
bigdf = bigdf.merge(df_noaa_new, on='datetime', how='left')
bigdf = bigdf.merge(df_pa_new, on='datetime', how='left')

df_75 = pd.DataFrame(['Cutoff', '# Observations', '# Predictors', 'Ridge MSE', 'Lasso MSE', 'ElasticNet MSE', 'LinearRegression MSE'])
for c in cuts:
    x = pd.Series(master_model(bigdf, cut=c))
    df_75 = pd.concat([df_75,x], axis=1)


# %%
#A little reformatting
for df in [df_original, df_75, df_85, df_95]:
    cols = df.columns.tolist()
    cols[0] = "Ind"
    df.columns = cols
    df.set_index("Ind")


# %%
"""df_original_50 = df_original
df_95_50 = df_95
df_85_50 = df_85
df_75_50 = df_75"""


# %%
df_75


# %%
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('viridis', 4)
colors = [viridis.colors[3], viridis.colors[2], viridis.colors[1], viridis.colors[0],"#34495e"]

plt.figure(figsize=(15,5))
for i, df in enumerate([df_original, df_75, df_85, df_95]):
    plt.plot(df.iloc[0,1:], df.iloc[3,1:], linestyle="-", color=colors[i])
    plt.plot(df.iloc[0,1:], df.iloc[4,1:], linestyle="-.", color=colors[i])
    plt.plot(df.iloc[0,1:], df.iloc[5,1:], linestyle="--", color=colors[i])
    plt.plot(df.iloc[0,1:], df.iloc[6,1:], linestyle=":", color=colors[i])
    
#Making legend
plt.plot(df.iloc[0,1], df.iloc[3,1], linestyle="-", color='black', label="Ridge")
plt.plot(df.iloc[0,1], df.iloc[4,1], linestyle="-.", color='black', label="Lasso")
plt.plot(df.iloc[0,1], df.iloc[5,1], linestyle="--", color='black', label="ElasticNet")
plt.plot(df.iloc[0,1], df.iloc[6,1], linestyle=":", color='black', label="OLS")
plt.plot(df.iloc[0,1], df.iloc[3,1], linestyle="-", color=colors[0], label="No corr. adjustment")
plt.plot(df.iloc[0,1], df.iloc[4,1], linestyle="-", color=colors[1], label="Corr. limit .75")
plt.plot(df.iloc[0,1], df.iloc[5,1], linestyle="-", color=colors[2], label="Corr. limit .85")
plt.plot(df.iloc[0,1], df.iloc[6,1], linestyle="-", color=colors[3], label="Corr. limit .95")
    
plt.ylim(0,20)
plt.xlabel("Cutoff for max number of missing data points per predictor")
plt.ylabel("Test MSE")
plt.title("Comparing MSEs across data-quality cutoffs and de-correlation adjustments across four models")
plt.legend(loc='lower left', ncol=2)
plt.savefig("SinglePointModel.png")


# %%
df_95

# %% [markdown]
#  <font color='red'>should just go back in hw and make sure this is right process.......</font>
# %% [markdown]
#  ## Interpretation and Conclusions (20 points)
#  In this section you must relate your modeling and forecasting results to your original research question.  You must
#  1. What do the answers mean? What advice would you give a decision maker on the basis of your results?  How might they allocate their resources differently with the results of your model?  Why should the reader care about your results?
#  2. Discuss caveats and / or reasons your results might be flawed.  No model is perfect, and understanding a model's imperfections is extremely important for the purpose of knowing how to interpret your results.  Often, we know the model output is wrong but we can assign a direction for its bias.  This helps to understand whether or not your answers are conservative.
# 
#  Shoot for 500-1000 words for this section.

# %%


