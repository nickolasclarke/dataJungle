# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  # Predicting Regulatory Grade Air Quality Sensor Output
#  ## ER131 Final Project
# 
#    - Nick Clarke: Research data sources and various methods to get the data. Data aquistiion and extensive cleaning of NOAA and Purple Air Data. Trained others on how to use Git for collaboration effectively. Significant input on selection and design of our various models. Help debugging others code. Helped analyze and interpret model results. Write and edit significant portions of final notebook.
#    - Margaret McCall: Collected Beacon, EPA, and trucking data; wrote scripts to clean & process these 3 datasets to fit format needed by each models. Wrote draft code for singlepoint and multipoint models; led coding/writeup/poster outputs of singlepoint model.
#    - Jake McDermott: Initial data acquisition and subsetting of PurpleAir data. Formulated and ran cases for Questions 2 and 3. Contributed heavily to poster design and interpretations/analysis portion of multi-point model.
# 
# %% [markdown]
#  ## Abstract (5 points)
#  This project attempts to forecast PM 2.5 readings from EPA sensors for a variety of resource allocation questions. The explosion of Machine Learning and Artificial Intelligence has created new use cases for quantitative analysis to solve pressing social problems. For the purposes of this project, we are concerned with the flurry of wildfires occuring in CA over the last few years. With increased wildfire risk comes increased concern from public health officials about how best to respond by way of public tax dollars. Our project creates a few hypothetical questions to answer prior to data collection and cleaning, modeling, and drawing conclusions. The results aren't promising. Regardless of model choice (OLS, Ridge, Lasso), we find that there exists a large amount of cross validated error. This makes it difficult for us to recommend that public sector managers, decision makers, and everyday citizens rely on the modeling efforts contained within. Still, there is a glimmer of hope - for if we can get better, cleaner, input data (and more of it), we might find that the efficacy of our modeling improves.
# %% [markdown]
#  ## Project Background (5 points)
#  Air quality is a serious environmental issue that has major implications for public health. Over[ 90% of the world lives](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health) in areas that do not meet WHO guidelines for safe air quality, and leads to [millions of premature deaths every year](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health). It is often a highly localized problem, both in space and time. Until recently it has been rather difficult to get continous, high temporal and spatial resolution data for monitoring air quality. Previously, monitoring was done either with spot measurements periodically, or continously with research-grade monitors run by governments and research groups. While these monitors provide good, continous data, they cost tens of thousands of dollars. As such, these monitoring networks have poor spatial coverage.
# 
#  With the advent of consumer-grade, low cost air quality monitors however, it is possible to get continous monitoring coverage for significantly cheaper. While these monitors do not have the same accuracy as a research grade monitor, in large enough numbers, [research suggests](https://www.epa.gov/sites/production/files/2018-03/documents/final_em-3_master_slide_set.pdf) that they can provide valuable insights into air quality trends in a given area. Moreover, air quality is greatly impacted by weather, topography, and local point sources of emissions, etc. By combining these additional timeseries and static datasets with low-cost monitoring data, it may be possible to make reasonable predictions of air quality across space and time in places where a research grade monitor is not present. If so, this would be a boon for researchers and policy makers alike, as they look to improve their understanding of of the impacts of air pollution and dedicate resources to monitoring, curtailing and abating air pollution.
# 
#  Convienently, the Purple Air low cost air quality sensor has seen significant deployment across the world in recent years. Over 2,000 low-cost sensors have been deployed across California alone, and the historical data for the vast majority of these sensors is publically available. In addition, high quality weather and truck freight traffic data is also available.
# # Project Objective (5 points)
#  With this background in mind, we set out to answer the following questions:
#  1. Should California replace faulty or broken regulatory-grade sensors? This question requires that we predict PM2.5 levels at specific locations of existing regulatory sensors;
#  2. When should Californians plan to purchase N95 masks? This query requires that we create forecasts for any given point in California that maintain accuracy throughout the year to help guide spending decisions; and
#  3. Should the government subsidize the purchase of PurpleAir or other low quality air monitors? Our final question builds off our prior forecasts for question 2, but only uses PurpleAir data as a feature

# %%
import glob
import os
import datetime
import zipfile
import json
import csv
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely

import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import scale
from pandas.io.json import json_normalize


pd.set_option('display.max_columns', 500)


# %%
#Setting directory where csv files live, to make this document easier to pass between us
filehome = '/Users/margaretmccall/Documents/2019 Fall/ER 131- Data Enviro Soc/Data/'

# %% [markdown]
#   ## Input Data Description (5 points)
#   Here you will provide an initial description of your data sets, including:
#   1. The origins of your data.  Where did you get the data?  How were the data collected from the original sources?
#   2. The structure, granularity, scope, temporality and faithfulness (SGSTF) of your data.  To discuss these attributes you should load the data into one or more data frames (so you'll start building code cells for the first time).  At a minimum, use some basic methods (`.head`, `.loc`, and so on) to provide support for the descriptions you provide for SGSTF.
# 
#   [Chapter 5](https://www.textbook.ds100.org/ch/05/eda_intro.html) of the DS100 textbook might be helpful for you in this section.
# %% [markdown]
#    For our project we used the following data sets:
# 
#    1. Purple Air Data
#    2. Berkeley Enviro Air-Quality and CO2 Observation Network (BEACON)
#    3. EPA
#    4. NOAA Weather Data
#    5. Trucking Data from the Freight Analysis Framework
# %% [markdown]
#  ### Purple Air (PA):
#  #### Origins:
# We obtained historic PA data through two workflows. First, we used a python script that pulls metadata for all PA sensors. This includes:
#   - Channel IDs
#   - Thingspeak API read-only keys
#   - Lat/lon
#   - Device location (outside/inside)
# 
# From this, we  filtered for outside devices read the resulting CSV into R where we performed a spatial subset to isolate only sensors in California. With a filtered list of devices in hand, we used another python script to query the ThingSpeak API which hosts all of the historic purple air data. The ThingSpeak API is organized by `channel`s, and each Purple Air device uses two channels (A and B). This is because each Purple Air device has two discrete laser sensors that are used to validate each other. If they do not detect issues in the data, PA later combines the data from both these channels to give a single value on their website and apps. This final aggregated value is not available via their API, and it is unclear what the full logic is behind their aggreation. Purple Air takes a value for each  `channel` every 80 seconds and pushes them to their respective ThingSpeak `channel_id` via the ThingSpeak API. We pull data for every channel in CA.
# 
#  Below we read in our data:

# %%
df_pa = pd.read_csv(f'{filehome}pa_full.csv')
df_pa.rename(columns={"lat":"latitude","lon":"longitude"})
df_pa.head(10)

df_pa['PM2.5 (ATM)'].plot.hist(bins=50,range=(0,100))

# %% [markdown]
#  #### SGSTF
#  - **Structure:**
#    - This is an 8 column X 4424491 row CSV file, which we read in as a dataframe.
#  - **Granularity**
#    - We have 8 columns which represent the following:
#      -`datetime`: ISO8601-compatible datetime string in UTC, normalized to the hour.
#      -`PM1.0 (ATM)`: PM 1.0 ug/m^3 @ atmospheric pressure
#      -`PM2.5 (ATM)`: PM 2.5 ug/m^3 @ atmospheric pressure
#      -`PM10.0 (ATM)`: PM 10 ug/m^3 @ atmospheric pressure
#      -`PM2.5 (CF) `: PM 1.0 ug/m^3 with a correction factor applied
#      -`lat`: latitude
#      -`lon`: longitude
#      -`id`: unique ID of the Thingspeak Channel
#  Using `datetime` as an index, each row represents a set of measurements from a discrete PA sensor `channel` at a given hour throughout the year. This means there are ~8760 hourly readings for each sensor `channel`.
#  - **Scope**
#   - Because we subset the entire universe of PA sensors by just those in CA, we know that the scope is adequate for our purposes.
#  - **Temporality**
#    - As mentioned this is hourly resolution data, with all data normalized to the top of the hour @ UTC. It covers all 8760 hours of 2018, namely  2018-01-01 00:00 to 2018-12-31 23:00.
#  - **Faithfulness**
#    - we do believe it captures reality, generally. PA provides several flags for incorrect or suspect data, which we used to subset the data during data aquisition. However, not all channels had data for the full 8760 hours. Where these gaps exist, there are null values on these rows.
# 
# Our histogram plotted above is meant to illustrate the range of values for one of our features of interest. In this case, an expected long tailed distribution is created. Again, based on the results of the visualization, it appears as though the data has not been tampered with and represents reality.
# %% [markdown]
#   ### BEACON:
#  #### Origins
# We obtained the data directly from [ Beacon](http://www.beacon.berkeley.edu/). Beacon is a small Berkeley-run network of lower-quality air pollution sensors that seems to have some of the same benefits as PurpleAir, in that they make a tradeoff between sensor quality and density. Most sensors are in the Bay Area, but some are in NYC and Houston.
#   Beacon has the feeling of being a much smaller operation than any of the other data sources we used--in addition to having a small sensor network in the first place, we had to download data (in CSV form) from each of the ~80 sensors by hand--there was no API or preformatted data download.
# 
#   #### SGSTF
# The Beacon data have some drawbacks, which we discovered when we first started looking through the data.
# 
#  - **Structure:**
#  The data were downloadable as CSV files that had air pollution readings in a certain timeframe for an individual sensor. Each row is an hourly average value of air pollution. The air pollution readings given are for particulate matter (PM) and CO2. To combine the data into the raw structure shown below, we read in each individual CSV file (75 total), omitting those that threw an error for being empty (21 in total). We concatenated the remaining 56 into a dataframe, which we then merged with a metadata df containing lat/lon information and the names of different sites. This merged df, with irrelevant columns dropped, is shown below. <p> The columns represent the following data: <p>
#       - `Year`, `Month`, `Day`, `HourStarting`: columns representing date and time in UTC, pulled from original datetime column. Format: integer
#       - `node_id`: a unique ID for each sensor. Format: integer
#       -`pm_pct_fs`: the raw measurement from the sensor instrument--this number is proportional to the percentage of time scattering is detected by the unit. Format: float
#       - `PM_QC_level`: the quality control level of the PM record (0 = raw data; 1 = raw data*18.9 as rough conversion to ug/m3; 1a = level 1 with an offset; 2 = 1a but inspected for quality. -999 and -111 seem to reflect raw data as well as we explored, but are not explicitly stated to do so). Format: integer OR string (fixed!)
#       - `PM_ug/m3`: if the raw data has been converted to units of micrograms per cubic meter (standard air pollution units). `-999` value implies no conversion. Format: integer OR string (fixed!)
#       - `node_name_long`: name of the sensor. Matches with `node_id`. Format: string
#       - `lat`: latitude. Format: float
#       - `lng`: longitude. Format: float
#       - `height_above_ground`: how far sensor is above ground (units unknown). Not ultimately used in analysis. Format: float
#       - `height_above_sea`: how far sensor is above sea (units unknown). Not ultimately used in analysis. Format: float
# 
#   - **Granularity:**
#  Each record in our dataset represents a particular hour at which an hourly air pollution average is given at a particular sensor. The original data were aggregated into the outputs shown in this table according to this process: "Averages have been calculated by taking measurements for the whole hour, then assigning them to beginning of the hour. So 12 AM will include measurements from `12:00:00-12:59:59`"
#   - **Scope:**
#  The scope of the data is where the trouble starts. Geographically, they only cover the Bay Area, whereas we're interested in the whole of California; however, we knew that would be a limitation going in. Temporally, however, we discover that huge chunks of data are missing for most of the sensors. There should be 8760 hourly observations (at least for raw data) in a complete year, but only 8 of the 54 sensors had more than 7000 observations. (We chose this level of data completeness pretty arbitarily, so that we could at least keep some sensors but not lose too much data.) See below for a rough depiction of the air pollution levels recorded over time at each sensor--this was enough to encourage us to ditch most of them.
#   - **Temporality:**
#  As mentioned, we were working with UTC data to begin with, which represent air pollution measurements from a given hour. We subsetted the data to include 2018 only (since the data pulled had some stragglers from surrounding years). As shown below, no values are out of the ordinary.
#   - **Faithfulness:**
#  The notable faithfulness issue is that even though raw data (`PM_QC_level = 0, or -999 or -111`) should be easily converted to `PM_ug/m3` (and a `PM_QC_level == 1`) by multiplying by 18.9, this wasn't always done; there are fewer missing raw data points than there are points that have failed to be converted to PM_ug/m3. As such, we will use the raw data in our model. We also spot-checked level-1 `PM_ug/m3` readings against `pm_pct_fs` readings to see if they were roughly 18.9x bigger, as expected.

# %%
#Structure
df_beac_raw = pd.read_csv(filehome+"Beacon_Data_RAW.csv")
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
#   ### EPA:
# 
#   #### Origins
#   We collected data from the EPA via a REST API. The API is used to collected data from the EPA's Air Quality System (AQS) database. From the EPA's website: "AQS contains ambient air sample data collected by state, local, tribal, and federal air pollution control agencies from thousands of monitors around the nation. It also contains meteorological data, descriptive information about each monitoring station (including its geographic location and its operator), and information about the quality of the samples."
#  #### SGSTF
#   - **Structure**:
#  The data was originally obtained as a single JSON file for each month. In order to work with the data more easily, we used a function called `json_normalize` to convert to a dataframe as we read in each file, which we then appended together. After this simple conversion, the records were neatly organized in rows. **The fields of relevance are described further in the data cleaning section, but are:
#    -`date_gmt`: ISO8601-compatible date string, in UTC
#    -`time_gmt`: ISO8601-compatible time string, in UTC
#    -`sample_measurement`: the measurment of PM2.5
#    -`sample_frequency`: the resolution of the sample
#    -`latitude`: latiude of the sensor
#    -`longitude`: longittude of the sensor
# 
#   - **Granularity**:
#  Each record represents a time period at a specific sensor site at which an air pollution sample was taken. However, the raw data contains two columns--`sample_duration` and `sample_frequency`--that differ across different observations. We are interested in samples taken every hour, and it seems that if we select observations with `sample_duration=='1 HOUR'`, that corresponds almost exclusively to data points with `sample_frequency=='HOURLY'` except for one point that appears to be mislabeled `'SEASONAL'`.
#   - **Scope**:
#  When cleaning our data, we subset for CA only sensor locations and their respective data.
#   - **Temporality**:
#  As described above, our final data is hourly resolution for all hours of 2018. As the data fields suggest, the dates and time are in UTC.
#   - **Faithfulness**:
# Compared to other datasource, such as PA or BEACON this data is considerablly cleaner. While there are some missing data, it has good temporal coverage. However, there are some missing hours at some sensor locations, well within the 7000 hours threshold we used for other data sources.

# %%
#Structure
df_epa_raw = pd.read_csv(filehome+"EPA_Data_Complete.csv")
df_epa_raw.head(3)


# %%
#Granularity
print(df_epa_raw['sample_frequency'].unique())
print(df_epa_raw['sample_duration'].unique())
print(df_epa_raw['sample_frequency'][df_epa_raw['sample_duration']=='1 HOUR'].unique())

# %% [markdown]
#   Our dataframe consists of 5 columns and 67801 rows pulled from a CSV file. Because the data was extracted via a REST API, we were able to pull only 2018 data from California. The data set includes the latitude and longitude from each of the 119 distinct EPA sensors across CA. Each observation (row) represents a reading from a sensor at a specific hour throughout 2018. The PM2.5 reading is given in the column `epa_meas`. That column is what we are ultimately going to be trying to forecast. Our histogram of the EPA data shows another right tailed distribution. Interestingly, there appears to be one bin in particular that is much more frequent than the rest.
# %% [markdown]
#  ### NOAA
#  #### Origins
#   Our NOAA data was collected from the Synoptic MESONET Timeseries API. We collect one year of hourly resolution data for all MESONET stations that were located in CA, and append each csv together into a complete dataframe, similar to the PA data.

# %%
df_noaa = pd.read_csv(filehome+"noaa_full.csv")
df_noaa.head(10)
#### SGSTF
#  - **Structure**:
# The final form of the NOAA data consists of a dataframe similar to our PA data. each row represents 1 hour of the year for a given station, and has 32 features. This results in a dataframe that is ~2 million rows, covering ~200 sensors.
#
#  - **Granularity**:
#
# -`Date_Time`: ISO8601-compatible datetime string, UTC
# -`altimeter_set_1`
# -`air_temp_set_1`
# -`dew_point_temperature_set_1`
# -`relative_humidity_set_1`
# -`wind_speed_set_1`
# -`wind_direction_set_1`
# -`wind_gust_set_1`
# -`sea_level_pressure_set_1`
# -`weather_cond_code_set_1`
# -`cloud_layer_3_code_set_1`
# -`pressure_tendency_set_1`
# -`precip_accum_one_hour_set_1`
# -`precip_accum_three_hour_set_1`
# -`cloud_layer_1_code_set_1`
# -`cloud_layer_2_code_set_1`
# -`precip_accum_six_hour_set_1`
# -`precip_accum_24_hour_set_1`
# -`visibility_set_1`
# -`metar_remark_set_1`
# -`air_temp_high_6_hour_set_1`
# -`air_temp_low_6_hour_set_1`
# -`peak_wind_speed_set_1`
# -`ceiling_set_1`
# -`pressure_change_code_set_1`
# -`air_temp_high_24_hour_set_1`
# -`air_temp_low_24_hour_set_1`
# -`peak_wind_direction_set_1`
# -`dew_point_temperature_set_1d`
# -`wind_chill_set_1d`
# -`pressure_set_1d`
# -`sea_level_pressure_set_1d`
# -`heat_index_set_1d`
# -`Station_ID`
#
#  - **Scope**:
# The data covers both the temporal (all hours of 2018) and spatial scope well (all MESONET sensors in CA)
#  - **Temporality**:
# The Raw data from the MESONET timeseries API are a mix of 5 minute to 1 hour resolution datasets. After resampling the data, we had extremely thorough hourly data for all points. 
#  - **Faithfulness**:
# The MESONET API allows you to filter thoroughly for quality issues and remove suspect data beforehand, and accurrately labeling `null` data as such. This was helpful. Most of these sensors appear to be at airports, and have quite good data coverage.

# %% [markdown]
# ### Trucking data
# #### Origins:
# We obtained truck data from the Freight Analysis Framework (FAF) model from the Bureau of Transportation Statistics. FAF data is suitable for geospatial analysis, and contains historical and predicted figures relevant to freight movement (e.g., annual number of freight trucks, non-freight trucks, tons of goods by truck, etc) for each highway segment. We ultimately only used this trucking data in our multi-point model--since it doesn't vary in time (the values presented are annual averages), we realized that it has no explanatory power when included in the single-point model.
# 
# #### SGSTF:
# - **Structure**: 
# The data was initially read in as a shapefile, and then a separate DBF dataframe containing attributes associated with each linestring in the shapefile was read in and merged. The DBF file was tabular already (I think) but was transformed into a dataframe to facilitate the merge. The relevant columns are addressed in the data cleaning section.
# - **Granularity**: 
# Each row of the truck data represents data for a specific linestring--that is, a geometric representation of a certain highway segment. While all rows are at this level of granularity, the length of each segment of highway may be different, so the features are not comparable on their face.
# - **Scope**: 
# The initial dataset is country-wide and extends into some parts of Canada as well.
# - **Temporality**: 
# This data is aggregated data (in fact, estimated population data based on sample data) representing truck traffic in 2012. Forward-looking estimates for 2045 were also available in this dataset
# - **Faithfulness**: 
# The main faithfulness check that was feasible to do was to determine whether the length of the highway segment given in the 'LENGTH' column corresponded to the difference between the BEGMP and ENDMP (start and end mile marker) columns, which it did.

# %%
#Structure
df_trucks_raw = pd.read_csv(filehome+"Truck_Data_RAW.csv")
df_trucks_raw.head(3)

# %% [markdown]
#    ## Data Cleaning (10 points)
#    In this section you will walk through the data cleaning and merging process.  Explain how you make decisions to clean and merge the data.  Explain how you convince yourself that the data don't contain problems that will limit your ability to produce a meaningful analysis from them.
# 
#   [Chapter 4](https://www.textbook.ds100.org/ch/04/cleaning_intro.html) of the DS100 textbook might be helpful to you in this section.
# %% [markdown]
#   Below we describe the separate data cleaning process for each of our five datasets. The PurpleAir and the Beacon datasets definitely required the most cleaning/processing compared to the three government-sourced datasets. Both Purple Air and Beacon had a lot of missing data.
# 
# The data merging process was straightforward and consistent across each of the following datasets: we merged the data in two different ways to accommodate two different models. For the single-point prediction model, we had a row for each unique hour in 2018, and merged each dataframe on that datetime column such that each row was a unique hour and each column was a unique sensor. For the multi-point prediction model, we merged the datasets on which EPA sensor they were associated with, obtaining a dataframe with a row for each EPA sensor and a column for the value of each other variable at that sensor's location.
# 
# 
#   **Formatting**:
# 
# For each dataset that was going to be fed into both the single-point and the multi-point model, we formatted it in two different ways. For the single-point model, we created a "pivoted" dataframe, with a column for each individual sensor and type of data (e.g., a column for each NOAA sensor with precipitation data), and a row for each hour. For the multi-point model, we created a "melted" dataframe, which was easier to pass into a KNN function to get the value of each variable at each EPA sensor point.
# 
# The format of each pivoted dataframe was to have the same first four columns to feed into the multi-point model functions: a column for datetime, latitude, longitude, and the sensor name (for readability). The subsequent columns should each contain a unique dataset. For the single-point model, each dataframe had a datetime column first, and then a column for each unique data type at each unique sensor.

# %%
sampl = pd.read_csv(filehome+"NOAA_Data_SinglePointModel.csv")
sampl.head()

# %% [markdown]
#    ### PurpleAir
#As mentioned in the first section, we obtained a list of all PurpleAir sensor metadata from the Purple Air REST API. Purple Air's API only provides meta data and the latest reading from a given sensor, while historical data is stored on the ThingSpeak IoT platform. We used the list of sensors, filtering in R for sensors located in California. From this subset list, we further subset for outdoor sensors, sensors and sensors flagged by Purple Air as having data quality issues. We then wrote a script to pull historical data for 2018 from the ThingSpeak REST API for each sensor. 
#
#The data aquistion and data cleaning process was fairly time intensive for Purple Air because of the nature of how Purple Air historical data is organized, and limited documentation of the ThingSpeak API. To quote from the section above: "The ThingSpeak API is organized by `channel`s, and each Purple Air device uses two channels (A and B). This is because each Purple Air device has two discrete laser sensors that are used to validate each other. If they do not detect issues in the data, PA later combines the data from both these channels to give a single value on their website and apps. This final aggregated value is not available via their API, and it is unclear what the full logic is behind their aggreation. Purple Air takes a value for each `channel` every 80 seconds and pushes them to their respective ThingSpeak `channel_id` via the ThingSpeak API. We pull data for every channel in CA."
# 
# When attempting to pull this historical data from ThingSpeak, we found that due to API design, it was very difficult to pull a full year of hourly data for a given channel. The API is limited to accessing 8000 datpoints per call, including any aggregation that must be done with those points. Nominally there is data every 80 seconds, all of which would need to be read by the API to return an aggregated value. This means that for every channel we had to split the year up into ~7.2 day chunks and make a seperate call for each of these chunks over the course of a year. This meant we had to make ~50 calls for each channel, stiching the results together. Since every sensor has 2 channels, this meant over 200,000 calls that had to be made to the ThingSpeak platform. Since python's standard http library `requests` does not support asyncronous requests, this had to be done in series, and was severely I/O bound as we waited for each call to be filled. This approach also lead to some duplicate and excess data that had to be scrubbed, since at the chunked calls were not always on the top of the hour.
# 
# Finally we were left with data for each channel for all sensors in California. However, it quickly became apparent that a large portion of these sensors had not been added to the Purpleair network until mid-2018 or later. There appeared to be a particular spike in new sensors between 2018 and today. This meant many of these data channels did not have sufficient data to be useful. We then filtered for sensors that had 6000 rows or more, dropping the rest. All this lead to a final count of ~200 sensors out of ~2000 that are publically listed as in CA by Purple Air.
# %% [markdown]
#    ### BEACON
# 
#    Much of the Beacon data-cleaning process was described in the first section: essentially, we
#    * Downloaded and concatenated all CSV files by hand, leaving us with 54 sensors with data in 2018;
#    * Merged dataframe with metadata to attach lat/lon data to each observation;
#    * Dropped data from sensors that had <7000 hourly observations for the year, leaving us with 8 sensors;
#    * Looked at a timeseries of the 8 sensors to see if anything looked amiss, and dropped another sensor (Laney College) that seemed to have a lot of observations with no variation (suggesting a faulty sensor)
# 
#    After finishing our initial data exploration, our main concern was in getting sensors with decent data quality, which we believe to have accomplished with this process.
# %% [markdown]
#    ### EPA
#    The principal cleaning that we did to the EPA data was:
#    * Subset it to contain observations where the sample_duration was one hour;
#    * Reformat the datetime column to be in a consistent format with other data frames;
#    * Drop all unnecessary columns, keeping only datetime, lat, lon, name, and the pollution measurement
# 
#    For the single-point model, the goal of which is to try to predict pollution at the location of a single EPA sensor, we chose a sensor in the Bay Area in order to be able to leverage the Beacon sensors in the model. To do this, we plotted the location of the Beacon sensors and EPA sensors, and visually designated a box from within which we selected an EPA sensor, using geopandas operations and then taking the first EPA sensor in the list. See graph below for some of the process.

# %%
beac_coords = pd.read_csv(filehome+"Beacon_Coordinates.csv")
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
#    ### NOAA
# 
#    To acquire NOAA data, we used the National Centers for Environmental Information's (NCEI's) REST API. However, this proved difficult to subset for the information we needed. After significant searching, we settled on using a combiniation of the NOAA weather API and the Synaptic MESONET API. The NOAA weather API allowed us to easily filter for stations only located in CA. However, similar to Purple Air, it only provides current conditions and some forecasting data. The MESONET timeseries API allowed us to access historical data from these stations. Using these together, we were able to pull data for all of the MESONET weather stations in CA. Most of these provided data at an hourly resolution, but not normalized to the top of the hour. Others, such as SFO or LAX airports, provided significantly higher data resolution at 2 - 5 minute resolution. 
#
#Once aquiring CSVs with a years worth of data for each of these stations, we resampled them to normalized hourly data at UTC. This resampling required us to drop all of the qualitative observation types such as `visbility == 'hazy'`. Finally we combined the data into two final dataframes, as described above, normalizing column names and values to match the other datasets.
# 
# 
# %% [markdown]
#    ### Trucking data
# 
#    The main data cleaning and manipulation we did on the truck data was as follows:
#    * We merged the geodataframe containing the highway linestrings with the data table containing information about truck traffic at those locations based on the column FAF4_ID (as instructed via the FAF website);
#    * We subsetted the data to be specific to California;
#    * We chose a few columns with data that seemed particularly relevant and maybe not too repetitive (all numbers are averages for 2012): AADT12 (average annual daily traffic on this segment), FAF12 (the number of FAF freight trucks on this segment daily), NONFAF12 (same, but non-freight trucks), YKTON12 (thousands of tons of freight daily), KTONMILE12 (thousands of ton-miles of freight daily).
#    * We then created three buffers--100m, 1000m, and 5000m--and summed the value for each relevant trucking variable within those buffers at each EPA sensor site.
# %% [markdown]
#    ## Data Summary and Exploratory Data Analysis (10 points)
# 
#    In this section you should provide a tour through some of the basic trends and patterns in your data.  This includes providing initial plots to summarize the data, such as box plots, histograms, trends over time, scatter plots relating one variable or another.
# 
#    [Chapter 6](https://www.textbook.ds100.org/ch/06/viz_intro.html) of the DS100 textbook might be helpful for providing ideas for visualizations that describe your data.
# %% [markdown]
#    ### PurpleAir

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
#    Our visualizations reveal a few notable insights. Looking at our comparison of PM2.5 to the PM2.5 with a correction factor,
#    it appears as though a constant correction factor is applied across the board (i.e., the slope of the line is constant).
#    This might mean that purple air sensors are all initially biased in the same direction. Our second visualization comparing PM2.5
#    to PM1.0 shows that for PM2.5 values above 2000, the relationship to PM1.0 is constant. Interestingly se see a lot of noise below
#    2000 on the x axis, meaning that there might be some threshold of PM2.5 that we'd need to see before we see a constant linear relationship
#    between the two variables. Our last comparison of PM2.5 to PM10.0 reveals a similar trend. Let's view these relationships in a different format:

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
#    The scatter plots above show us the distribution of our random sample of Purple Air PM readings over 2018. Let's look at the raw scatter plot of PM2.5 readings with no correction factor included:

# %%
plt.scatter(pa_sample["datetime"], pa_sample["PM2.5 (ATM)"], s = 5, color = 'c')
plt.title("2018 PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

# %% [markdown]
#    Based on our scatterplot, we see that most of our readings from this sample fall between 0 and 20 throughout the year. There are
#    values that exceed 20 here and there, but they seem to be on the higher side of the sample. If we plot another histogram of our sample
#    we can see this:

# %%
pa_sample['PM2.5 (ATM)'].plot.hist(bins=60,range=(0,70))

# %% [markdown]
#    ### Beacon
#    Below we plot a basic histogram showing the distribution of our Beacon data:

# %%
df_beac = pd.read_csv(filehome+"Beacon_Data_MultiPointModel.csv")
df_beac.head(10)
df_beac['beac_data'].plot.hist(bins=50,range=(0,10))

# %% [markdown]
#    ### EPA

# %%
df_epa = pd.read_csv(filehome+"EPA_Data_MultiPointModel.csv")
df_epa.head(10)
df_epa['epa_meas'].plot.hist(bins=100,range=(0,120))

# %% [markdown]
#    We included this histogram above, but we include it again to higlight the distribution of our EPA readings that we are interested in forecasting.
# 

# %%
epa_sample = df_epa.sample(n=60,random_state=1)

plt.scatter(epa_sample["datetime"], epa_sample["epa_meas"], s = 5, color = 'c')
plt.title("2018 EPA PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

# %% [markdown]
#    Because our data set is so large, we needed to take a random sample to actually plot the readings over the time of the year. It appears as though the readings are mostly consistent throughout the year with one outlier in the middle of the year, and a few higher
#    points scattered throughout.
# %% [markdown]
#    ### NOAA

# %%
df_noaa = pd.read_csv(filehome+"NOAA_Data_MultiPointModel.csv")
df_noaa.head(10)
df_noaa['wind_speed_set_1'].plot.hist(bins=50,range=(0,20))
df_noaa['air_temp_set_1'].plot.hist(bins=50,range=(0,20))
df_noaa['wind_direction_set_1'].plot.hist(bins=50,range=(0,600))

# %% [markdown]
#    Above we show basic distributions of three more features from our NOAA data set, the air temp, wind direction, and wind speed.
# %% [markdown]
#    ### Trucking

# %%
df_truck = pd.read_csv(filehome+"Truck_Data_MultiPointModel.csv")
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
# Below are histograms plotted for each buffer level and each individual trucking metric. The x- and y-scales are the same for each row (for each metric) to make it easier to evaluate the impact of the buffers. What's clear is that for trucking values for a given EPA sensor fall in a very narrow range in the 100m buffer (with many values being 0), and while the 1000m buffer demonstrates some spread, most values across all trucking metrics are pretty low at both the 100m and 1000m buffers. The values really spread out at the 5000m buffer.

# %%
#Plotting histograms
fig, axs = plt.subplots(5, 3, sharey='row', sharex='row', figsize=(15,15))
buff = ['100m', '1000m', '5000m']
metric = ['AADT12', 'FAF12', 'NONFAF12', 'YKTON12', 'KTONMILE12']
for i in range(len(metric)):
    for j in range(len(buff)):
        axs[i,j].hist(df_truck[buff[j]+metric[i]])
        axs[i,j].set_title(buff[j]+metric[i])
plt.tight_layout()
plt.show()

# %% [markdown]
#    ## Forecasting and Prediction Modeling (25 points)
# 
#    This section is where the rubber meets the road.  In it you must:
#    1. Explore at least 3 prediction modeling approaches, ranging from the simple (e.g. linear regression, KNN) to the complex (e.g. SVM, random forests, Lasso).
#    2. Motivate all your modeling decisions.  This includes parameter choices (e.g., how many folds in k-fold cross validation, what time window you use for averaging your data) as well as model form (e.g., If you use regression trees, why?  If you include nonlinear features in a regression model, why?).
#    1. Carefully describe your cross validation and model selection process.  You should partition your data into training, testing and *validation* sets.
#    3. Evaluate your models' performance in terms of testing and validation error.  Do you see evidence of bias?  Where do you see evidence of variance?
#    4. Very carefully document your workflow.  We will be reading a lot of projects, so we need you to explain each basic step in your analysis.
#    5. Seek opportunities to write functions allow you to avoid doing things over and over, and that make your code more succinct and readable.
# 
# %% [markdown]
#    ### Single-point prediction model (Question 1)
#    In summary, this model attempts to predict PM2.5 levels at the location of a specific EPA sensor (as described in the data cleaning section) in the Bay Area. It combines EPA data, Beacon data, NOAA data (across 5 data types), and PurpleAir PM2.5 data.
# 
#    In terms of modeling approaches, this single-point model draws on OLS, lasso, ridge, and the elastic net. We ran into two principal issues in our data, which motivated our investigation of sub-questions:
#    * **Data quality:** Many of our predictors had incomplete data for the year (ie, less than 8760 hours' worth of data). We explored various "cutoffs" designating the maximum number of missing data points that any given column could have in order to remain in the dataset. After setting the cutoff, we removed any observations that had any NaNs remaining--thus, increasing the cutoff led to fewer observations being used for the model.
#    * **Collinearity:** Because collinearity was a big issue in our data for both PurpleAir and NOAA (ie, nearby sensors had highly correlated readings), both lasso and elastic net often failed to converge during gradient descent. As such, we leaned on ridge and OLS predominantly. However, we also tried to directly address collinearity per the suggestion of ISLR, by building a function to purge a dataframe of variables exceeding a certain correlation threshold.
# 
# ### Multi-point prediction model (Questions 2 and 3)
# 
#    In summary, this model attempts to predict PM2.5 levels at every CA EPA sensor across specific hours (which also act as inputs). The multi-point model draws on OLS, lasso, and ridge. We ran into one principal issue with this model:
#    * **Data/Observation Availability** Many of our features did not have data points that easily matched neatly within the rows meant to represent a PM2.5 reading at a specific sensor (i.e., other input data didn't match neatly to EPA data). This was especially problematic for Question 2 - our multi-point model used several input data sources which meant that many rows were culled because they featured at least one null value.
# 
#   Below we provide the code for our single-point and multi-point models, starting first with single-point:

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
df_beac = pd.read_csv(filehome+"Beacon_Data_SinglePointModel.csv")
df_epa = pd.read_csv(filehome+"EPA_Data_SinglePointModel.csv")
df_noaa = pd.read_csv(filehome+"NOAA_Data_SinglePointModel.csv")
df_pa = pd.read_csv(filehome+"PA_Data_SinglePointModel.csv")

# %% [markdown]
#   In the course of building the model, we noticed that the lasso model wasn't converging. When features are highly correlated, as we learned in class, lasso can become unstable. As such, we decided to A) investigate what correlation was present in the NOAA and PurpleAir datasets (which are contributing the bulk of the predictors and are very likely correlated), and B) see if we could address it by removing some of the semi-redundant variables. (This is one of the methods suggested by ISLR, although they don't give much detail on how to do it--so we did a few sensitivities.) After doing all this work, we determined that the converge issue could be solved by increasing the number of iterations allowed for gradient descent in our main functions, but we still find the results interesting!
# 
#   We wrote a function to identify which variables in a dataframe exceed a certain correlation limit and to drop the redundant rows (e.g., if one variable is 98% correlated with another, it's not adding much value to the model).

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
#    We used this function to "purge" the NOAA and PurpleAir datasets of a certain amount of correlated data. See the visual representation of the correlation matrices for each below. Many of the nearly-1 columns are gone, decreasing the amount of bright yellow you see, and the overall number of features is reduced.

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
#    Next we decide how many missing values we are willing to tolerate, and designate a "cutoff" we use to drop columns with too many NaNs. We then remove each row with any NaNs in it, so that we can analyze a clean, complete dataframe. We keep track of the cutoff as well as the resulting number of observations and predictors we're working with, in case it affords any insight into our model's functioning...
# 
#    We then formatted our dataframe into X's and y's to be able to feed into sklearn models. Importantly, we standardized our data, because the scale of the data makes a dfiference when using ridge.
# 
#    We packaged this into two functions to be able to more easily manipulate the cutoff and produce X and y.

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
#    Finally, we defined the two functions that we used to generate and tune our candidate models:
#    * **fit_model**, which fits a model (either OLS, ridge, lasso, or elastic net) to the training data and returns the MSE and a list of the coefficients of the model.
#    * **model_cv_mse**, which conducts a k-fold cross validation using the training dataset. This optimizes the hyperparameter alpha (or lambda) in the ridge/lasso/elastic net models, but doesn't make sense for OLS (which doesn't have a hyperparamter like that). We defaulted to using k=5 on the recommendation of ISLR, which recommends a k between 5 and 10.

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
def model_cv_mse(Model, X, y, alphas, k = 5):
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
#    Finally, we write a function that puts it all together: it takes in the original dataframe, the data missingness cutoff parameter, and outputs a list of MSEs. We use the model_cv_mse function to fit the lasso/ridge/elasticnet models and tune their hyperparameters via cross-validation (using the validation data partitioned out of the training set), and then call fit_model for all four models, including OLS (using the test data partitioned out of the full dataset at the beginning).

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
#   Finally, we merged all of our data (waiting until this late in the notebook to do so to make it easier to change the correlation-purge inputs) and ran the models across a series of data quality cutoffs, making the limit for missing values for a predictor range from 175 to 975.

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
#    We then reran the above code three more times--once each for 95%, 85%, and 75% purges for the correlation matrices.

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
#Plotting all data to see which model, at which cutoff and with which collinearity purge,
#performs best

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

# %% [markdown]
#  Finally, we take a look at bias and variance in our model. We do this in two ways:
#  * **Assess bias and variance for the model that seems to be the best** (ridge model, ~380 cutoff, .95 correlation limit) and contrast with a model that technically has lower MSE but seems more volatile (ridge, ~900 cutoff, no correlation correction). We look at the training vs test error for each of these models--if the training error is significantly lower than the test error, the model likely has higher variance / lower bias, whereas if the training error is only somewhat lower or equal to the test error, the model likely has lower variance but higher bias.
#  * **Look how error varies across the independent variable**--does it have higher variance at some values of y?
# %% [markdown]
#  The function written below outputs train/test MSE and train/test adjusted R2 for the apparently stable best model and the volatile best model (ie, the two models with the lowest test MSE above)

# %%
from sklearn.metrics import r2_score
def compare_trainWtest(Model, quality_cutoff, df_noaa, df_pa):
    """
    Takes in the model, the quality cutoff for missing data, and the 
    NOAA and PurpleAir dataframes (adjusted or not for correlation issues),
    and outputs (prints) the train and test MSE and R2
    Returns y_pred_test and y_test for the residual investigation later on
    """
    
    #Merging dataframes
    bigdf = df_epa.merge(df_beac, how='left', on='datetime').reset_index(drop=True)
    bigdf.drop(columns=['Laney College'], inplace=True) #laney college is messed up
    bigdf = bigdf.merge(df_noaa, on='datetime', how='left')
    bigdf = bigdf.merge(df_pa, on='datetime', how='left')

    #Shaping data
    bigdf_subset = set_cutoff(bigdf, cutoff=quality_cutoff)
    X_train, X_test, y_train, y_test = process_XY(bigdf_subset)
    
    #Getting hyperparam of model with best CV MSE
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 10**5, 10**6, 10**7]
    mses = model_cv_mse(Model, X_train, y_train, alphas, k=5)
    optimal_alpha = alphas[np.argmin(mses)]
    
    #Fitting model
    model = Model(max_iter = 10**7, alpha = optimal_alpha, tol=.01)
    model.fit(X_train, y_train)    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test) 
    mse_train = mean_squared_error(y_pred_train,y_train)
    mse_test = mean_squared_error(y_pred_test,y_test)

    #Getting adjusted R2
    n = bigdf_subset.shape[0]
    p = bigdf_subset.shape[1]
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    adj_r2_train = (1 - (1-r2_train)*(n-1)/(n-p-1))
    adj_r2_test = (1 - (1-r2_test)*(n-1)/(n-p-1))
    
    print("Training MSE is:", mse_train)
    print("Test MSE is:", mse_test)
    print("Training adjusted R2 is:", adj_r2_train)
    print("Test adjusted R2 is:", adj_r2_test)
    print("# obs is:", n)
    print("# predictors is:", p)
    
    return y_pred_test, y_test


# %%
###GOOD MODEL
#Params
lim=.95

#Dropping correlated terms
df_noaa_new = drop_correlated_cols(df_noaa, limit=lim)
df_pa_new = drop_correlated_cols(df_pa, limit=lim)

y_pred_test, y_test = compare_trainWtest(Model=Ridge, quality_cutoff=355, df_noaa=df_noaa_new, df_pa=df_pa_new)


# %%
###VOLATILE MODEL
compare_trainWtest(Model=Ridge, quality_cutoff=910, df_noaa=df_noaa, df_pa=df_pa)

# %% [markdown]
#  Wow--the "good" model does indeed seem to be good--the training and the test MSE are similar, and the training and the test adj. R2 are not radically different--the test error is a little higher than the training error, as you'd expect. This indicates that the model is relatively less flexible, with more bias and lower variance.
# 
#  The volatile model shows itself to be incredibly overfitted--the training MSE is near-0 (which makes sense; the model probably went straight through the only 20 observations it was working with), whereas the test MSE is low but many orders of magnitude higher. The adjusted R2 values are hard to interpret, being >1.
# 
#  The second model doesn't seem "good", anyway--20 observations and almost 800 predictors won't predict anything very well, so we were right to think the first was a better model.
# %% [markdown]
#  **Investigating residuals**
#  Next we plot the model residuals against the dependent variable to see if the model predicts equally well at all values of y.

# %%
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
resids = y_pred_test - y_test
plt.plot(y_test, resids, 'ob')
plt.axhline(y=0)
plt.xlabel("EPA sensor reading (ug/m3)")
plt.ylabel("Residual from best model estimate (ug/m3)")
plt.title("Assessing patterns in residuals from best model")

plt.subplot(1,2,2)
resids_log = np.log(y_pred_test) - np.log(y_test)
plt.plot(y_test, resids_log, 'ob')
plt.axhline(y=0)
plt.xlabel("EPA sensor reading (ug/m3)")
plt.ylabel("LOG-TRANFORMED Residual from best model estimate (ug/m3)")
plt.title("Assessing patterns in LOG-TRANSFORMED \n residuals from best model")
plt.show()

# %% [markdown]
#  This assessment shows that at low pollution levels, our model has a significant positive bias, and at higher levels (roughly above the mean and below the mean), there is a significant negative bias. This is troubling, because the danger of underestimating high levels of pollution is greater than vice versa.
# 
#  However, a rough log-transformation on our y helps get the residuals centered around 0, which is desireable--we would consider (perhaps later in this notebook if time permits!) log-transforming y before running our analysis. The residual at low values of y in the log-transformed version are still quite biased, but we don't care that much about our accuracy on the very lowest-pollution days (I'd say), at least compared to polluted days.
#
#   Below we provide the code used for the multi-point model. In the interest of space, we will only be providing the code to run the model for one of our selected cases. We will identify the point in the model at which users specifically designate which hour they are trying to analyze:
# %%
#Load dependencies
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
from datetime import timezone
import zipfile
import pickle

pd.set_option('display.max_columns', 500)

# %%
#Load all data and perform minor cleaning
df_epa = pd.read_csv("EPA_Data_MultiPointModel.csv")
df_epa.head(1)

df_beac = pd.read_csv("Beacon_Data_MultiPointModel.csv")
df_beac.head(1)

df_noaa = pd.read_csv("NOAA_Data_MultiPointModel.csv")
df_noaa.head(10)

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
df_truck.head(1)

df_pa = pd.read_csv("pa_full.csv")
df_pa = df_pa.rename(columns={"lat":"latitude","lon":"longitude"})
df_pa['name'] = 'none'
cols = ['datetime', 'latitude', 'longitude','name','PM1.0 (ATM)','PM2.5 (ATM)','PM10.0 (ATM)','PM2.5 (CF=1)','id']
df_pa = df_pa[cols]
df_pa.head(10)

# %%
# Just as an early point of exploration, here we find the most commonly represented hours in each data set

group_epa = df_epa.groupby("datetime")
epa_counts = group_epa["datetime"].value_counts()

EPA = epa_counts.loc[epa_counts==max(epa_counts)]
print(EPA)

group_noaa = df_noaa.groupby("datetime")
noaa_counts = group_noaa["datetime"].value_counts()

NOAA = noaa_counts.loc[noaa_counts==max(noaa_counts)]
print(NOAA)

group_beac = df_beac.groupby("datetime")
beac_counts = group_beac["datetime"].value_counts()

BEAC = beac_counts.loc[beac_counts==max(beac_counts)]
print(BEAC)

group_pa = df_pa.groupby("datetime")
pa_counts = group_pa["datetime"].value_counts()

PA = pa_counts.loc[pa_counts==max(pa_counts)]
print(PA)

# %%
# Here is where we define our major functions that help us create our final DFs for analysis
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
# %%
# In this cell we create our dataframe for analysis by running our functions and performing some minor cleaning (removing nulls, renaming columns etc)
# A note: this is where users define which hour of the year they wish to analyze. Within the get_final_timevarying_dataframe function, we pass in the month of May, 2nd day, 8th hour. This is how we manually choose our cases. In retrorespect, it would have been smoother to have written a loop but alas time constraints made manual inputting easiest. To be extra clear, we only provide the code for one discrete "case" (hour) out of the 7 hours identified in our poster project.
df_analysis = get_final_timevarying_dataframe(df_epa, [df_noaa, df_beac,df_pa], month=5,day=2,hour=8)
df_analysis = df_analysis.drop(columns=['latitude_y','longitude_y','id','latitude','longitude'])
df_analysis = df_analysis.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'})
df_analysis = df_analysis.merge(df_truck, on=['latitude','longitude'], how='left') 
df_analysis = df_analysis.drop(columns=['precip_accum_one_hour_set_1'])
nullcount = df_analysis.isnull().sum(axis=1)
df_analysis = df_analysis[nullcount == 0].reset_index(drop=True)

# %%
# Here we load in a few more dependencies and start the x / y, test/train split process
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

y = df_analysis['epa_meas']
X_raw = df_analysis.drop(columns=['epa_meas','latitude','longitude','datetime','name'])
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)
# %%
# Now we define a new function to fit our models
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
# %%
#Here's a quick loop to use our function to run through OLS, Ridge, Lasso and print out the MSEs and Coef values
mses_all = {} 
coefs_all = {}
for model in [Ridge, Lasso, LinearRegression]:
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test)
    mses_all[model] = mse
    coefs_all[model] = coef

mses_all
coefs_all
# %%

#Here's an intersting visualization of our beta values by model selection
entries = np.arange(0,len(coefs_all[Lasso])) #chose Len of Lasso but should get same result across all three
plt.figure(figsize=(20, 12))
plt.subplot(1, 3, 1)
plt.scatter(x=entries,y=coefs_all[Lasso], c='b')
plt.scatter(x=entries,y=coefs_all[Ridge], c='g')
plt.scatter(x=entries,y=coefs_all[LinearRegression], c='r')
plt.title('Coefficient Values (Betas)')
plt.xlabel('Coefficients')
plt.ylabel('Coefficient Values')
# %%
# Now we start to really get into the modeling, here we do a K-fold cross validation with differing values of alpha
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

# %%
#Now we re-run all 3 of our models with optimized alphas

mses_all = {}
coefs_all = {}

models = [Ridge, Lasso, LinearRegression]
alphas = [alphas_ridge[np.argmin(mses_ridge)], alphas_lasso[np.argmin(mses_lasso)], 0]

for model, alpha in zip(models,alphas):
    mse, coef = fit_model(model, X_train, X_test, y_train, y_test, alpha)
    mses_all[model] = mse
    coefs_all[model] = coef

mses_all

coefs_all

#Visualization of Beta Coefficients

entries = np.arange(0,len(coefs_all[Lasso])) #chose Len of Lasso but should get same result across all three
plt.figure(figsize=(20, 12))
plt.subplot(1, 3, 1)
plt.scatter(x=entries,y=coefs_all[Lasso], c='b')
plt.scatter(x=entries,y=coefs_all[Ridge], c='g')
plt.scatter(x=entries,y=coefs_all[LinearRegression], c='r')
plt.title('Coefficient Values (Betas)')
plt.xlabel('Coefficients')
plt.ylabel('Coefficient Values')
plt.savefig('Initial_run.png', bbox_inches='tight')

#This concludes 1 case run for our multi point model for question 2. For the sake of reiteration, this multi point model takes in multiple data sources in addition to the base EPA data. Now we move to a modified version of this model where we only use the 4 features from PurpleAir as predictors for question 3. Because this code is adapted from the multi-point model introduced above, we won't spend the time setting up dependendies and loading in data, it's already been done above. Instead, the modeling below is meant to show outcomes from when we only use PA data with the EPA data to forecast the EPA data:

# %% [markdown]
#    ## Interpretation and Conclusions (20 points)
#    In this section you must relate your modeling and forecasting results to your original research question.  You must
#    1. What do the answers mean? What advice would you give a decision maker on the basis of your results?  How might they allocate their resources differently with the results of your model?  Why should the reader care about your results?
#    2. Discuss caveats and / or reasons your results might be flawed.  No model is perfect, and understanding a model's imperfections is extremely important for the purpose of knowing how to interpret your results.  Often, we know the model output is wrong but we can assign a direction for its bias.  This helps to understand whether or not your answers are conservative.
# 
#    Shoot for 500-1000 words for this section.
# %% [markdown]
#    ### Single-point model
# 
#  Original question: Should California replace faulty or broken regulatory-grade sensors (i.e., can we predict air pollution to a satisfactory level without EPA sensors being present)? A good model would predict air pollution at a single point with a low enough error rate to be as precise as an actual EPA sensor would be. In that case, you could either avoid replacing ailing sensors, or make high-quality predictions for areas where no sensors have been allocated but air pollution is critical (e.g., West Oakland). (However, we would have to consider how areas that have never received a government sensor might differ systematically from areas that have--maybe West Oakland has such bad air qulity that our model wouldn't work well on it after all.)
# 
#  Given our results so far, we'd tell a decisionmaker to keep investing in high-quality sensors (at least in areas where our accuracy is lower than the 7-15% recommended by the EPA). Alternatively, we could recommend that the EPA put resources into a data science team to investigate what other predictors could potentially improve the model's accuracy: more land use data, daily/hourly traffic flow data, more and more PurpleAir data, wind-direction-based buffers (good idea, Duncan! Wish we'd had time)...all of these seem like they could help. In light of these alternatives, a reader should either care about this work because A) they're grumpy about spending tax dollars on air pollution sensors and want to see it justified, or B) they're interested in the fact that we can do **pretty** well estimating air pollution without the sensors--just not good enough for government work.
# 
#  Caveats abound for this model:
#  * As mentioned above, there were many features we could have included but didn't
#  * We didn't assess the model's performance at other EPA sensor sites; we just did the one in the Bay Area so we could use the Beacon data
#  * As discussed above, our model errors are skewed high below the mean and low above the mean; we also don't know how the model performs on days with very high air pollution, since they weren't in the test set
#  * The process we used to purge the dataframe of correlation was pretty cool (I think) but not best-practice according to ISLR, although best practice would have been way too computationally intensive. It is worth noting again that we originally developed this purging method because our lasso model wouldn't converge, but then we just cranked up the number of iterations for gradient descent and it worked...so this correlation-purging experiment is just a neat sensitivity.
#  * A log-transformation of each air pollution data input (which are all skewed) would probably benefit this work
# 
#   ### Multi-point model
# 
#    Original Question: When should Californians plan to purchase N95 masks?
# 
#    Given our results so far, we can't determine optimal times for Californians to purchase N95 masks. This question requires us to get consistently strong forecasts for all 8760 hours of the year, and in particular we need to be able to forecast air quality on days and hours where it's expected to be quite poor. Additionally:
# 
#    * By far the most interesting insight revealed by running our model for 7 discrete hours throughout 2018 is that the number of observations utilized in each model run matters. Our best (i.e., lowest MSE) model run was for May 14 at 6:00 am. We had just over 2000 observations split into test/train/validate sets. Our MSE for each of our model fits was between 1 and 2.2. That’s pretty good, especially relative to some of our other runs. Our worst run was actually for November 11 at 7:00 am, exactly 3 hours after the ignition of Camp Fire. We chose this time thinking that there may be some additional fundamental value of our model. That is, can the model predict the EPA readings after a major fire event without explicitly knowing that a fire had occurred. Unfortunately, with only 62 observations to use, the MSE for each of our three fits was very poor, ranging from 620 to over 1000. It’s unclear at this point, if the poor performance is only a function of the lack of useful observations used to train and test the model, or if there are other features associated with fire events that we should have included to provide better predictive power.
#    * Ultimately, it’s hard to really know truly how useful this multi point model is. We’re hesitant to give the general model (irrespective of OLS, Ridge, Lasso) much credence, if only for the seemingly high variability in MSEs across different hours of the year. An interesting next step would be to loop our model in all 8760 hours of the year and find average hourly MSEs. Perhaps many or most of the hours have enough observations that for any one hour, the models will provide meaningfully accurate results. On the other hand, this may not be true, and perhaps many hours of the year have few observations like the Nov 11 7:00 am time - and therefore the model we’ve built may have limited applicability. Still, what we can say is that overall, there seems to be (based off of 7 sample runs, admittedly, not a great sample size to judge this off of) a threshold number of observations that can help us identify which hours are likely to be better forecasted by our model.
# 
#    Original Question: Should the government subsidize the purchase of PurpleAir or other low quality air monitors?
# 
#    Our results again are a toss up: While the models perform reasonably well in certain cases (May 2 and June 7), we see a large variance in the MSEs across our cases - again, the case of Nov 11, 72 hours after the ignition of the 2018 Camp Fire. The real world application of our analysis is as follows: it may be possible to further subset our forecasts to see if there are certain EPA sensors that we can more reliably forecast for. If we can accurately and consistently forecast certain EPA sensors by only the PurpleAir input data, we may be able to convince state decisionmakers to spend less money on replacing faulty or broken EPA sensors, and instead use a cluster of local and cheaper PurpleAir sensors to understand and record air quality. Additionally:
#    * Much like our analysis in question 2, it’s worth looking again at how our different models performed across different hours. Also similar to the prior question, we’re seeing a lot of variation between hours. Unlike question 2, it doesn’t appear to solely be motivated by the number of observations used to train and test our model. For example, our first two cases on May 2 and June 7 both contain around 140 observations. The MSEs for our 3 models across those two cases ranges from 8.8 to 13.7. Our next case on May 14 features about half as many observations (70) but a better MSE across all three models, ranging from 4.8 to 5.3. This partially calls into question this particular modeling exercise because it shows that increasing the number of observations will not always correspond to an decrease in MSE. Perhaps there’s something specific to these hours that we haven’t accounted for in this particular question, but it’s difficult to say one way or the other.
# 

# %%
