#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Input Data Description (5 points)
Here you will provide an initial description of your data sets, including:
get_ipython().set_next_input('1. The origins of your data.  Where did you get the data?  How were the data collected from the original sources');get_ipython().run_line_magic('pinfo', 'sources')
2. The structure, granularity, scope, temporality and faithfulness (SGSTF) of your data.  To discuss these attributes you should load the data into one or more data frames (so you'll start building code cells for the first time).  At a minimum, use some basic methods (`.head`, `.loc`, and so on) to provide support for the descriptions you provide for SGSTF. 

[Chapter 5](https://www.textbook.ds100.org/ch/05/eda_intro.html) of the DS100 textbook might be helpful for you in this section.


# In[ ]:

For our project we used the following data sets:

1: Purple Air Data;
2: Berkeley Enviro Air-Quality and CO2 Observation Network (BEACON);
3: EPA;
4: NOAA Weather Data; and
5: Trucking Data.

Purple Air:

    We obtained historic Purple Air data through two workflows. First, we used a python script to create a scraped list (using a REST API) of every purple air sensor including their
    Channel IDs, read key numbers, lat/lon, and device location (outside/inside). We filtered for outside devices and ported the data into R where we performed a 
    spatial subset to isolate only sensors in California. We ported the data back into python and used another script with a REST API to query the ThingSpeak database which hosts
    all of the historic purple air data. The data was initially collected from discrete sensors - each purple air sensor has two channels (A and B). Purple Air takes readings from these
    channels every 120 seconds and stores them via the ThingSpeak API, a platform used to host large amounts of data. Below we read in our data:

df_pa = pd.read_csv("pa_full.csv")
df_pa.rename(columns={"lat":"latitude","lon":"longitude"})
df_pa.head(10)

df_pa['PM2.5 (ATM)'].plot.hist(bins=50,range=(0,100))

    Let's first consider the granularity of our data. We have 8 columns which represent a datetime, PM1.0, PM2.5, PM10.0, PM2.5 with a correction factor, lat, lon, and id numbers.
    Each cell of our datetime column contains the datatime in ___ format (UNX/UTC?). Each row in this dataframe represent a measurement from a CA sensor
    at a given hour throughout the year. Each cell of our lat and lon columns provide the discrete latitude and longitude for where the specific sensor is
    located. The ID numbers provide the Purple Air identification for that particular sensor. The readings for the air quality are in the standard ug/m3 format.

    The structure of this data set is an 8 column by 4424491 row dataframe which we read in from a CSV file. 

    With respect to scope, because we subset the entire universe of PA sensors by just those in CA, we know that the scope is adequate for our purposes.

    The temporality of the purple air data set is also adequate for our purposes. Because 2018 was the last full calendar year in which we have access to 12 
    months of data, we were sure to query the ThingSpeak API for all 2018 data from the subset of California sensors. 
    The datatime column provides the year, month, day, and hour of when the specific row (reading) occured. 

    As for faithfulness, we do believe it captures reality. Given how much information is collected from these sensors every 120 seconds, it's unlikely that any
    of the values we have here are hand-entered or falsified. Additionally, the actual cells themselves seem to reflect realistic or accurate findings.

    Our histogram plotted above is meant to illustrate the range of values for one of our features of interest. In this case, an expected long tailed distribution
    is created. Again, based on the results of the visualization, it appears as though the data has not been tampered with and represents reality. 


BEACON:

    We obtained BEACON by ______. Let's take a closer look at the BEACON data:

df_beac = pd.read_csv("Beacon_Data_MultiPointModel.csv")
df_beac.head(10)
df_beac['beac_data'].plot.hist(bins=50,range=(0,10))


    The structure of this data set is a 5 column and 70080 row dataframe, which we read in from a CSV file. We have a lat, lon, name of location, datetime, and 
    reading from the individual node. Each row represents a reading at a certain location at a certain hour during 2018. The Lat and Lon data correspond to where
    the node is located. The scope of this data almost perfectly matches our projct given that each CA sensor is within the bay area. It would have been better if 
    there were BEACON nodes across the entire state, but this may help us getting an accurate forecast for some of our areas nearby. Much like the PurpleAir data,
    because the data is collected from sensors, there appears to be little reason to doubt the reliability of what we've collected. The lat and lons seem to make sense,
    and the actual readings themselves match with what we'd expect as well. 

    The temporality is also adequate, the data contained within is for 2018. Our histogram of the beacon data also shows a similar long-tailed distribution of our feature
    of interest. 

EPA:

    We collected data from the EPA via a REST API. The API is used to collected data from the EPA's Air Quality System (AQS) database. From the EPA's website:

    "AQS contains ambient air sample data collected by state, local, tribal, and federal air pollution control agencies from thousands of monitors around the nation.
    It also contains meteorological data, descriptive information about each monitoring station (including its geographic location and its operator), and information
    about the quality of the samples."

    Let's take a look at our EPA data after we've cleaned it:

df_epa = pd.read_csv("EPA_Data_MultiPointModel.csv")
df_epa.head(10)
df_epa['epa_meas'].plot.hist(bins=100,range=(0,120))


    Our dataframe consists of 5 columns and 67801 rows pulled from a CSV file. Because the data was extracted via a REST API, we were able to pull only 2018 data from 
    California. The data set includes the latitude and longitude from each of the 119 distinct EPA sensors across CA. Each observation (row) represents a reading from
    a sensor at a specific hour throughout 2018. The PM2.5 reading is given in the column "epa_meas". That column is what we are ultimately going to be trying to forecast.

    Our histogram of the EPA data shows another right tailed distribution. Interestingly, there appears to be one bin in particular that is much more frequent than the rest.

NOAA:

df_noaa = pd.read_csv("NOAA_Data_MultiPointModel.csv")
df_noaa.head(10)
df_noaa['wind_speed_set_1'].plot.hist(bins=50,range=(0,20))
df_noaa['air_temp_set_1'].plot.hist(bins=50,range=(0,20))

Our NOAA data (like our other data) was collected through a REST API. NOAA maintains historic information regarding weather, including wind direction, wind speets, air pressure,
temperature. Our dataframe consists of 9 columns and 1033680 observations spread across CA in 2018. The datetimes in this particular frame are for each hour at different weather 
stations throughout CA. Each row (observation) is a measurement taken at a specific station for a given hour during the year. 


# In[ ]:


## Data Cleaning (10 points)
In this section you will walk through the data cleaning and merging process.  Explain how you make decisions to clean and merge the data.  Explain how you convince yourself that the data don't contain problems that will limit your ability to produce a meaningful analysis from them.  

[Chapter 4](https://www.textbook.ds100.org/ch/04/cleaning_intro.html) of the DS100 textbook might be helpful to you in this section.  


# In[ ]:

#Jake note: After re reading this section's expectations and skimming the python scripts that clean the data, it seems like all we
#need here is a written explanation of how we cleaned the data? I can't imagine Duncan actually wants us to throw our cleaning code here.




# In[ ]:


## Data Summary and Exploratory Data Analysis (10 points)

In this section you should provide a tour through some of the basic trends and patterns in your data.  This includes providing initial plots to summarize the data, such as box plots, histograms, trends over time, scatter plots relating one variable or another.  

[Chapter 6](https://www.textbook.ds100.org/ch/06/viz_intro.html) of the DS100 textbook might be helpful for providing ideas for visualizations that describe your data.  


# In[ ]:





# In[ ]:




