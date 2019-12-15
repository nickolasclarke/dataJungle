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

Trucking Data:

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

Purple Air:

df_pa.head(10)

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

Our visualizations reveal a few notable insights. Looking at our comparison of PM2.5 to the PM2.5 with a correction factor,
it appears as though a constant correction factor is applied across the board (i.e., the slope of the line is constant). 
This might mean that purple air sensors are all initially biased in the same direction. Our second visualization comparing PM2.5
to PM1.0 shows that for PM2.5 values above 2000, the relationship to PM1.0 is constant. Interestingly se see a lot of noise below 
2000 on the x axis, meaning that there might be some threshold of PM2.5 that we'd need to see before we see a constant linear relationship
between the two variables. Our last comparison of PM2.5 to PM10.0 reveals a similar trend. Let's view these relationships in a different format:

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

The scatter plots above show us the distribution of our random sample of Purple Air PM readings over 2018.
Let's look at the raw scatter plot of PM2.5 readings with no correction factor included:

plt.scatter(pa_sample["datetime"], pa_sample["PM2.5 (ATM)"], s = 5, color = 'c')
plt.title("2018 PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

Based on our scatterplot, we see that most of our readings from this sample fall between 0 and 20 throughout the year. There are
values that exceed 20 here and there, but they seem to be on the higher side of the sample. If we plot another histogram of our sample
we can see this:

pa_sample['PM2.5 (ATM)'].plot.hist(bins=60,range=(0,70))


EPA: 

df_epa['epa_meas'].plot.hist(bins=100,range=(0,120))

We included this histogram above, but we include it again to higlight the distribution of our EPA readings that we
are interested in forecasting. 

epa_sample = df_epa.sample(n=60,random_state=1)

plt.scatter(epa_sample["datetime"], epa_sample["epa_meas"], s = 5, color = 'c')
plt.title("2018 EPA PM 2.5 readings")
plt.xlabel("Date")
plt.ylabel("PM readings")
plt.show()

Because our data set is so large, we needed to take a random sample to actually plot the readings over the time of the year.
It appears as though the readings are mostly consistent throughout the year with one outlier in the middle of the year, and a few higher
points scattered throughout.


# In[ ]:





# In[ ]:




