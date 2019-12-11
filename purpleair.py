# pull 3rd part purpleair data from json api
import json
import urllib
import requests
import time
from datetime import datetime
import calendar
import sys
import itertools
import os
import pandas as pd
import numpy as np

file_name = "pa_id_key.txt"
dir_name = "/data/pa_id_key/"
full_path = os.getcwd() + "//" + dir_name + "//" + file_name
row = 0

d = datetime.utcnow()
unixtime = calendar.timegm(d.utctimetuple())

df = pd.DataFrame(columns=['datetime'
    ,'ID'
    ,'ParentID'
    ,'Label'
    ,'DEVICE_LOCATIONTYPE'
    ,'THINGSPEAK_PRIMARY_ID'
    ,'THINGSPEAK_PRIMARY_ID_READ_KEY'
    ,'THINGSPEAK_SECONDARY_ID'
    ,'THINGSPEAK_SECONDARY_ID_READ_KEY'
    ,'Lat'
    ,'Lon'
    ,'PM2_5Value'
    ,'LastSeen'
    ,'State'
    ,'Type'
    ,'Hidden'
    ,'Flag'
    ,'isOwner'
    ,'A_H'
    ,'temp_f'
    ,'humidity'
    ,'pressure'
    ,'AGE'
    ,'Stats'
    ])

print

## assigning PurpleAir API to url
url = "https://www.purpleair.com/json"

## GET request from PurpleAir API
try:
    r = requests.get(url)
    print ('[*] Connecting to API...')
    print ('[*] GET Status: ', r.status_code)

except Exception as e:
    print ('[*] Unable to connect to API...')
    print ('GET Status: ', r.status_code)
    print (e)
print

try:
    ## parse the JSON returned from the request
    j = r.json()

except Exception as e:
    print ('[*] Unable to parse JSON')
    print (e)



#Create DF from dictionary of results
new_df = pd.DataFrame.from_dict(j["results"])

#Create New DF with the columns we want to pull into R
df_for_r = new_df[["DEVICE_LOCATIONTYPE","THINGSPEAK_PRIMARY_ID","THINGSPEAK_PRIMARY_ID_READ_KEY","THINGSPEAK_SECONDARY_ID","THINGSPEAK_SECONDARY_ID_READ_KEY"]]
df_for_r["THINGSPEAK_PRIMARY_ID"] = df_for_r["THINGSPEAK_PRIMARY_ID"].astype(int)

#Create DF with only the keys that are listed as outside
outside = df_for_r.loc[df_for_r["DEVICE_LOCATIONTYPE"] == "outside"]

#Set values to be integers
outside['THINGSPEAK_PRIMARY_ID'] = outside['THINGSPEAK_PRIMARY_ID'].astype(int)

#Create series of new primary key ids that are outside but listed as NAN in initial DF
keys = outside['THINGSPEAK_PRIMARY_ID'].apply(lambda num: num + 2)

#Set NaN values to outside provided that the Primary ID matches something in Keys
df_for_r.loc[df_for_r["THINGSPEAK_PRIMARY_ID"].isin(keys), "DEVICE_LOCATIONTYPE"] = "outside"

#final copy for export to r - only outside data
final_DF = df_for_r.loc[df_for_r["DEVICE_LOCATIONTYPE"] == "outside"]

#Save as CSV for import to R
final_DF.to_csv("for_import.csv", sep = ",", index = False, encoding = 'utf-8')


