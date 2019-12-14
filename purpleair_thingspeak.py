import os
import json
import requests
import thingspeak

import datetime
import pandas as pd 

from functools import reduce
from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_block(start, end):
    params = {  'start': str(start),
                  'end': str(end),
              'average': '60'
             }
    try:
        r = channel.get(params)
    except:
        raise
        print('error')

    data = [list(feed.values()) for feed in json.loads(r)['feeds']]
    block_df = pd.DataFrame(data, columns = cols)
    return block_df


sensor_list = pd.read_csv('pa_sensor_list.csv')
#sensor_list = sensor_list.drop(columns=['Unnamed: 0', 'DEVICE_LOCATIONTYPE'])
#group the channel ID and key into pairs
pairs = list(zip(sensor_list['THINGSPEAK_PRIMARY_ID'],
                 sensor_list['THINGSPEAK_PRIMARY_ID_READ_KEY']))
#split data into 7.2 days, ~ 8000 data points @ 80 second resolution. 8000 is the Thingspeak limit
dates = pd.date_range(start='1/1/2018', end='1/4/2019', freq='7.2D')

def download_pa():
    #TODO desperately needs to be async
    for cid, key in pairs:
        channel_id = cid
        read_key   = key

        channel = thingspeak.Channel(id=channel_id,api_key=read_key)
        #prep columns for dataframe
        cols = json.loads(channel.get({'results': '1'}))['channel'] # get the json for channel columns
        cols = [(key, value) for key, value in cols.items() if key.startswith('field')] # extract only field values
        cols = [unit for field, unit in cols]
        cols = [col + f'_{channel_id}' for col in cols]
        cols.insert(0,'datetime')
        df = pd.DataFrame(columns = cols)

        for start, end in pairwise(dates):
            df = df.append(get_block(start,end))

        # ['datetime', 'PM1.0 (ATM)_{channel_id}', 'PM2.5 (ATM)_{channel_id}',
        #    'PM10.0 (ATM)_248887', 'Mem_248887', 
        #    'Unused_248887', 'PM2.5 (CF=1)_248887']

        # # df = df.drop(columns = [f'Uptime_{channel_id}', 
        # #                         f'RSSI_{channel_id}',
        # #                         f'Temperature_{channel_id}',
        # #                         f'Humidity_{channel_id}'
        # #                        ])

        df.to_csv(f'data/purple_air/{channel_id}.csv')
        print(f'Channel: {channel_id} processed')


def clean_pa(data, path):
    thingspeak_id = int(data[:6])

    df = pd.read_csv(path + data)
    df = df.drop(df[df['datetime'].duplicated() == True].index)
    df = df.drop(df[df['datetime'] > '2018-12-31T23:00:00Z'].index)

    if df.shape[0] >= 6000:
        try:
            lat, lon = zip(*sensor_list[
                sensor_list['THINGSPEAK_PRIMARY_ID'] == thingspeak_id][['lat', 'lon']].values)
            df['lat'] = lat[0]
            df['lon'] = lon[0]
            df['id']  = thingspeak_id
        except:
            print('lat / lon not found, skipping')
            return pd.DataFrame()
        #filter for only the columns we want
        df = df[df.columns[df.columns.str.startswith((
            'datetime','lat','lon','PM','id'))]]
        return df
    else:
        return pd.DataFrame()

def pa_concat(path):
    files = os.listdir(path)
    cols  = ['datetime',     'PM1.0 (ATM)', 
             'PM2.5 (ATM)',  'PM10.0 (ATM)', 
             'PM2.5 (CF=1)', 'lat', 
             'lon',          'id'
             ]

    df = pd.concat([
         pd.read_csv(path+file, names = cols, skiprows=1, parse_dates=['datetime']) for file in files])
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def reducer(x,y):
    df = pd.merge(x,y,left_index= True, right_index= True, how='outer')
    print(df.shape)
    return df

def pivot_pa(path):
    files = os.listdir(path)
    dfs = [pd.read_csv(path+file,).set_index('datetime') for file in files]
    pivot_df = reduce(lambda x,y: reducer(x,y), dfs)
    print('subset columns')
    pivot_df = pivot_df[pivot_df.columns[pivot_df.columns.str.startswith(('PM1.0 (ATM)', 'PM2.5 (ATM)','PM10.0 (ATM)', 'PM2.5 (CF=1)'))]]
    pivot_df.index = pd.to_datetime(pivot_df.index)
    return pivot_df

files = os.listdir('data/purple_air/')
cleaned_path = 'data/pa_cleaned/'

[clean_pa(file, 'data/purple_air/').to_csv(cleaned_path+file,index=False) for file in files]
#concat individual sensor DF together, columns being common indicators
full_df = pa_concat('data/pa_cleaned/')
full_df.to_csv('pa_full.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
#pivot that table to have unique columns for each sensor+indicator, along the same hour
pivot_df = pivot_pa(cleaned_path)
# print(pivot_df.head())
# pivot_df.to_csv('pa_pivoted.csv', date_format='%Y-%m-%d %H:%M:%S')

# subset the pivoted table into seperate CSV's for each indicator type
sub_dfs = [{indicator: pivot_df[pivot_df.columns[pivot_df.columns.str.startswith(('datetime',f'{indicator}'))]]} for indicator in ['PM1.0 (ATM)', 'PM2.5 (ATM)','PM10.0 (ATM)', 'PM2.5 (CF=1)']]

def write_sub_df(val):
    key = list(val.keys())[0]
    df  = val[key]
    df.to_csv(f'data/{key}.csv', date_format='%Y-%m-%d %H:%M:%S')
    print(f'{key} written')

#write these subset dfs to csv
[write_sub_df(val) for val in sub_dfs]



# #group channels into A, B pairs #TODO need parent_ID column to complete
# for group in sensor_list.groupby(['lat', 'lon']):
#     key_pairs = list(zip(
#         list(group[1]['THINGSPEAK_PRIMARY_ID']),
#         list(group[1]['THINGSPEAK_PRIMARY_ID_READ_KEY'])))
#     pairs.append(key_pairs)

# #https://api.thingspeak.com/channels/9/feeds.json?start=2018-01-01%2000:00:00&end=2018-12-31%2023:59:59&average=60
# r_json = json.loads(r)
# with open('personal.json', 'w') as json_file:
#     json.dump(r_json, json_file)
