import sys
import io
import json
import requests
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from noaa_sdk         import noaa

meso_base_url = 'https://api.synopticdata.com/v2/stations/'

mesonet_token = ''

def flatten_dict(dic):
    result = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            result.update(flatten_dict(dic[key]))
        else:
            result[key] = dic[key]
    return result

def get_meso_ts(start, end, stations, token, qc='on', tz='UTC', output='csv'):
    '''Pull timeseries data from mesonet

       If output='CSV', only 1 station per
       request is supported
    '''
    url    = meso_base_url + 'timeseries'
    params = {'start'     : start,
              'end'       : end,
              'stids'     : stations,
              'token'     : token,
              'qc'        : qc,
              'obtimezone': tz,
              'output'    : output
             }

    try:
        r = requests.get(url, params)
        print(f'{stations} downloaded')
    except requests.exceptions.Timeout:
       print('Request Timed Out')
    except requests.exceptions.TooManyRedirects:
       print('Bad request, try again')
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)

    return (stations,r)

def parse_meso(response, output_dir, type='csv'):
    '''Parse response, and write to file.
    '''
    if type == 'csv':
        filename = response[0] + '.csv'
        data     = response[1].content.decode('utf-8')
        with open(output_dir + filename, 'w') as handle:
            handle.write(data)
        return
    else:
        print(f'{type} not yet supported')
        return

n = noaa.NOAA()
#get all stations in CA and convert to df
ca_stations = n.stations(state='CA')
ca_stations = pd.DataFrame.from_dict(ca_stations['features'])
#flatten and clean up columns
ca_stations = ca_stations.join(pd.DataFrame.from_dict([flatten_dict(x) for x in ca_stations['properties']]))
ca_stations = ca_stations.drop(columns=['@id', 'type', 'properties', 'id', 'unitCode'])
ca_stations = ca_stations.rename(columns = {'value': 'altitude_m', '@type': 'type'})
ca_stations.geometry = ca_stations.geometry.apply(lambda x: Point(x['coordinates'][0], x['coordinates'][1]))
#extract just the station names
station_list = list(ca_stations.stationIdentifier)

#download year of data for all stations, save as CSV.
for station in station_list:
    resp = get_meso_ts('201801010000', '201812312359', station, mesonet_token)
    print(resp[1].url)
    parse_meso(resp, './data/')
