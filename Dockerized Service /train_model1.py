"""
To train this model, in your terminal:
> python train_model1.py
"""

from sklearn.externals import joblib
from zipfile import ZipFile
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from math import ceil

zip_file = ZipFile('cycle-share-dataset.zip')

weather_data = pd.read_csv(zip_file.open('weather.csv'))

trip_data = pd.read_csv(zip_file.open('trip.csv'), skiprows=50793)

station_data = pd.read_csv(zip_file.open('station.csv'))

weather_data['year'] = pd.DatetimeIndex(weather_data['Date']).year

weather_data['month'] = pd.DatetimeIndex(weather_data['Date']).month

weather_data['day'] = pd.DatetimeIndex(weather_data['Date']).day


# Define a function to map the values 
def set_value(row_number, assigned_value): 
    return assigned_value[row_number]

season_dic ={3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:4, 1:4, 2:4}

weather_data['season'] = weather_data['month'].apply(set_value, args =(season_dic, ))

trip_data.drop(trip_data.loc[:, 'tripduration':'birthyear'].columns, axis = 1, inplace=True)

trip_data.columns = ['trip_id','starttime', 'stoptime', 'bikeid', 'tripduration', 
'from_station_name', 'to_station_name', 'from_station_id', 'to_station_id', 'usertype', 'gender', 'birthyear']

trip_data['Date'] = trip_data['starttime'].astype(str).str[:-6]

trip_data.rename(columns = {'trip_id':'trip_count'}, inplace = True) 

trip_data = pd.merge(trip_data,
                 weather_data,
                 left_on='Date', 
                 right_on='Date',
                 how='left')

trip_data['route'] = trip_data['from_station_id'] + ' ' + 'to' + ' ' + trip_data['to_station_id']

trip_data['starttime'] = trip_data['starttime'].astype('datetime64[ns]')



newTrip = trip_data

col = ['trip_count', 'starttime', 'bikeid', 'tripduration','from_station_id',
       'usertype', 'Date',
       'Max_Temperature_F', 'Mean_Temperature_F', 'Min_TemperatureF',
       'Max_Dew_Point_F', 'MeanDew_Point_F', 'Min_Dewpoint_F', 'Max_Humidity',
       'Mean_Humidity', 'Min_Humidity', 'Max_Sea_Level_Pressure_In',
       'Mean_Sea_Level_Pressure_In', 'Min_Sea_Level_Pressure_In',
       'Max_Visibility_Miles', 'Mean_Visibility_Miles', 'Min_Visibility_Miles',
       'Max_Wind_Speed_MPH', 'Mean_Wind_Speed_MPH', 'Max_Gust_Speed_MPH',
       'Precipitation_In', 'Events', 'year', 'month', 'day', 'season',
       'route']


newTrip['trip_count'] = 1
new_trip = newTrip.groupby(['from_station_id', pd.Grouper(key='starttime', freq='W-MON')])[col].sum().reset_index().sort_values('starttime')
new_trip['month'] = pd.DatetimeIndex(new_trip['starttime']).month
new_trip['day'] = pd.DatetimeIndex(new_trip['day']).day
new_trip['season'] = new_trip['month'].apply(set_value, args =(season_dic, ))

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))

new_trip['week_of_month'] = new_trip['starttime'].apply(week_of_month)
station_dic = {}
keys = range(len(new_trip.from_station_id.unique()))
values = new_trip.from_station_id.unique()
j=0
for i in values:
    station_dic[i] = j
    j = j+1

new_trip['from_station_id'] = new_trip['from_station_id'].apply(set_value, args =(station_dic, ))



new_trip.dropna(inplace=True)

X = new_trip.drop(["starttime", 'trip_count', 'year'], axis=1)

y = new_trip['trip_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=997)


rf = RandomForestRegressor(n_estimators=100)
rf.fit(X = X_train,y = y_train)
y_pred = rf.predict(X= X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)
print('explained variance: ', explained_variance)


joblib.dump(rf, 'model/bike_classifier.joblib')