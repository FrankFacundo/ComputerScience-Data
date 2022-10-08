#! /usr/bin/env python3
import lnetatmo
import os

# 1 : Authenticate
# To get ID go to https://dev.netatmo.com/apps/
clientId=os.getenv('CLIENT_ID')
clientSecret=os.getenv('CLIENT_SECRET')
username=os.getenv('EMAIL')
password=os.getenv('PW_NETATMO')
scope="read_station read_homecoach"

authorization = lnetatmo.ClientAuth(clientId=clientId,
                               clientSecret=clientSecret,
                               username=username,
                               password=password,
                               scope=scope)

# 2 : Get devices list
weatherData = lnetatmo.WeatherStationData(authorization)

# 3 : Access most fresh data directly
lastData = weatherData.lastData()

print (
  " Temperature (inside/outside): {} / {} °C\n".format
            ( lastData['Indoor']['Temperature'],
              lastData['Outdoor']['Temperature']),
  "Humidity (inside/outside): {} / {} %\n".format(lastData['Indoor']['Humidity'],
              lastData['Outdoor']['Humidity']),
  "CO2: {}ppm\n".format(lastData['Indoor']['CO2']),
  "Noise Indoor: {}dB".format(lastData['Indoor']['CO2']),
)
