#! /usr/bin/env python3
import lnetatmo
import os

# 1 : Authenticate
# To get ID go to https://dev.netatmo.com/apps/
clientId = os.getenv("CLIENT_ID")
clientSecret = os.getenv("CLIENT_SECRET")
refreshToken = os.getenv("REFRESH_TOKEN")

authorization = lnetatmo.ClientAuth(
    clientId=clientId, clientSecret=clientSecret, refreshToken=refreshToken
)

# 2 : Get devices list
weatherData = lnetatmo.WeatherStationData(authorization)

# 3 : Access most fresh data directly
lastData = weatherData.lastData()

print(
    " Temperature (inside/outside): {} / {} °C\n".format(
        lastData["Indoor"]["Temperature"], lastData["Indoor"]["Temperature"]
    ),
    "Humidity (inside/outside): {} / {} %\n".format(
        lastData["Indoor"]["Humidity"], lastData["Indoor"]["Humidity"]
    ),
    "CO2: {}ppm\n".format(lastData["Indoor"]["CO2"]),
    "Noise Indoor: {}dB".format(lastData["Indoor"]["Noise"]),
)
