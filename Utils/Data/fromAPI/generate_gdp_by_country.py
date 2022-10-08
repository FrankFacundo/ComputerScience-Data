import json
import pprint

import pandas as pd
from world_bank import WorldBankAPI

pp = pprint.PrettyPrinter(indent=4).pprint

##############################################
##############################################
##############################################
year = 2019
##############################################
##############################################
##############################################

with open('json/country_list.json', 'r', encoding='utf8') as f:
    countries_data = json.load(f)

api_client = WorldBankAPI(verbose=True)

gdp = []
for country_data in countries_data:
    print('###############')
    country_id = country_data["id"]
    print(country_id)
    response = api_client.get_gdp(country_id=country_id, date=year)
    gdp = gdp + response
    print(response)
    print('###############')
print('\n**********************')
print(gdp)

df_gdp = pd.DataFrame.from_dict(gdp)

print(df_gdp.head())

df_gdp.to_csv("csv/gdp_{}.csv".format(year), index=False)