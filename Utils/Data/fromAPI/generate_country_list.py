import json
import pprint
import os

import pandas as pd

pp = pprint.PrettyPrinter(indent=4).pprint


with open('json/country_info.json', 'r', encoding='utf8') as f:
    countries_data = json.load(f)

result = [{
    "name": country_data["name"],
    "id": country_data["id"],
    "region": country_data["region"]["value"],
    "adminregion": country_data["adminregion"]["value"],
    "incomeLevel": country_data["incomeLevel"]["value"],
    "lendingType": country_data["lendingType"]["value"],
    "capitalCity": country_data["capitalCity"],
} for country_data in countries_data
          if country_data["capitalCity"] != ""]

############
# pp(result)
pp(len(result))

df_countries = pd.DataFrame.from_dict(result)

print(df_countries.head())

df_countries.to_csv("csv/countries.csv", index=False)

############

directory = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(directory, "json", "country_list.json")
with open(filepath, "w", encoding='utf8') as outfile:
    json.dump(result, outfile, indent=4, ensure_ascii=False)