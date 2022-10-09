import json
import pprint
import os

import pandas as pd

pp = pprint.PrettyPrinter(indent=4).pprint


def clean_households():
    households = pd.read_csv('households.csv')
    # print(households.head())

    households = households[[
        "Country or area", "Reference date (dd/mm/yyyy)",
        "Average household size (number of members)"
    ]]
    households.rename(columns={
        'Country or area': 'Country',
        "Reference date (dd/mm/yyyy)": "date",
        "Average household size (number of members)": "household"
    },
                      inplace=True)

    # print(households.head())

    households['date'] = pd.to_datetime(households['date'], format='%m/%d/%Y')
    # households['transacion_date'] = pd.to_datetime(households['transacion_date'])
    # print(households.head())
    # print(len(households))

    households['household']
    households['household'] = households['household'].replace('..', 'NaN')
    households['household'] = households['household'].astype(float)
    households.dropna(inplace=True)

    idx = households.groupby('Country')["date"].transform(
        max) == households["date"]
    households = households[idx].drop_duplicates()

    households = households.groupby(['Country', 'date'
                                     ])["household"].mean().reset_index()
    households.drop(["date"], axis=1, inplace=True)

    print(households.head(10))
    print(len(households))
    # print(len(households["Country"].unique()))
    # print(households.dtypes)
    households.to_csv("households_clean.csv", index=False)


print("****************")
households = pd.read_csv('households_clean.csv')
print(households.head())

print("****************")
population = pd.read_csv('../csv/population_2020.csv')
# print(population.head())
population = population[["country", "value", "region"]]
# print(population.head())
print("****************")
hh_enriched = pd.merge(households,
                       population,
                       how='inner',
                       left_on=['Country'],
                       right_on=['country'])

hh_enriched = hh_enriched.drop("country", axis=1)
hh_enriched.rename(columns={
    'Country': 'country',
    "value": "population in millions",
    "household": "people per household",
},
                   inplace=True)

hh_enriched["total households in millions"] = hh_enriched["population in millions"] / hh_enriched["people per household"]


print(hh_enriched.head(10))
print(len(hh_enriched))

hh_enriched.to_csv("households_enriched.csv", index=False)