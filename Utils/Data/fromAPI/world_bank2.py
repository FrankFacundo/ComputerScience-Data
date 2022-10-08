from pprint import pprint
import requests
import csv
import pandas as pd

non_dac_aid = {
    'CHN': 38000000000,
    'ARE': 4390000000,
    'TUR': 3910000000,
    'QAT': 2000000000,
    'IND': 1600000000,
    'RUS': 1140000000,
    'ISR': 210000000,
    'HUN': 210000000,
    'HRV': 50000000,
    'LTU': 40000000,
    'EST': 30000000,
    'LVA': 20000000,
    'MLT': 10000000,
    'BRA': 793250000,
    'CHL': 41790000
}

country_prefs = pd.read_csv('cpref.csv',
                            header=None,
                            index_col=0,
                            squeeze=True).to_dict()

del country_prefs[1]
country_prefs[1] = country_prefs[2]
del country_prefs[2]
#print(dic)


class Data:
    """
    Aggregate:
    - country
    - region
    - adminregion
    - incomeLevel
    - lendingType
    
    Information about Indicators: 
    - https://datahelpdesk.worldbank.org/knowledgebase/articles/201175-how-does-the-world-bank-code-its-indicators
    - https://web.worldbank.org/archive/website00564A/WEB/DOC/CATALOG_.DOC
    - https://databank.worldbank.org/data/download/site-content/WDI_CETS.xls
    
    TLDR:
    Topic (2 digits)
    General Subject (3 digits)
    Specific Subject (4 digits)
    Extensions (2 digits each)

    For example: 
    DT.DIS.PRVT.CD would read 
    "External debt disbursements by private creditors in current US dollars."
    """
    def __init__(self, country, year_index):
        self.country = country
        self.year_index = year_index

    def GDPPerCapPPP(self):
        data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/NY.GDP.PCAP.PP.CD?format=json'
            .format(self.country))
        ret_val = data.json()
        return ret_val[1][self.year_index]['value']

    def Unemployment(self):
        data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/SL.UEM.TOTL.ZS?format=json'
            .format(self.country))
        ret_val = data.json()
        return ret_val[1][self.year_index]['value']

    def GDPGrowthRate(self):
        data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/NY.GDP.MKTP.KD.ZG?format=json'
            .format(self.country))
        ret_val = data.json()
        return ret_val[1][self.year_index['value']]

    def PovertyRate(self):
        data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/SI.POV.DDAY?format=json'
            .format(self.country))
        ret_val = data.json()
        return ret_val[1][self.year_index]['value']

    def PopDensity(self):
        data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/EN.POP.DNST?format=json'
            .format(self.country))
        ret_val = data.json()
        return ret_val[1][self.year_index]['value']

    def NativeRefugeePercent(self):
        refugee_data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/SM.POP.REFG.OR?format=json'
            .format(self.country))
        population_data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/SP.POP.TOTL?format=json'
            .format(self.country))
        refugee_ret_val = refugee_data.json()[1][self.year_index]['value']
        population_ret_val = population_data.json()[1][
            self.year_index]['value']
        return refugee_ret_val / population_ret_val * 100

    def NativesPerRefugee(self):
        refugee_data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/SM.POP.REFG?format=json'
            .format(self.country))
        population_data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/SP.POP.TOTL?format=json'
            .format(self.country))
        refugee_ret_val = refugee_data.json()[1][self.year_index]['value']
        population_ret_val = population_data.json()[1][
            self.year_index]['value']
        return population_ret_val // refugee_ret_val

    def HumanitarianAid(self):
        OECD_data = requests.get(
            'http://api.worldbank.org/v2/country/{}/indicator/DC.ODA.TOTL.CD?format=json'
            .format(self.country))
        ret_val = OECD_data.json()[1][self.year_index]['value']
        if ret_val == None:
            if self.country in non_dac_aid:
                return non_dac_aid[self.country]
        return ret_val


#Need to include method for case where no data is available... preferably to output last available data

brazil = Data('BRA', 1)
print(brazil.GDPPerCapPPP())


def name_n_year(country, year):
    try:
        prefix = country_prefs[1][country]
    except KeyError as error:
        print(
            'Check the spelling of your country. Country name needs to be capitalized and as per the World Bank naming format:'
        )
        print(
            'https://wits.worldbank.org/wits/wits/witshelp/content/codes/country_codes.htm'
        )
    else:
        year_index = -year + 2018
        return prefix, year_index


# print(name_n_year('Brazil', 2019))
'''
OECD_data = requests.get('http://api.worldbank.org/v2/country/BRA/indicator/DC.ODA.TOTL.CD?format=json')
print(OECD_data.json()[1][1])
'''