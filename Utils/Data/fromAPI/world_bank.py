import json
import os
from collections import namedtuple
from typing import Final

import pandas as pd
import pprint
import requests

from scale import NumberScale
from structure.tree_structure import Aggregation

pp = pprint.PrettyPrinter(indent=4).pprint


class FormatFile(object):
    JSON = "json"
    JSONSTAT = "jsonstat"
    XML = "xml"


class WorldBankAPI(object):
    """
    Developer information page:
    https://datahelpdesk.worldbank.org/knowledgebase/topics/125589

    
    Information about Indicators: 
    - https://datahelpdesk.worldbank.org/knowledgebase/articles/201175-how-does-the-world-bank-code-its-indicators
    - https://web.worldbank.org/archive/website00564A/WEB/DOC/CATALOG_.DOC
    - https://databank.worldbank.org/data/download/site-content/WDI_CETS.xls
    
    TLDR:
    Topic (2 digits)
    General Subject (3 digits)
    Specific Subject (4 digits)
    Extensions (2 digits each)

    Check WDI_CETS.csv CATALOG.pdf and to check possibilities.
    Other pages are 
    - https://datatopics.worldbank.org/world-development-indicators.
    - https://data.worldbank.org/indicator/

    For example: 
    DT.DIS.PRVT.CD would read 
    "External debt disbursements by private creditors in current US dollars."

    Check comments of class tree_structure to have a little more information.
    """
    RequestArgs = namedtuple('RequestArgs',
                             ['aggregation', 'id', 'indicator', 'format'])
    Indicator = namedtuple(
        'Indicator',
        ['topic', 'general_subject', 'specific_subject', 'extensions'])

    metadata_reponse_index: Final[int] = 0
    data_reponse_index: Final[int] = 1

    def __init__(self, default_format=None, verbose=False):

        self.format = default_format if default_format is not None else FormatFile.JSON
        self.verbose = verbose
        self.country_list = self.get_filtered_country_list()
        self.save_json_by_default = False
        self.save_csv_by_default = False

    def get_request(self,
                    request_args: RequestArgs,
                    optional_args: dict = None):
        request_link: str = "http://api.worldbank.org/v2/{}/{}/indicator/{}?format={}"
        request_link_without_indicator: str = "http://api.worldbank.org/v2/{}/{}/?format={}"

        if request_args.indicator is None:
            request = request_link_without_indicator.format(
                request_args.aggregation, request_args.id, request_args.format)
        else:
            request = request_link.format(request_args.aggregation,
                                          request_args.id,
                                          request_args.indicator,
                                          request_args.format)
        if optional_args:
            for key, value in optional_args.items():
                request = "{}&{}={}".format(request, key, value)
        return request

    def format_indicator(self, indicator: Indicator):
        return "{}.{}.{}.{}".format(indicator.topic, indicator.general_subject,
                                    indicator.specific_subject,
                                    indicator.extensions)

    def request(self, request, is_first_iteration=True):
        if self.verbose:
            print("request : {}".format(request))
        try:
            response = requests.get(request).json()
            response = self.process_response(response)
            if is_first_iteration:
                pages = response[self.metadata_reponse_index]["pages"]
                response = response[self.data_reponse_index]
                for page in range(pages - 1):
                    # Start by page 2
                    page = page + 2
                    request_intermediate = "{}&page={}".format(request, page)
                    response_by_page = self.request((request_intermediate),
                                                    is_first_iteration=False)
                    response = response + response_by_page
                if self.verbose:
                    print("Number of results (element in json) : {}".format(
                        len(response)))
            else:
                response = response[self.data_reponse_index]
            return response

        except Exception as e:
            print(e)
            raise

    def save_json(self, dictionary: dict, filename:str):
        directory = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(directory, "json", filename)

        with open(filepath, "w", encoding='utf8') as outfile:
            json.dump(dictionary, outfile, indent=4, ensure_ascii=False)

    def save_csv(self, dictionary:dict, filename:str):
        dataframe = pd.DataFrame.from_dict(dictionary)
        dataframe.to_csv("csv/{}".format(filename), index=False)

    def process_response(self, response: dict):
        error_message = response[0].get("message")
        if error_message is not None:
            error_message_value = error_message[0].get("value")
            raise Exception(error_message_value)
        else:
            return response

    def change_scale(self, value, number_scale):
        if value is None:
            return None
        if number_scale == NumberScale.MILLION:
            return (value / (10**6))
        if number_scale == NumberScale.BILLION:
            return (value / (10**9))
        if number_scale == NumberScale.TRILLION:
            return (value / (10**12))

    def get_filtered_country_list(self):
        """
        This filter country aggregations as "Europe" or "Africa" for example.
        """
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
        return result

    def country_id_list_for_request(self, countries_id="all"):
        if countries_id == "all":
            country_list_id = [country["id"] for country in self.country_list]
        else:
            country_list_id = countries_id
        country_list_req = ";".join(country_list_id)
        return country_list_req

    def get_country_info(self, country_id="all"):
        """
        To get country id go to file country.json
        """

        request_args = WorldBankAPI.RequestArgs(
            aggregation=Aggregation.COUNTRY,
            id=country_id,
            format=self.format,
            indicator=None)

        request = self.get_request(request_args)
        data = self.request(request)

        if len(data) == 1:
            data = data[0]

        return data

    def get_gdp(self, country_id: str = "all", date: str = None):
        """
        date is a year: Ex. 2021
        """
        topic = "NY"  # National accounts: income
        general_subject = "GDP"  # Gross domestic product
        specific_subject = "MKTP"  # Market
        extensions = "CD"  # Current US$

        if date is not None:
            optional_args = {"date": date}
        else:
            optional_args = None

        indicator = WorldBankAPI.Indicator(topic=topic,
                                           general_subject=general_subject,
                                           specific_subject=specific_subject,
                                           extensions=extensions)

        request_args = WorldBankAPI.RequestArgs(
            aggregation=Aggregation.COUNTRY,
            id=country_id,
            format=self.format,
            indicator=self.format_indicator(indicator))

        request = self.get_request(request_args, optional_args)
        data = self.request(request)

        number_scale = NumberScale.TRILLION

        result = [{
            "country":
            data_point["country"]["value"],
            "indicator":
            data_point["indicator"]["value"],
            "value":
            self.change_scale(value=data_point["value"],
                              number_scale=number_scale),
            "scale":
            number_scale
        } for data_point in data]

        if len(result) == 1:
            result = result[0]

        return result

    def get_life_expectancy(self, country_id="all", year=2020):
        """
        date is a year: Ex. 2021
        """
        topic = "SP"  # Social: population
        general_subject = "DYN"  # Dynamic
        specific_subject = "LE00"  # Life expectancy at birth
        extensions = "IN"  # Interpolated

        if country_id == 'all':
            country_id = self.country_id_list_for_request("all")

        if year is not None:
            optional_args = {"date": year}
        else:
            optional_args = None

        indicator = WorldBankAPI.Indicator(topic=topic,
                                           general_subject=general_subject,
                                           specific_subject=specific_subject,
                                           extensions=extensions)

        request_args = WorldBankAPI.RequestArgs(
            aggregation=Aggregation.COUNTRY,
            id=country_id,
            format=self.format,
            indicator=self.format_indicator(indicator))

        request = self.get_request(request_args, optional_args)
        data = self.request(request)

        result = [{
            "country": data_point["country"]["value"],
            "indicator": data_point["indicator"]["value"],
            "value": data_point["value"],
            "scale": "years",
            "year": data_point["date"],
        } for data_point in data]

        if len(result) == 1:
            result = result[0]

        if self.save_json_by_default: self.save_json(result, "life_expectancy_{}.json".format(year))
        if self.save_csv_by_default: self.save_csv(result, "life_expectancy_{}.csv".format(year))

        return result


api_client = WorldBankAPI(verbose=True)
api_client.save_json_by_default = True
api_client.save_csv_by_default = True
# response = api_client.get_country_info(country_id="ALL")
# response = api_client.get_gdp(country_id="all", date=2019)
# api_client.save_json(response, "country_info.json")

response = api_client.get_life_expectancy(year=2020)
print(response)
# api_client.save_json(response, "life_expectancy.json")