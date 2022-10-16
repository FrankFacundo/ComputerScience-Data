"""
"""
import os
import re
from glob import glob


import pandas as pd

class IngestData():
    """
    Class used to ingest data of xls format to database.
    """

    # db = Database()
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir

    def format_xls_to_df(self, filename):
        path = os.path.join(self.base_dir, filename)
        expenses = pd.read_excel(io=path, header=2)
        # print(expenses.head())
        # print(list(expenses.columns))
        return expenses

    def read_extra(self, filename="export_13_10_2022_23_53_30.xls"):
        path = os.path.join(self.base_dir, filename)
        extra = pd.read_excel(io=path, header=0)

        extra = list(extra.columns)
        doc_type = extra[0]
        date = extra[1][len("Solde au "):]
        solde = extra[2]
        return doc_type, date, solde

    def filter_expenses(self, dataframe):
        return dataframe

    def save_extra(self, data):
        # type, date, solde = data
        self.save_on_database()
        return False

    def save_expenses(self, data):
        self.save_on_database()
        return False

    def save_on_database(self):
        return False

    def format_df(self, dataframe):
        # dataframe["Operation day"] = [re.sub(r'[\n\r]*','', str(x)) for x in df['team']]
        new_col = []
        for description_and_date in zip(dataframe['Libelle operation'], dataframe['Date operation']):
            # print(description_and_date)
            date_in_operation_description = re.findall(r"[0-9]{2}/[0-9]{2}", str(description_and_date[0]))
            # print(date_in_operation_description)
            if date_in_operation_description:
                new_value = date_in_operation_description[0]
                new_value = re.sub('/', '-', new_value)
                new_value = new_value + str(description_and_date[1])[-5:]
            else:
                new_value = description_and_date[1]
            
            new_col.append(new_value)
            # print(new_value)
            # print("\n")

        dataframe["Operation day"] = new_col
        dataframe = dataframe.rename(columns={"Date operation": "Date debit operation"})
        cols = dataframe.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        dataframe = dataframe[cols]
        return dataframe

    def extract_data(self, filename):
        expenses = None

        print(filename)
        extra = self.read_extra(filename)
        # print(extra)

        if extra[0].find("Compte") != -1:
            expenses = self.format_xls_to_df(os.path.basename(filename))
            expenses = self.format_df(expenses)
        
        return extra, expenses


    def ingest(self, filename):
        file_path = os.path.join(self.base_dir, filename)
        extra, expenses = self.extract_data(file_path)
        return 

filename = "export_13_10_2022_23_53_30.xls"
ingest_date = IngestData()
extra, expenses = ingest_date.extract_data(filename)
print(expenses.head())
expenses.to_csv("expenses_formatted.csv", index=False)


# exa = "PAIEMENT CB LA BELLE EPOQUE DU 12/10 A VITRY S - CARTE*5047"
# res = re.findall(r"[0-9]{2}/[0-9]{2}", exa)[0]
# print(res)

