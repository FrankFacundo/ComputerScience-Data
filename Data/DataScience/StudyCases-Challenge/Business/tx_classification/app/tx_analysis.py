import os

import kaggle

from dask import dataframe as dd


class TxAnalysis:

    def __init__(self):
        return

    @staticmethod
    def load_dataset(dataset_ref: str):
        dataset_path_parent = '../dataset/'
        dataset_path = os.path.join(dataset_path_parent,
                                    "bank_transactions.csv")
        # Ensure the Kaggle API client is properly configured
        kaggle.api.authenticate()
        # Download the specified dataset
        kaggle.api.dataset_download_files(dataset_ref,
                                          path=dataset_path_parent,
                                          unzip=True)

        # Import a csv dataset from dask
        dataset = dd.read_csv(dataset_path)

        return dataset

    @staticmethod
    def prepare_dataset(dataset):
        dataprep = dataset.groupby(
            "CustomerID")["TransactionAmount (INR)"].agg(['mean', 'count'])
        dataprep = dataprep.reset_index()
        dataprep = dataprep.rename(
            columns={
                "TransactionAmount (INR)": "amount",
                "mean": "avg_amount",
                "count": "nb_tx",
            })
        return dataprep

    @staticmethod
    def analyse(dataprep):
        result = dataprep
        return result

    @staticmethod
    def train_model(dataprep):
        model = None
        return model

    def run(self):
        ref_dataset = "shivamb/bank-customer-segmentation"
        dataset = TxAnalysis.load_dataset(dataset_ref=ref_dataset)
        # print(type(dataset))
        # print(dataset.head(compute=True))
        # print(len(dataset))
        
        dataprep = TxAnalysis.prepare_dataset(dataset)
        dataprep = dataprep.sort_values(by=['nb_tx'], ascending=False)
        print(dataprep.head(compute=True))

        # analysis = TxAnalysis.analyse(dataprep)
        # model = TxAnalysis.train_model(dataprep)

        return
