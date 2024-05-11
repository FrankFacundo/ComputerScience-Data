import pandas as pd
from data.custom_dataset import TFTDataset
from data_formatters.electricity import ElectricityFormatter

electricity = pd.read_csv("data/electricity.csv", index_col=0)
data_formatter = ElectricityFormatter()
train, valid, test = data_formatter.split_data(electricity)

train_dataset = TFTDataset(train)

point = train_dataset[0]
print(train_dataset[0])
