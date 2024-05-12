import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Download and prepare the dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
filepath = os.path.join(os.path.dirname(__file__), "household_power_consumption.zip")

df = pd.read_csv(
    filepath,
    compression="zip",
    sep=";",
    parse_dates={"dt": ["Date", "Time"]},
    infer_datetime_format=True,
    low_memory=False,
    na_values=["nan", "?"],
    index_col="dt",
)

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Use Global Active Power as an example target variable
df_daily = df.resample("D").sum()
data = df_daily["Global_active_power"].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()


# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i : (i + look_back)]
        X.append(a)
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)


look_back = 10
X, Y = create_dataset(data_normalized, look_back)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


model = LSTMModel(input_size=look_back, hidden_size=50, num_layers=1, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
train_loader = DataLoader(
    TensorDataset(X_train_tensor, Y_train_tensor), batch_size=64, shuffle=True
)

model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
predicted = model(X_test_tensor)
predicted = scaler.inverse_transform(predicted.detach().numpy())
Y_test = scaler.inverse_transform(Y_test_tensor.detach().numpy())

test_accuracy = 100 - np.sqrt(mean_squared_error(Y_test, predicted))

print(f"Test Accuracy: {test_accuracy:.2f}")

# Plotting one of the predictions
plt.plot(Y_test[:100], label="Actual Data")
plt.plot(predicted[:100], label="Predicted Data")
plt.legend()
plt.show()
