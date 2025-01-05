import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -------------------------------
# 1. Define a TimeSeriesDataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, weight_col, seq_length=16):
        """
        df           : DataFrame containing chronological data (sorted by time)
        feature_cols : columns used as input features
        target_col   : column to predict
        weight_col   : column for weights
        seq_length   : how many timesteps to include in each input sequence
        """
        self.df = df.reset_index(drop=True)
        self.features = df[feature_cols].values
        self.targets = df[target_col].values
        self.weights = df[weight_col].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.df) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        weight = self.weights[idx + self.seq_length - 1]

        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        weight = torch.tensor(weight, dtype=torch.float32)

        return x_seq, y, weight


# -------------------------------
# 2. Define an LSTM Model
# -------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


# -------------------------------
# 3. Load Data and Use the Model
# -------------------------------
if __name__ == "__main__":
    # Load the dataset from the CSV file
    csv_file_path = "combined_data.csv"
    df = pd.read_csv(csv_file_path)

    # Define features, target, and weight columns
    feature_cols = [col for col in df.columns if col.startswith("feature")]
    target_col = "responder_6"
    weight_col = "weight"
    seq_length = df.shape[0] - 1

    # Sort data chronologically (important for time-series)
    df = df.sort_values(["date_id", "time_id"]).reset_index(drop=True)

    # Create the dataset and dataloader
    dataset = TimeSeriesDataset(df, feature_cols, target_col, weight_col, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate model
    model = LSTMRegressor(input_size=len(feature_cols), hidden_size=64)
    criterion = nn.MSELoss(reduction="none")  # Using reduction="none" to apply weights
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(4):
        total_loss = 0
        for x_batch, y_batch, w_batch in dataloader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            weighted_loss = (loss * w_batch).mean()  # Apply weights to the loss
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item() * x_batch.size(0)
        print(f"Epoch {epoch+1}, Weighted Loss: {total_loss / len(dataset):.4f}")
