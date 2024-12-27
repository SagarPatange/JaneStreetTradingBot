import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Example: read one partition of train data (the competition provides 10 partitions)
train_df = pd.read_parquet("train.parquet")  # e.g., "train_0.parquet"

# Read lags
lags_df = pd.read_parquet("lags.parquet")

# Merge to bring the lagged responders for the *previous* date
# We merge on [date_id, symbol_id], but we have to shift date_id by 1 in lags
lags_df = lags_df.rename(columns={col: col + "_lag" for col in ["responder_0","responder_1","responder_2",
                                                                "responder_3","responder_4","responder_5",
                                                                "responder_6","responder_7","responder_8"]})

# In lags_df, we have something like date_id_lag, symbol_id, responder_0_lag, etc.
# Typically, you'd do "train_df['date_id'] - 1" to match lags. But the exact logic
# may differ if date_id’s are not strictly daily increments.

lags_df["date_id"] = lags_df["date_id"] + 1  # so it lines up with the next day in train_df

# Merge so that for each (date_id, symbol_id), we attach the previous day’s lagged responders
train_df = pd.merge(
    train_df,
    lags_df.drop_duplicates(["date_id", "symbol_id"]),  # drop duplicates if they exist
    on=["date_id", "symbol_id"],
    how="left"
)

# Keep only first 50,000 rows to avoid memory issues in an example
train_df = train_df.head(50000).copy()

# Sort the data by date_id and time_id so sequences are in correct chronological order
train_df = train_df.sort_values(by=["date_id", "time_id"]).reset_index(drop=True)

# Example: filter to a single symbol for demonstration (NOT for production!)
unique_symbols = train_df["symbol_id"].unique()
symbol_to_model = unique_symbols[0]
train_symbol_df = train_df[train_df["symbol_id"] == symbol_to_model].reset_index(drop=True)

# List of feature columns we want to use
feature_cols = [f"feature_{i:02d}" for i in range(79)]
# Optionally add the lagged responders as features
lag_cols = [f"responder_{i}_lag" for i in range(9)]

# Target column
target_col = "responder_6"

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length=16):
        """
        df: DataFrame containing sorted time-series data for a single symbol.
        feature_cols: columns used as input features
        target_col: column to predict
        seq_length: length of look-back window
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length
        
        # Convert relevant columns to numpy arrays for speed
        self.features = df[feature_cols].values
        self.targets = df[target_col].values
        
    def __len__(self):
        # We can only start from seq_length-1
        return len(self.df) - self.seq_length
    
    def __getitem__(self, idx):
        # Sequence of features from idx -> idx+seq_length
        x_seq = self.features[idx : idx + self.seq_length, :]
        # The target is the last row's responder_6
        y = self.targets[idx + self.seq_length - 1]
        
        # Convert to torch tensors
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x_seq, y

train_size = int(len(train_symbol_df) * 0.8)
train_df_ = train_symbol_df.iloc[:train_size].reset_index(drop=True)
val_df_   = train_symbol_df.iloc[train_size:].reset_index(drop=True)

SEQ_LENGTH = 16

train_dataset = TimeSeriesDataset(train_df_, feature_cols + lag_cols, target_col, seq_length=SEQ_LENGTH)
val_dataset   = TimeSeriesDataset(val_df_,   feature_cols + lag_cols, target_col, seq_length=SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, drop_last=True)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        """
        input_size: number of input features per timestep
        hidden_size: number of hidden units in the LSTM
        num_layers: number of stacked LSTM layers
        """
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Final linear layer to map hidden state -> 1 output (responder_6)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        x: (batch, seq_length, input_size)
        """
        # h_0, c_0 default to zeros if not provided
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_length, hidden_size)
        
        # We can take the last timestep's output for regression
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Map to 1 dimension
        out = self.fc(out)   # (batch, 1)
        
        return out.squeeze(-1)  # (batch,)

input_size = len(feature_cols + lag_cols)  # 79 features + 9 lagged = 88
model = LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=1)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

NUM_EPOCHS = 3  # Increase for real training

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            val_preds = model(x_val)
            loss_val = criterion(val_preds, y_val)
            val_loss += loss_val.item() * x_val.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

# Suppose you have a new DataFrame 'test_df' with the same features + lag columns
model.eval()
test_sequences = []  # store predictions

with torch.no_grad():
    # Transform test_df rows into sequences the same way as train
    # For real usage, you’d do it one step at a time or date/time id at a time
    # if the evaluation API calls your script repeatedly.
    
    # Example with a small dataset:
    test_data = TimeSeriesDataset(test_df, feature_cols + lag_cols, target_col, seq_length=SEQ_LENGTH)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    predictions = []
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch)
        predictions.extend(preds.cpu().numpy().tolist())

# 'predictions' now contains the model outputs for each sequence





