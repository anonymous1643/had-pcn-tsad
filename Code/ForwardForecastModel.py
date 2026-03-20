import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, forecast_horizon):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size : idx + self.window_size + self.forecast_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- Hybrid Model with GRU + Dropout + Raw logits ---
class HybridForecastModel(nn.Module):
    def __init__(self, input_size, discrete_indices, tcn_channels=[64, 64],
                 hidden_dim=128, proj_dim=64, heads=2, gru_hidden=64,
                 forecast_horizon=1, dropout_prob=0.3):
        super().__init__()
        from torch.nn.utils import weight_norm

        self.input_size = input_size
        self.discrete_indices = torch.tensor(discrete_indices, dtype=torch.long)
        self.continuous_indices = torch.tensor(
            [i for i in range(input_size) if i not in discrete_indices],
            dtype=torch.long
        )
        self.forecast_horizon = forecast_horizon

        # TCN Layers
        self.tcn = nn.Sequential(*[
            layer for i in range(len(tcn_channels))
            for layer in (
                weight_norm(nn.Conv1d(
                    in_channels=input_size if i == 0 else tcn_channels[i - 1],
                    out_channels=tcn_channels[i],
                    kernel_size=3,
                    padding=(3 - 1) * (2 ** i),
                    dilation=2 ** i
                )),
                nn.ReLU()
            )
        ])

        # GRU + Transformer Encoder
        self.gru = nn.GRU(input_size=tcn_channels[-1], hidden_size=gru_hidden, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=gru_hidden, nhead=heads, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.dropout = nn.Dropout(dropout_prob)
        self.projector = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        # Output heads
        self.continuous_head = nn.Linear(gru_hidden, forecast_horizon * len(self.continuous_indices))
        self.discrete_head = nn.Linear(gru_hidden, forecast_horizon * len(self.discrete_indices))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, D, T]
        feat = self.tcn(x)      # [B, C, T]
        feat = feat.permute(0, 2, 1)  # [B, T, C]

        gru_out, _ = self.gru(feat)  # [B, T, H]
        encoded = self.transformer(gru_out.permute(1, 0, 2)).mean(dim=0)  # [B, H]
        encoded = self.dropout(encoded)

        embed = self.projector(encoded)

        # Prediction
        cont_pred = self.continuous_head(encoded).view(-1, self.forecast_horizon, len(self.continuous_indices))
        disc_pred = self.discrete_head(encoded).view(-1, self.forecast_horizon, len(self.discrete_indices))

        # Merge predictions into full shape
        full_pred = torch.zeros((x.size(0), self.forecast_horizon, self.input_size), device=x.device)
        full_pred[:, :, self.continuous_indices] = cont_pred
        full_pred[:, :, self.discrete_indices] = disc_pred  # raw logits (sigmoid later)

        return full_pred, embed



# --- Mixed Loss ---
def hybrid_loss(pred, target, discrete_indices, alpha=1.0, beta=1.0):
    all_indices = list(range(pred.size(-1)))
    continuous_indices = [i for i in all_indices if i not in discrete_indices]

    loss = 0.0

    if continuous_indices:
        cont_loss = nn.HuberLoss()(pred[:, :, continuous_indices], target[:, :, continuous_indices])
        loss += alpha * cont_loss

    if discrete_indices:
        disc_pred = pred[:, :, discrete_indices]
        disc_target = torch.clamp(target[:, :, discrete_indices], 0, 1)
        disc_loss = nn.BCEWithLogitsLoss()(disc_pred, disc_target)
        loss += beta * disc_loss

    return loss


# --- Train ---
def train_model(model, loader, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            loss = hybrid_loss(pred, y, model.discrete_indices.tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# --- Predict ---
def collect_predicted_normals(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred, _ = model(x)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)  # shape: [N, H, D]

# --- Load Data ---
train_df = pd.read_csv("Data/MSL_train.csv", index_col=0)
test_df = pd.read_csv("Data/MSL_test.csv", index_col=0)

drop_cols = [col for col in ["timestamp", "attack"] if col in train_df.columns]
train_data = train_df.drop(columns=drop_cols, errors="ignore").values
test_data = test_df.drop(columns=drop_cols, errors="ignore").values
labels = test_df["attack"].values

# --- Feature split ---
discrete_indices = []
continuous_indices = []

threshold = 10  # max unique values for a discrete feature

for i in range(train_data.shape[1]):
    num_unique = len(np.unique(train_data[:, i]))
    if num_unique <= threshold:
        discrete_indices.append(i)
    else:
        continuous_indices.append(i)

discrete_indices = np.array(discrete_indices)
continuous_indices = np.array(continuous_indices)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# --- Config ---
window_size = 5
forecast_horizon = 1
input_size = train_data.shape[1]
batch_size = 32

g = torch.Generator()
g.manual_seed(42)

# --- Datasets ---
train_dataset = TimeSeriesDataset(train_data, window_size, forecast_horizon)
test_dataset = TimeSeriesDataset(test_data, window_size, forecast_horizon)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)

# --- Model ---
model = HybridForecastModel(input_size=input_size, discrete_indices=discrete_indices.tolist(), forecast_horizon=forecast_horizon).to(device)
train_model(model, train_loader, epochs=5)
