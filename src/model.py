# src/model.py
"""
Tiga arsitektur model time series:
  1. LSTMModel  — baseline klasik
  2. GRUModel   — baseline ringan
  3. TFTModel   — main model (Temporal Fusion Transformer, simplified)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT, PRED_LEN, MODEL_DIR


# ─────────────────────────────────────────────────────────────────────────
# 1. LSTM
# ─────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, output_size: int = PRED_LEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ─────────────────────────────────────────────────────────────────────────
# 2. GRU
# ─────────────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, output_size: int = PRED_LEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.gru     = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ─────────────────────────────────────────────────────────────────────────
# 3. Temporal Fusion Transformer (simplified)
# ─────────────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """GRN — komponen inti TFT."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, output_size)
        self.gate    = nn.Linear(hidden_size, output_size)
        self.norm    = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.skip    = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h        = torch.relu(self.fc1(x))
        h        = self.dropout(h)
        output   = self.fc2(h)
        gate     = torch.sigmoid(self.gate(h))
        gated    = gate * output
        residual = self.skip(x)
        return self.norm(gated + residual)


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer (simplified).
    Referensi: Lim et al. 2021 — https://arxiv.org/abs/1912.09363
    """
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_heads: int = 4, num_layers: int = 2,
                 output_size: int = PRED_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Variable selection network (per timestep)
        self.vsn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # LSTM encoder
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Positionwise feed-forward (GRN)
        self.ffn = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)               # → (B, T, H)
        x = self.vsn(x)                      # variable selection
        enc_out, _ = self.encoder(x)         # LSTM encode
        attn_out, _ = self.attn(enc_out, enc_out, enc_out)
        attn_out = self.attn_norm(attn_out + enc_out)   # residual
        out = self.ffn(attn_out)
        out = self.dropout(out[:, -1, :])    # ambil timestep terakhir
        return self.output_proj(out)         # → (B, PRED_LEN)


# ─────────────────────────────────────────────────────────────────────────
# Training & evaluasi
# ─────────────────────────────────────────────────────────────────────────

def get_model(name: str, input_size: int) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMModel(input_size)
    elif name == "gru":
        return GRUModel(input_size)
    elif name == "tft":
        return TFTModel(input_size)
    else:
        raise ValueError(f"Model '{name}' tidak dikenal. Pilih: lstm, gru, tft")


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, lr: float = 0.001, batch_size: int = 32,
                model_name: str = "model") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"► Training {model_name.upper()} di {device}...")
    model.to(device)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion   = nn.MSELoss()
    history     = {"train_loss": [], "val_loss": []}
    best_val    = float("inf")
    patience_cnt = 0
    PATIENCE    = 20

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}_best.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  ⏹ Early stopping di epoch {epoch}")
                break

    print(f"  ✓ Best val loss: {best_val:.5f}")
    return history


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                   scaler, target_idx: int = 0) -> dict:
    """Evaluasi dengan MAE, RMSE, MAPE dalam skala asli (Rupiah)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_t  = torch.FloatTensor(X_test).to(device)
        pred = model(X_t).cpu().numpy()

    # Inverse transform — hanya kolom target
    n_features = scaler.n_features_in_

    def inv(arr):
        dummy = np.zeros((arr.shape[0], n_features))
        dummy[:, target_idx] = arr[:, 0]
        return scaler.inverse_transform(dummy)[:, target_idx]

    pred_inv = np.array([inv(pred[:, i:i+1]) for i in range(pred.shape[1])]).T
    true_inv = np.array([inv(y_test[:, i:i+1]) for i in range(y_test.shape[1])]).T

    mae  = np.mean(np.abs(pred_inv - true_inv))
    rmse = np.sqrt(np.mean((pred_inv - true_inv) ** 2))
    mape = np.mean(np.abs((pred_inv - true_inv) / (true_inv + 1e-8))) * 100

    print(f"  MAE  : Rp {mae:,.0f}")
    print(f"  RMSE : Rp {rmse:,.0f}")
    print(f"  MAPE : {mape:.2f}%")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape,
            "pred": pred_inv, "true": true_inv}
