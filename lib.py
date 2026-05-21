"""
Funcoes compartilhadas: baixar dados, LSTM, treinar e avaliar.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "LSTMPrevisao",
    "baixar_acao",
    "calcular_metricas",
    "criar_sequencias",
    "dividir_treino_teste",
    "imprimir_metricas",
    "normalizar",
    "prever",
    "treinar",
]


# ---------------------------------------------------------------------------
# 1) Dados
# ---------------------------------------------------------------------------

def baixar_acao(ticker: str, inicio: str, fim: Optional[str]) -> pd.DataFrame:
    """Baixa High, Low, Close, Volume no Yahoo Finance."""
    raw = yf.download(ticker, start=inicio, end=fim, progress=False, auto_adjust=True)
    if raw is None:
        raise ValueError(f"Sem dados para {ticker}")
    df = raw if isinstance(raw, pd.DataFrame) else raw.to_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"Sem dados para {ticker}")
    out = df[["High", "Low", "Close", "Volume"]].dropna()
    out["Volume"] = np.log1p(out["Volume"].clip(lower=0))
    return cast(pd.DataFrame, out)


def criar_sequencias(df: pd.DataFrame, janela: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Para cada dia, usa `janela` dias anteriores para prever o Close desse dia.
    Retorna X [amostras, janela, 4], y [amostras, 1] e lista de datas.
    """
    valores = df.values.astype(np.float32)
    datas = []
    X, y = [], []
    for i in range(len(valores) - janela):
        X.append(valores[i : i + janela])
        y.append(valores[i + janela, 2])  # coluna Close
        datas.append(str(df.index[i + janela])[:10])
    return np.stack(X), np.array(y).reshape(-1, 1), datas


def dividir_treino_teste(X, y, datas, parte_treino: float):
    """Divisao cronologica (sem embaralhar)."""
    corte = int(len(X) * parte_treino)
    return (
        X[:corte], y[:corte], datas[:corte],
        X[corte:], y[corte:], datas[corte:],
    )


def normalizar(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_outros: np.ndarray,
    y_outros: Optional[np.ndarray] = None,
):
    """StandardScaler: fit so no treino."""
    n_feat = int(X_treino.shape[2])
    sx = StandardScaler().fit(X_treino.reshape(-1, n_feat))
    sy = StandardScaler().fit(y_treino)

    def tx(X: np.ndarray) -> np.ndarray:
        a = X.reshape(-1, n_feat)
        scaled = np.asarray(sx.transform(a), dtype=np.float32)
        return scaled.reshape(X.shape)

    def ty(y: np.ndarray) -> np.ndarray:
        return np.asarray(sy.transform(y), dtype=np.float32)

    def inversa_y(y_norm):
        return sy.inverse_transform(y_norm)

    X_t = tx(X_treino)
    X_o = tx(X_outros)
    y_t = ty(y_treino)
    y_o = ty(y_outros) if y_outros is not None else None
    return X_t, y_t, X_o, y_o, sx, sy, inversa_y


# ---------------------------------------------------------------------------
# 2) Modelo LSTM
# ---------------------------------------------------------------------------

class LSTMPrevisao(nn.Module):
    def __init__(self, entradas: int = 4, neuronios: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(entradas, neuronios, num_layers=2, batch_first=True)
        self.linear = nn.Linear(neuronios, 1)

    def forward(self, x):
        saida, _ = self.lstm(x)
        return self.linear(saida[:, -1, :])


def treinar(modelo, loader, epocas, lr, device):
    modelo.train()
    otim = torch.optim.Adam(modelo.parameters(), lr=lr)
    perda_fn = nn.MSELoss()
    historico = []
    for ep in range(epocas):
        soma = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            otim.zero_grad()
            pred = modelo(xb)
            loss = perda_fn(pred, yb)
            loss.backward()
            otim.step()
            soma += loss.item()
        media = soma / len(loader)
        historico.append(media)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoca {ep + 1:3d}/{epocas}  perda={media:.4f}")
    return historico


def prever(modelo, X, device):
    modelo.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=device)
        return modelo(t).cpu().numpy()


# ---------------------------------------------------------------------------
# 3) Metricas
# ---------------------------------------------------------------------------

def calcular_metricas(y_real: np.ndarray, y_previsto: np.ndarray) -> dict:
    y_real = y_real.flatten()
    y_previsto = y_previsto.flatten()
    mae = float(np.mean(np.abs(y_real - y_previsto)))
    rmse = float(np.sqrt(np.mean((y_real - y_previsto) ** 2)))
    mask = y_real != 0
    mape = float(np.mean(np.abs((y_real[mask] - y_previsto[mask]) / y_real[mask])) * 100)
    acuracia = max(0.0, 100.0 - mape)
    return {"mae": mae, "rmse": rmse, "mape": mape, "acuracia": acuracia}


def imprimir_metricas(titulo: str, m: dict) -> None:
    print(f"\n--- {titulo} ---")
    print(f"  MAE:      R$ {m['mae']:.4f}")
    print(f"  RMSE:     R$ {m['rmse']:.4f}")
    print(f"  MAPE:     {m['mape']:.2f}%  (erro; quanto menor, melhor)")
    print(f"  Acuracia: {m['acuracia']:.2f}%  (meta > 85%)")
