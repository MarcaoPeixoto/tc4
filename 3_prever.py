"""
PASSO 3 — Usar o modelo TREINADO em dias RECENTES (dados novos do Yahoo).

Execute:  python 3_prever.py
(Requer: python 2_treinar.py antes)
"""

from datetime import datetime, timedelta
from pathlib import Path


import numpy as np
import pandas as pd
import torch

from config import (
    ARQUIVO_MODELO,
    JANELA_DIAS,
    PREVER_FIM,
    PREVER_INICIO,
    PREVER_ULTIMOS_DIAS,
    TICKER,
)
from lib import (
    LSTMPrevisao,
    baixar_acao,
    calcular_metricas,
    criar_sequencias,
    imprimir_metricas,
    prever,
)


def _data_str(dias: int, base: str) -> str:
    """Subtrai dias de uma data base e retorna YYYY-MM-DD."""
    return (datetime.strptime(base, "%Y-%m-%d") - timedelta(days=dias)).strftime(
        "%Y-%m-%d"
    )


def resolver_periodo_previsao() -> tuple[str, str]:
    """Define inicio/fim: ultimos N dias ou datas fixas em config."""
    fim = PREVER_FIM or datetime.today().strftime("%Y-%m-%d")
    if PREVER_INICIO:
        inicio = PREVER_INICIO
    else:
        inicio = _data_str(PREVER_ULTIMOS_DIAS + 90, fim)
    return inicio, fim


def inicio_avaliacao(fim: str) -> str:
    """Primeiro dia da tabela (ultimos N dias uteis)."""
    if PREVER_INICIO:
        return PREVER_INICIO
    return _data_str(PREVER_ULTIMOS_DIAS, fim)


print("=" * 55)
print(" PASSO 3: PREVISAO NOS DIAS MAIS RECENTES")
print("=" * 55)

if not Path(ARQUIVO_MODELO).exists():
    raise SystemExit("Rode antes: python 2_treinar.py")

try:
    pacote = torch.load(ARQUIVO_MODELO, map_location="cpu", weights_only=False)
except TypeError:
    pacote = torch.load(ARQUIVO_MODELO, map_location="cpu")

scaler_x = pacote["scaler_x"]
scaler_y = pacote["scaler_y"]
janela = pacote["janela"]
neuronios = pacote["neuronios"]

busca_inicio, busca_fim = resolver_periodo_previsao()
dia_corte = inicio_avaliacao(busca_fim)

print(f" Ticker: {TICKER}")
print(f" Baixando: {busca_inicio} ate {busca_fim}")
print(f" Tabela (ultimos {PREVER_ULTIMOS_DIAS} dias): desde {dia_corte}")
print()

# +120 dias antes do corte para montar janelas de 60 dias
buffer_inicio = _data_str(120, dia_corte)
df = baixar_acao(TICKER, buffer_inicio, busca_fim)
X, y, datas = criar_sequencias(df, janela)

idx = [i for i, d in enumerate(datas) if d >= dia_corte]
if not idx:
    raise SystemExit(
        f"Sem dados desde {dia_corte}. Verifique internet ou PREVER_ULTIMOS_DIAS em config.py."
    )

X_n = X[idx]
y_n = y[idx]
datas_n = [datas[i] for i in idx]

n_feat = X_n.shape[2]
X_flat = scaler_x.transform(X_n.reshape(-1, n_feat)).reshape(X_n.shape).astype(np.float32)

device = torch.device("cpu")
modelo = LSTMPrevisao(neuronios=neuronios)
modelo.load_state_dict(pacote["modelo"])
modelo.eval()

pred_norm = prever(modelo, X_flat, device)
pred = scaler_y.inverse_transform(pred_norm)
real = y_n

metricas_novos = calcular_metricas(real, pred)

mt = pacote.get("metricas_treino", {})
print("--- Metricas do TREINO (arquivo modelo.pt) ---")
if mt:
    print(
        f"  MAE: R$ {mt.get('mae', 0):.4f}  |  MAPE: {mt.get('mape', 0):.2f}%  "
        f"|  Acuracia: {mt.get('acuracia', 0):.2f}%"
    )

imprimir_metricas(f"DADOS RECENTES ({len(datas_n)} dias)", metricas_novos)

# Ultimos 10 dias em destaque
ultimos = min(10, len(datas_n))
print(f"\n  >>> ULTIMOS {ultimos} DIAS (para apresentacao) <<<")
print("  Data          Real (R$)   Previsto (R$)   Erro %")
print("  " + "-" * 48)
for d, r, p in zip(datas_n[-ultimos:], real[-ultimos:].flatten(), pred[-ultimos:].flatten()):
    erro = abs(r - p) / r * 100 if r else 0
    print(f"  {d:<12} {r:>10.4f}   {p:>12.4f}   {erro:>6.2f}%")

if len(datas_n) > ultimos:
    print(f"\n  (... + {len(datas_n) - ultimos} dias anteriores no periodo)")

print(f"\n  Ultimo pregão: {datas_n[-1]}")
print(f"  Fechamento real: R$ {real[-1, 0]:.4f}")
print(f"  Previsao proximo dia: R$ {float(pred.flatten()[-1]):.4f}")
erro_perc = abs(real[-1, 0] - float(pred.flatten()[-1])) / real[-1, 0] * 100 if real[-1, 0] else 0
print(f"  Erro percentual: {erro_perc:.2f}%")
print("=" * 55)
