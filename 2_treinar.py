"""
PASSO 2 — Treinar a LSTM e mostrar resultados no conjunto de TESTE (20% final).

Execute:  python 2_treinar.py
"""

from config import ARQUIVO_MODELO, TICKER
from lib import imprimir_metricas
from servico_previsao import executar_treino

print(f" Ticker: {TICKER}")
metricas = executar_treino(log=True)
imprimir_metricas("Resultados no TESTE do treino (20% final)", metricas)
print(f"\n Modelo salvo em: {ARQUIVO_MODELO}")
print(" Proximo passo: python 3_prever.py  ou  python api.py")
