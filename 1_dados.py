"""
PASSO 1 — Coletar dados e mostrar como viram sequencias para a LSTM.

Execute:  python 1_dados.py
"""

from config import DATA_FIM, DATA_INICIO, JANELA_DIAS, PARTE_TREINO, TICKER
from lib import baixar_acao, criar_sequencias, dividir_treino_teste

print("=" * 55)
print(" PASSO 1: COLETA DE DADOS")
print("=" * 55)
print(f" Ticker: {TICKER}")
print(f" Periodo: {DATA_INICIO} ate {DATA_FIM}")
print()

df = baixar_acao(TICKER, DATA_INICIO, DATA_FIM)
print(f" Linhas baixadas: {len(df)}")
print(f" Colunas: High, Low, Close, Volume (Volume em log)")
print()
print(" Ultimos 5 dias:")
print(df.tail().to_string())
print()

X, y, datas = criar_sequencias(df, JANELA_DIAS)
X_tr, y_tr, d_tr, X_te, y_te, d_te = dividir_treino_teste(X, y, datas, PARTE_TREINO)

print(f" Janela (SEQ): {JANELA_DIAS} dias por amostra")
print(f" Total de amostras: {len(X)}")
print(f" Treino: {len(X_tr)} amostras  ({d_tr[0]} ... {d_tr[-1]})")
print(f" Teste:  {len(X_te)} amostras  ({d_te[0]} ... {d_te[-1]})")
print(f" Formato X: {X_tr.shape}  -> [amostras, dias, 4 features]")
print(f" Formato y: {y_tr.shape}  -> preco de fechamento (Close)")
print()
print(" Proximo passo: python 2_treinar.py")
