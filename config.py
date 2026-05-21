"""Todas as configuracoes do projeto em um unico arquivo."""

from pathlib import Path

# --- Acao ---
TICKER = "CEAB3.SA"

# --- Datas (treino) ---
DATA_INICIO = "2022-01-01"
DATA_FIM = "2026-05-01"

# --- Datas (previsao em dados novos, apos o treino) ---
PREVER_FIM = None  # None = ate hoje (Yahoo Finance)
# Use ultimos N dias de pregão (recomendado para apresentacao):
PREVER_ULTIMOS_DIAS = 30
# Ou fixe uma data (deixe None para usar PREVER_ULTIMOS_DIAS):
PREVER_INICIO = None  # ex: "2026-05-01"

# --- Modelo ---
JANELA_DIAS = 60          # quantos dias a LSTM olha para prever o proximo
PARTE_TREINO = 0.8        # 80% treino, 20% teste (cronologico)
EPOCAS = 50
TAXA_APRENDIZADO = 0.001
TAMANHO_LOTE = 32
NEURONIOS_LSTM = 48

# --- Pastas ---
PASTA = Path(__file__).parent
PASTA_ARTEFATOS = PASTA / "artifacts"
ARQUIVO_MODELO = PASTA_ARTEFATOS / "modelo.pt"

# --- API (python api.py) ---
API_PORTA = 8765   # 8000 costuma estar ocupado por outros servicos
API_TREINAR_SE_FALTAR = True   # treina automaticamente se modelo.pt nao existir
API_ABRIR_NAVEGADOR = True     # abre navegador quando servidor estiver pronto
API_DIAS_PADRAO = 10           # linhas na tabela (ultimos dias)
