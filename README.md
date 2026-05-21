# LSTM Ações — Versão Simples (apresentação)

Projeto **mínimo** para gravar o vídeo do Tech Challenge: 3 scripts numerados + 2 arquivos de apoio.

## Estrutura

```
fase4-simple/
├── config.py      ← todas as configuracoes
├── lib.py         ← dados + LSTM + metricas (um unico modulo)
├── 1_dados.py     ← PASSO 1: coleta
├── 2_treinar.py   ← PASSO 2: treino + teste
├── 3_prever.py         ← PASSO 3: dados novos (terminal)
├── api.py              ← API + pagina visual /view
├── servico_previsao.py ← logica da API (reusa lib.py)
└── artifacts/
    └── modelo.pt
```

**Sem:** MLflow, PyTorch Lightning, FastAPI, Strategy pattern, pasta `src/`.

## Setup (uma vez)

```powershell
cd pos\fase4-simple
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Um comando so (recomendado para apresentacao)

```powershell
python api.py
```

Abre o navegador em **http://localhost:8765** e:
1. Treina o modelo automaticamente se `artifacts/modelo.pt` nao existir
2. Baixa dados recentes do Yahoo
3. Mostra historico + previsoes + proximo dia

## Scripts separados (opcional, terminal)

```powershell
python 1_dados.py      # mostra tabela e formas
python 2_treinar.py    # treina + metricas no teste
python 3_prever.py     # tabela real vs previsto (dados novos)
```

## URLs da API

| URL | O que mostra |
|-----|----------------|
| http://localhost:8765/ | Pagina principal (abre sozinha) |
| http://localhost:8765/?data=2026-05-15&dias=30 | Filtra dia e historico |
| http://localhost:8765/api/previsao | JSON |
| http://localhost:8765/status | Status do servidor |

## O que falar em cada passo

| Script | Ideia em 1 frase |
|--------|------------------|
| `1_dados.py` | Baixamos CEAB3 no Yahoo; cada amostra = 60 dias → prever o Close do dia seguinte. |
| `2_treinar.py` | LSTM treina nos 80% iniciais; avaliamos nos 20% finais (sem embaralhar). |
| `3_prever.py` | Mesmo modelo, período recente (`PREVER_INICIO`) — simula uso em produção. |

## Métricas

- **MAPE** = erro percentual médio (menor é melhor)
- **Acurácia** = `100 - MAPE` (meta **> 85%**)

## Ajustes rápidos (`config.py`)

| Variável | Significado |
|----------|-------------|
| `TICKER` | Ação no Yahoo (ex: CEAB3.SA) |
| `DATA_INICIO` / `DATA_FIM` | Período do treino |
| `PREVER_ULTIMOS_DIAS` | Quantos dias recentes mostrar (padrão: 30) |
| `PREVER_INICIO` | Data fixa (ou `None` = usa últimos N dias) |
| `JANELA_DIAS` | 60 dias de histórico por previsão |
| `EPOCAS` | Rodadas de treino (50 padrão) |
