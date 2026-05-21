"""
API — abra http://localhost:8765 (porta em config.py)

  python api.py

Treina automaticamente se faltar modelo.pt, depois mostra a pagina.
"""

from __future__ import annotations

import traceback
import webbrowser
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from config import (
    API_ABRIR_NAVEGADOR,
    API_DIAS_PADRAO,
    API_PORTA,
    API_TREINAR_SE_FALTAR,
    TICKER,
)
from servico_previsao import garantir_modelo, modelo_existe, montar_painel

_pronto = False
_erro: Optional[str] = None


def _pagina_aguarde() -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"/>
<meta http-equiv="refresh" content="3"/>
<title>Carregando</title>
<style>
  body {{ font-family: Segoe UI, sans-serif; background:#f0f4f8; margin:0;
         min-height:100vh; display:flex; align-items:center; justify-content:center; }}
  .c {{ text-align:center; padding:32px; }}
  h1 {{ color:#1565c0; }}
  p {{ color:#555; }}
</style></head><body><div class="c">
  <h1>{TICKER}</h1>
  <p>Preparando modelo e baixando dados...</p>
  <p>Aguarde, recarregando em instantes.</p>
</div></body></html>"""


def _pagina_erro(msg: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"/><title>Erro</title>
<style>body{{font-family:Segoe UI,sans-serif;padding:40px;max-width:560px;margin:auto;}}
h1{{color:#c62828;}}</style></head><body>
<h1>Algo deu errado</h1>
<p>{msg}</p>
<p><strong>O que fazer:</strong> feche o terminal, rode de novo <code>python api.py</code>
e verifique sua conexao com a internet.</p>
</body></html>"""


def _html_painel(p: dict, dias: int) -> str:
    hist = p.get("historico") or []
    hist = hist[-dias:] if hist else []

    linhas = ""
    for h in reversed(hist):
        diff = float(h["close_previsto"]) - float(h["close_real"])
        sinal = "+" if diff >= 0 else ""
        linhas += f"""
        <tr>
          <td>{h['data']}</td>
          <td>R$ {float(h['close_real']):.2f}</td>
          <td>R$ {float(h['close_previsto']):.2f}</td>
          <td class="{'up' if diff >= 0 else 'down'}">{sinal}{diff:.2f}</td>
        </tr>"""

    if not linhas:
        linhas = '<tr><td colspan="4">Sem dados no periodo. Tente outra data.</td></tr>'

    acuracia = float(p.get("metricas_periodo", {}).get("acuracia", 0))
    eh_futuro = bool(p.get("dia_solicitado_eh_futuro"))
    passos = int(p.get("dia_solicitado_passos_futuro") or 0)
    preco_dia = float(p["dia_solicitado_close_previsto"])

    if eh_futuro:
        hero_titulo = f"Previsao para {p['dia_solicitado']}"
        hero_nota = (
            f"Estimativa ({passos} pregão(s) à frente do último dado) · "
            f"último real {p['ultimo_pregao']}: R$ {float(p['ultimo_fechamento_real']):.2f}"
        )
        card_real = """<div class="val">—</div>
      <div class="lbl">pregão ainda não realizado</div>"""
        card_pred_lbl = "estimativa LSTM"
    else:
        hero_titulo = "Proximo pregão (estimativa)"
        hero_nota = (
            f"Último fechamento {p['ultimo_pregao']}: "
            f"R$ {float(p['ultimo_fechamento_real']):.2f}"
        )
        card_real = (
            f'<div class="val">R$ {float(p["dia_solicitado_close_real"]):.2f}</div>'
            f'<div class="lbl">{p["dia_solicitado"]}</div>'
        )
        card_pred_lbl = f"erro {float(p['dia_solicitado_erro_pct']):.1f}%"

    hero_preco = preco_dia if eh_futuro else float(p["previsao_proximo_dia"])

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{p['ticker']}</title>
  <style>
    body {{ font-family: Segoe UI, system-ui, sans-serif; margin:0; padding:24px 16px;
            background:#f0f4f8; color:#222; max-width:640px; margin:0 auto; }}
    h1 {{ margin:0 0 4px; font-size:1.4rem; }}
    .sub {{ color:#666; margin:0 0 20px; font-size:0.9rem; }}
    .hero {{ background:#1565c0; color:#fff; border-radius:12px; padding:24px;
             text-align:center; margin-bottom:20px; }}
    .hero small {{ opacity:0.9; }}
    .hero .price {{ font-size:2.4rem; font-weight:700; margin:8px 0; }}
    .row {{ display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }}
    .card {{ flex:1; min-width:140px; background:#fff; border-radius:10px; padding:16px;
             text-align:center; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
    .card .lbl {{ font-size:0.75rem; color:#888; text-transform:uppercase; }}
    .card .val {{ font-size:1.4rem; font-weight:600; margin-top:6px; }}
    .real .val {{ color:#2e7d32; }}
    .pred .val {{ color:#1565c0; }}
    form {{ background:#fff; padding:14px; border-radius:10px; margin-bottom:20px;
            display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap;
            box-shadow:0 1px 3px rgba(0,0,0,.08); }}
    label {{ font-size:0.8rem; color:#666; display:block; margin-bottom:4px; }}
    input[type=date] {{ padding:8px; border:1px solid #ccc; border-radius:6px; }}
    button {{ padding:9px 18px; background:#1565c0; color:#fff; border:none;
              border-radius:6px; cursor:pointer; font-size:1rem; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px;
              overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.08); font-size:0.9rem; }}
    th {{ background:#e8eef5; padding:10px; text-align:left; }}
    td {{ padding:10px; border-top:1px solid #eee; }}
    td:not(:first-child) {{ text-align:right; }}
    .up {{ color:#2e7d32; }} .down {{ color:#c62828; }}
    footer {{ text-align:center; color:#888; font-size:0.8rem; margin-top:16px; }}
  </style>
</head>
<body>
  <h1>{p['ticker']}</h1>
  <p class="sub">Previsao LSTM · Yahoo Finance</p>

  <div class="hero">
    <small>{hero_titulo}</small>
    <div class="price">R$ {hero_preco:.2f}</div>
    <small>{hero_nota}</small>
  </div>

  <form method="get" action="/">
    <div>
      <label>Escolher dia (passado, hoje ou futuro)</label>
      <input type="date" name="data" value="{p['dia_solicitado']}"/>
    </div>
    <input type="hidden" name="dias" value="{dias}"/>
    <button type="submit">Atualizar</button>
  </form>

  <div class="row">
    <div class="card real">
      <div class="lbl">Real</div>
      {card_real}
    </div>
    <div class="card pred">
      <div class="lbl">Previsao</div>
      <div class="val">R$ {preco_dia:.2f}</div>
      <div class="lbl">{card_pred_lbl}</div>
    </div>
  </div>

  <table>
    <thead><tr><th>Data</th><th>Real</th><th>Previsao</th><th>Dif.</th></tr></thead>
    <tbody>{linhas}</tbody>
  </table>

  <footer>Precisao no periodo: {acuracia:.1f}%</footer>
</body>
</html>"""


def _inicializar() -> None:
    global _pronto, _erro
    print("=" * 50)
    print(" Iniciando API LSTM")
    print("=" * 50)
    try:
        if API_TREINAR_SE_FALTAR:
            garantir_modelo(log=True)
        elif not modelo_existe():
            raise FileNotFoundError(
                "Arquivo artifacts/modelo.pt nao existe. "
                "Rode: python 2_treinar.py"
            )
        _pronto = True
        _erro = None
        print(" Pronto! Modelo carregado.")
    except Exception as exc:
        _pronto = False
        _erro = str(exc)
        print(f" ERRO: {exc}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(application: FastAPI):
    _inicializar()
    url = f"http://127.0.0.1:{API_PORTA}/"
    print(f" Servidor: {url}")
    if API_ABRIR_NAVEGADOR and _pronto:
        webbrowser.open(url)
    yield
    print(" Servidor encerrado.")


app = FastAPI(title=f"LSTM {TICKER}", lifespan=lifespan)


def _gerar_pagina(data: Optional[str], dias: int) -> HTMLResponse:
    if _erro:
        return HTMLResponse(_pagina_erro(_erro), status_code=500)
    if not _pronto:
        return HTMLResponse(_pagina_aguarde())
    try:
        painel = montar_painel(data_alvo=data, dias_historico=dias)
        return HTMLResponse(_html_painel(painel, dias))
    except ValueError as exc:
        return HTMLResponse(_pagina_erro(str(exc)), status_code=400)
    except Exception as exc:
        traceback.print_exc()
        return HTMLResponse(_pagina_erro(str(exc)), status_code=500)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/view", response_class=HTMLResponse, include_in_schema=False)
def pagina(
    data: Optional[str] = Query(None),
    dias: int = Query(API_DIAS_PADRAO, ge=5, le=30),
):
    return _gerar_pagina(data, dias)


@app.get("/api/previsao", tags=["API"])
def api_json(
    data: Optional[str] = Query(None),
    dias: int = Query(API_DIAS_PADRAO, ge=5, le=90),
):
    if _erro:
        raise HTTPException(500, _erro)
    if not _pronto:
        raise HTTPException(503, "Ainda inicializando. Recarregue em alguns segundos.")
    try:
        return montar_painel(data_alvo=data, dias_historico=dias)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/status")
def status():
    return {"pronto": _pronto, "erro": _erro, "ticker": TICKER}


if __name__ == "__main__":
    import uvicorn

    print(f"Iniciando em http://127.0.0.1:{API_PORTA}")
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=API_PORTA,
        reload=False,
    )
