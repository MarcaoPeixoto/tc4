"""
Carrega modelo.pt e monta historico + previsao para a API.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, cast

import numpy as np
import pandas as pd
import torch

from config import (
    ARQUIVO_MODELO,
    DATA_FIM,
    DATA_INICIO,
    EPOCAS,
    JANELA_DIAS,
    NEURONIOS_LSTM,
    PARTE_TREINO,
    PASTA_ARTEFATOS,
    PREVER_FIM,
    TAMANHO_LOTE,
    TAXA_APRENDIZADO,
    TICKER,
)
from lib import (
    LSTMPrevisao,
    baixar_acao,
    calcular_metricas,
    criar_sequencias,
    dividir_treino_teste,
    normalizar,
    prever,
    treinar,
)
from torch.utils.data import DataLoader, TensorDataset

_treino_em_andamento = False


def _data_menos_dias(base: str, dias: int) -> str:
    """Subtrai dias de uma data YYYY-MM-DD (stdlib — evita NaT no type checker)."""
    return (datetime.strptime(base, "%Y-%m-%d") - timedelta(days=dias)).strftime("%Y-%m-%d")


def _proximo_pregao(ultima_data: str) -> pd.Timestamp:
    """Próximo dia útil após ultima_data (YYYY-MM-DD)."""
    d = datetime.strptime(ultima_data, "%Y-%m-%d") + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return cast(pd.Timestamp, pd.Timestamp(d.strftime("%Y-%m-%d")))


def modelo_existe() -> bool:
    return ARQUIVO_MODELO.exists()


def executar_treino(log: bool = True) -> dict:
    """Pipeline completo do passo 2: dados -> treino -> salvar modelo.pt."""
    global _treino_em_andamento
    if _treino_em_andamento:
        raise RuntimeError("Treinamento ja em andamento.")
    _treino_em_andamento = True
    try:
        if log:
            print("=" * 55)
            print(" TREINAMENTO (API / pipeline)")
            print("=" * 55)
            print(f" Ticker: {TICKER}  |  Epocas: {EPOCAS}")

        df = baixar_acao(TICKER, DATA_INICIO, DATA_FIM)
        X, y, datas = criar_sequencias(df, JANELA_DIAS)
        X_tr, y_tr, _, X_te, y_te, _ = dividir_treino_teste(X, y, datas, PARTE_TREINO)
        X_tr, y_tr, X_te, y_te, scaler_x, scaler_y, inversa_y = normalizar(
            X_tr, y_tr, X_te, y_te
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=TAMANHO_LOTE,
            shuffle=True,
        )
        modelo = LSTMPrevisao(neuronios=NEURONIOS_LSTM).to(device)
        if log:
            print(" Treinando...")
        treinar(modelo, loader, EPOCAS, TAXA_APRENDIZADO, device)

        pred_norm = prever(modelo, X_te, device)
        pred = inversa_y(pred_norm)
        real = inversa_y(y_te)
        metricas = calcular_metricas(real, pred)

        PASTA_ARTEFATOS.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "modelo": modelo.state_dict(),
                "scaler_x": scaler_x,
                "scaler_y": scaler_y,
                "metricas_treino": metricas,
                "janela": JANELA_DIAS,
                "neuronios": NEURONIOS_LSTM,
                "ticker": TICKER,
            },
            ARQUIVO_MODELO,
        )
        if log:
            print(f" Modelo salvo: {ARQUIVO_MODELO}")
            print(f" Acuracia teste: {metricas['acuracia']:.2f}%")
        return metricas
    finally:
        _treino_em_andamento = False


def garantir_modelo(log: bool = True) -> None:
    """Treina automaticamente se artifacts/modelo.pt nao existir."""
    if modelo_existe():
        if log:
            print(f" Modelo OK: {ARQUIVO_MODELO}")
        return
    if log:
        print(" Modelo nao encontrado — iniciando treino automatico...")
    executar_treino(log=log)


def _carregar_pacote(caminho: Path) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {caminho}. Execute: python 2_treinar.py"
        )
    try:
        return torch.load(caminho, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(caminho, map_location="cpu")


def _volume_original(volume_log: float) -> float:
    return float(np.expm1(volume_log))


def _linha_ohlcv(df: pd.DataFrame, data: str) -> dict:
    ts = pd.Timestamp(data)
    if ts not in df.index:
        return {}
    row = df.loc[ts]
    return {
        "data": data,
        "high": round(float(row["High"]), 4),
        "low": round(float(row["Low"]), 4),
        "close": round(float(row["Close"]), 4),
        "volume": round(_volume_original(float(row["Volume"])), 0),
    }


def _janela_ohlcv(df: pd.DataFrame, indice_ini: int, janela: int) -> List[dict]:
    """Dias de entrada [indice_ini .. indice_ini+janela-1] no DataFrame."""
    linhas = []
    for i in range(indice_ini, indice_ini + janela):
        d = str(df.index[i])[:10]
        ohlc = _linha_ohlcv(df, d)
        if ohlc:
            linhas.append(ohlc)
    return linhas


def _prever_proximo_dia(
    modelo: LSTMPrevisao,
    df: pd.DataFrame,
    janela: int,
    scaler_x,
    scaler_y,
    neuronios: int,
    device: torch.device,
) -> float:
    """Usa os ultimos `janela` pregões do df para prever o fechamento seguinte."""
    if len(df) < janela:
        raise ValueError("Historico curto demais para prever o proximo dia.")
    ultima = df.values[-janela:].astype(np.float32).reshape(1, janela, -1)
    n_feat = ultima.shape[2]
    ultima = scaler_x.transform(ultima.reshape(-1, n_feat)).reshape(ultima.shape).astype(np.float32)
    pred_norm = prever(modelo, ultima, device)
    return float(scaler_y.inverse_transform(pred_norm.reshape(-1, 1))[0][0])


def _pregoes_uteis_ate(ultimo_pregao: str, data_alvo: str) -> int:
    """Quantos pregões úteis existem entre o último dado e a data pedida (inclusive)."""
    inicio = pd.Timestamp(ultimo_pregao) + pd.offsets.BDay(1)
    fim = pd.Timestamp(data_alvo)
    if fim < inicio:
        return 0
    return len(pd.bdate_range(inicio, fim))


def _anexar_pregao_sintetico(df: pd.DataFrame, data: pd.Timestamp, close: float) -> pd.DataFrame:
    """Adiciona um dia sintético ao histórico para encadear previsões multi-dia."""
    df = df.copy()
    ultima = df.iloc[-1]
    ratio = close / float(ultima["Close"]) if float(ultima["Close"]) else 1.0
    df.loc[data] = {
        "High": float(ultima["High"]) * ratio,
        "Low": float(ultima["Low"]) * ratio,
        "Close": close,
        "Volume": float(ultima["Volume"]),
    }
    return df.sort_index()


def _prever_para_data_futura(
    modelo: LSTMPrevisao,
    df: pd.DataFrame,
    janela: int,
    scaler_x,
    scaler_y,
    neuronios: int,
    device: torch.device,
    data_alvo: str,
    ultimo_pregao: str,
) -> tuple[float, int]:
    """
    Preve o Close para uma data futura (após o último pregão com dados reais).
    Encadeia previsões dia a dia (cada passo usa a previsão anterior na janela).
    """
    alvo = pd.Timestamp(data_alvo)
    ultimo = pd.Timestamp(ultimo_pregao)
    if alvo <= ultimo:
        raise ValueError("Data nao e futura em relacao ao ultimo pregao.")

    passos = _pregoes_uteis_ate(ultimo_pregao, data_alvo)
    if passos == 0:
        passos = 1

    df_work = df.copy()
    pred = float(df_work.iloc[-1]["Close"])
    for _ in range(passos):
        pred = _prever_proximo_dia(
            modelo, df_work, janela, scaler_x, scaler_y, neuronios, device
        )
        proximo = _proximo_pregao(str(df_work.index[-1])[:10])
        df_work = _anexar_pregao_sintetico(df_work, proximo, pred)

    return round(pred, 4), passos


def montar_painel(
    data_alvo: Optional[str] = None,
    dias_historico: int = 30,
    fim: Optional[str] = None,
) -> dict:
    """
    data_alvo: dia cuja previsao de Close deseja ver (YYYY-MM-DD). None = ultimo dia disponivel.
    dias_historico: quantos dias na tabela historico (real vs previsto).
    """
    pacote = _carregar_pacote(ARQUIVO_MODELO)
    scaler_x = pacote["scaler_x"]
    scaler_y = pacote["scaler_y"]
    janela = int(pacote.get("janela", JANELA_DIAS))
    neuronios = int(pacote.get("neuronios", 48))

    fim_busca = fim or PREVER_FIM or datetime.today().strftime("%Y-%m-%d")
    inicio_busca = _data_menos_dias(fim_busca, dias_historico + janela + 60)

    df = baixar_acao(TICKER, inicio_busca, fim_busca)
    X, y, datas = criar_sequencias(df, janela)

    if not datas:
        raise ValueError("Dados insuficientes para montar sequencias.")

    n_feat = X.shape[2]
    X_scaled = scaler_x.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype(np.float32)

    device = torch.device("cpu")
    modelo = LSTMPrevisao(neuronios=neuronios)
    modelo.load_state_dict(pacote["modelo"])
    modelo.eval()

    pred_norm = prever(modelo, X_scaled, device)
    pred = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    real = y.flatten()

    # Tabela: ultimos N dias com previsao
    corte = _data_menos_dias(fim_busca, dias_historico)
    historico = []
    for d, r, p in zip(datas, real, pred):
        if d < corte:
            continue
        historico.append(
            {
                "data": d,
                "close_real": round(float(r), 4),
                "close_previsto": round(float(p), 4),
                "erro_pct": round(abs(r - p) / r * 100, 2) if r else 0.0,
                **_linha_ohlcv(df, d),
            }
        )

    metricas = calcular_metricas(
        np.array([h["close_real"] for h in historico]),
        np.array([h["close_previsto"] for h in historico]),
    )

    ultima_data = datas[-1]
    fechamento_ultimo = float(real[-1])
    previsao_proximo_dia = _prever_proximo_dia(
        modelo, df, janela, scaler_x, scaler_y, neuronios, device
    )
    data_proximo = _proximo_pregao(ultima_data).strftime("%Y-%m-%d")

    if data_alvo is None:
        data_alvo = ultima_data

    eh_futuro = pd.Timestamp(data_alvo) > pd.Timestamp(ultima_data)

    if eh_futuro:
        close_previsto_dia, passos_futuro = _prever_para_data_futura(
            modelo, df, janela, scaler_x, scaler_y, neuronios, device,
            data_alvo, ultima_data,
        )
        close_real_dia = None
        erro_pct = None
        janela_entrada = _janela_ohlcv(df, len(df) - janela, janela)
    elif data_alvo not in datas:
        raise ValueError(
            f"Data {data_alvo} nao encontrada. Disponivel: {datas[0]} ate {datas[-1]} "
            f"(ou escolha uma data apos {ultima_data} para previsao futura)."
        )
    else:
        passos_futuro = 0
        idx = datas.index(data_alvo)
        close_real_dia = round(float(real[idx]), 4)
        close_previsto_dia = round(float(pred[idx]), 4)
        erro_pct = round(
            abs(close_real_dia - close_previsto_dia) / close_real_dia * 100, 2
        ) if close_real_dia else 0.0
        janela_entrada = _janela_ohlcv(df, idx, janela)

    return {
        "ticker": TICKER,
        "modelo": str(ARQUIVO_MODELO),
        "janela_dias": janela,
        "dia_solicitado": data_alvo,
        "dia_solicitado_eh_futuro": eh_futuro,
        "dia_solicitado_passos_futuro": passos_futuro if eh_futuro else 0,
        "dia_solicitado_close_real": close_real_dia,
        "dia_solicitado_close_previsto": close_previsto_dia,
        "dia_solicitado_erro_pct": erro_pct,
        "janela_entrada_60_dias": janela_entrada,
        "historico": historico,
        "metricas_periodo": metricas,
        "metricas_treino": pacote.get("metricas_treino", {}),
        "ultimo_pregao": ultima_data,
        "ultimo_fechamento_real": round(fechamento_ultimo, 4),
        "previsao_proximo_dia": round(previsao_proximo_dia, 4),
        "data_previsao_proximo": data_proximo,
        "n_dias_historico": len(historico),
    }
