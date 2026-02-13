"""
generate_dataset.py - Geracao de datasets para o projeto Gelato Magico.

Gera dois datasets:
  1. ice_cream_sales_original.csv  - dados base inspirados no dataset Kaggle
     (raphaelmanayon/temperature-and-ice-cream-sales, licenca MIT).
  2. gelato_magico_vendas.csv      - dataset sintetico com features expandidas
     simulando um ano completo (2025) de vendas da sorveteria Gelato Magico
     em Sao Paulo, Brasil.

Uso:
    python src/generate_dataset.py          # executa a partir da raiz do projeto
    from src.generate_dataset import main   # importavel como modulo
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"

SEED = 42

# Feriados nacionais brasileiros em 2025
FERIADOS_BR_2025 = [
    datetime.date(2025, 1, 1),   # Confraternizacao Universal
    datetime.date(2025, 3, 3),   # Carnaval (segunda)
    datetime.date(2025, 3, 4),   # Carnaval (terca)
    datetime.date(2025, 4, 18),  # Sexta-feira Santa
    datetime.date(2025, 4, 21),  # Tiradentes
    datetime.date(2025, 5, 1),   # Dia do Trabalho
    datetime.date(2025, 6, 19),  # Corpus Christi
    datetime.date(2025, 9, 7),   # Independencia do Brasil
    datetime.date(2025, 10, 12), # Nossa Sra. Aparecida
    datetime.date(2025, 11, 2),  # Finados
    datetime.date(2025, 11, 15), # Proclamacao da Republica
    datetime.date(2025, 11, 20), # Consciencia Negra
    datetime.date(2025, 12, 25), # Natal
]


# ---------------------------------------------------------------------------
# 1. Dataset base (inspirado no Kaggle)
# ---------------------------------------------------------------------------
def generate_kaggle_like_data(n: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Gera dados de vendas de sorvete vs temperatura com correlacao ~0.95.

    Colunas: Temperature (C), Revenue (USD).
    Temperatura entre 0 e 45 C, receita entre ~10 e ~600.
    """
    rng = np.random.default_rng(seed)

    temperature = rng.uniform(0, 45, size=n)
    temperature.sort()

    # Relacao linear com ruido controlado para correlacao ~0.95
    slope = 12.5       # ~12.5 USD por grau C
    intercept = 20.0   # receita base
    noise_std = 35.0   # desvio-padrao do ruido

    revenue = intercept + slope * temperature + rng.normal(0, noise_std, size=n)
    revenue = np.clip(revenue, 10, 600)

    df = pd.DataFrame({
        "Temperature": np.round(temperature, 1),
        "Revenue": np.round(revenue, 2),
    })
    return df


# ---------------------------------------------------------------------------
# 2. Dataset sintetico Gelato Magico (365 dias - 2025)
# ---------------------------------------------------------------------------
def _estacao_hemisferio_sul(month: int) -> str:
    """Retorna a estacao do ano no hemisferio sul para um dado mes."""
    if month in (12, 1, 2):
        return "verao"
    if month in (3, 4, 5):
        return "outono"
    if month in (6, 7, 8):
        return "inverno"
    return "primavera"


def _temperatura_sao_paulo(day_of_year: int, rng: np.random.Generator) -> float:
    """Gera temperatura diaria realista para Sao Paulo (hemisferio sul).

    Modelo senoidal centrado no verao austral (pico em janeiro ~ dia 15)
    com amplitude e offset calibrados:
        media verao ~28 C, media inverno ~17 C.
    """
    # Dia do pico de temperatura (meio de janeiro)
    peak_day = 15
    # Senoide: maximo no verao, minimo no inverno
    seasonal = np.cos(2 * np.pi * (day_of_year - peak_day) / 365)
    mean_temp = 22.5 + 5.5 * seasonal  # varia entre ~17 e ~28
    noise = rng.normal(0, 2.5)
    return round(float(np.clip(mean_temp + noise, 8, 40)), 1)


def generate_gelato_magico_data(seed: int = SEED) -> pd.DataFrame:
    """Gera 365 registros (2025-01-01 a 2025-12-31) para Gelato Magico.

    Colunas:
        data, temperatura, dia_da_semana, eh_feriado, estacao, vendas
    """
    rng = np.random.default_rng(seed)

    start = datetime.date(2025, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(365)]
    feriados_set = set(FERIADOS_BR_2025)

    records: list[dict] = []
    for d in dates:
        day_of_year = d.timetuple().tm_yday
        temp = _temperatura_sao_paulo(day_of_year, rng)
        dow = d.weekday()  # 0=seg ... 6=dom
        feriado = 1 if d in feriados_set else 0
        estacao = _estacao_hemisferio_sul(d.month)

        # --- Modelo de vendas (R$) ---
        # Componente principal: temperatura
        vendas_base = 50 + 15 * temp  # R$50 base + R$15 por grau

        # Boost de fim de semana (~15%)
        if dow >= 5:  # sabado ou domingo
            vendas_base *= 1.15

        # Boost de feriado (~20%)
        if feriado:
            vendas_base *= 1.20

        # Efeito sazonal adicional (alem do que a temperatura ja captura)
        season_bonus = {
            "verao": 30,
            "primavera": 10,
            "outono": -10,
            "inverno": -25,
        }
        vendas_base += season_bonus[estacao]

        # Ruido aleatorio (~8% do valor)
        noise = rng.normal(0, 0.08 * vendas_base)
        vendas_final = max(round(float(vendas_base + noise), 2), 20.0)

        records.append({
            "data": d.isoformat(),
            "temperatura": temp,
            "dia_da_semana": dow,
            "eh_feriado": feriado,
            "estacao": estacao,
            "vendas": vendas_final,
        })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# 3. Descricao dos dados
# ---------------------------------------------------------------------------
DESCRICAO_DADOS = """\
================================================================================
  DESCRICAO DOS DATASETS - Projeto Gelato Magico
================================================================================

1. ice_cream_sales_original.csv
   ----------------------------
   Fonte: Inspirado no dataset publico do Kaggle
          "Temperature and Ice Cream Sales"
          (raphaelmanayon/temperature-and-ice-cream-sales)
          Licenca: MIT

   Descricao: 500 registros sinteticos que reproduzem a relacao entre
   temperatura ambiente e receita de vendas de sorvete. A correlacao
   entre as variaveis eh de aproximadamente 0.95.

   Colunas:
     - Temperature  (float) : Temperatura em graus Celsius (0 a 45 C)
     - Revenue      (float) : Receita em dolares americanos (USD 10 a 600)


2. gelato_magico_vendas.csv
   -------------------------
   Fonte: Dados sinteticos gerados para simular as vendas diarias da
          sorveteria ficticia "Gelato Magico", localizada em Sao Paulo,
          Brasil, ao longo do ano de 2025.

   Descricao: 365 registros (01/jan/2025 a 31/dez/2025) com temperatura
   diaria baseada no clima real de Sao Paulo (hemisferio sul) e vendas
   influenciadas por temperatura, dia da semana, feriados e estacao.

   Colunas:
     - data            (str)   : Data no formato ISO (AAAA-MM-DD)
     - temperatura     (float) : Temperatura media do dia em graus Celsius
     - dia_da_semana   (int)   : Dia da semana (0=segunda ... 6=domingo)
     - eh_feriado      (int)   : Indicador de feriado nacional (0 ou 1)
     - estacao         (str)   : Estacao do ano no hemisferio sul
                                 (verao / outono / inverno / primavera)
     - vendas          (float) : Receita do dia em reais (R$)

   Feriados nacionais considerados (2025):
     01/jan  Confraternizacao Universal
     03/mar  Carnaval (segunda)
     04/mar  Carnaval (terca)
     18/abr  Sexta-feira Santa
     21/abr  Tiradentes
     01/mai  Dia do Trabalho
     19/jun  Corpus Christi
     07/set  Independencia do Brasil
     12/out  Nossa Sra. Aparecida
     02/nov  Finados
     15/nov  Proclamacao da Republica
     20/nov  Consciencia Negra
     25/dez  Natal

   Modelo de geracao de vendas:
     vendas_base = 50 + 15 * temperatura
     + boost fim de semana  (+15%)
     + boost feriado        (+20%)
     + bonus sazonal        (verao +30, primavera +10, outono -10, inverno -25)
     + ruido gaussiano      (~8% do valor base)

================================================================================
  Gerado automaticamente por src/generate_dataset.py
  Seed fixa (42) para reprodutibilidade.
================================================================================
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Gera e salva todos os datasets e o arquivo de descricao."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset base (Kaggle-like)
    print("[1/3] Gerando ice_cream_sales_original.csv ...")
    df_original = generate_kaggle_like_data()
    path_original = INPUTS_DIR / "ice_cream_sales_original.csv"
    df_original.to_csv(path_original, index=False)
    corr = df_original["Temperature"].corr(df_original["Revenue"])
    print(f"      {len(df_original)} registros | correlacao = {corr:.4f}")
    print(f"      Salvo em: {path_original}")

    # 2. Dataset Gelato Magico
    print("[2/3] Gerando gelato_magico_vendas.csv ...")
    df_gelato = generate_gelato_magico_data()
    path_gelato = INPUTS_DIR / "gelato_magico_vendas.csv"
    df_gelato.to_csv(path_gelato, index=False)
    print(f"      {len(df_gelato)} registros")
    print(f"      Temperatura: {df_gelato['temperatura'].min()} - "
          f"{df_gelato['temperatura'].max()} C")
    print(f"      Vendas:      R${df_gelato['vendas'].min():.2f} - "
          f"R${df_gelato['vendas'].max():.2f}")
    print(f"      Feriados:    {df_gelato['eh_feriado'].sum()} dias")
    print(f"      Salvo em: {path_gelato}")

    # 3. Descricao
    print("[3/3] Salvando descricao_dados.txt ...")
    path_desc = INPUTS_DIR / "descricao_dados.txt"
    path_desc.write_text(DESCRICAO_DADOS, encoding="utf-8")
    print(f"      Salvo em: {path_desc}")

    print("\nConcluido com sucesso!")


if __name__ == "__main__":
    main()
