"""
data_preparation.py - Preparacao e analise exploratoria dos dados para o projeto Gelato Magico.

Funcionalidades:
  1. Carregamento e validacao de dados CSV.
  2. Analise exploratoria (EDA) com estatisticas descritivas.
  3. Limpeza de dados (valores ausentes, duplicatas, faixas validas).
  4. Divisao em conjuntos de treino e teste.
  5. Geracao de graficos EDA salvos em disco.

Uso:
    python src/data_preparation.py          # executa a partir da raiz do projeto
    from src.data_preparation import *      # importavel como modulo
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# 1. Carregamento de dados
# ---------------------------------------------------------------------------
def load_data(filepath: str | Path) -> pd.DataFrame:
    """Carrega um arquivo CSV em um DataFrame pandas.

    Parametros
    ----------
    filepath : str | Path
        Caminho para o arquivo CSV a ser carregado.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados carregados.

    Levanta
    -------
    FileNotFoundError
        Se o arquivo nao for encontrado no caminho especificado.
    ValueError
        Se o arquivo estiver vazio ou nao for um CSV valido.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

    if filepath.suffix.lower() != ".csv":
        raise ValueError(f"Formato de arquivo nao suportado (esperado .csv): {filepath.suffix}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"O arquivo esta vazio: {filepath}")

    print(f"[load_data] Dados carregados com sucesso: {filepath.name}")
    print(f"            {df.shape[0]} registros x {df.shape[1]} colunas")

    return df


# ---------------------------------------------------------------------------
# 2. Analise exploratoria
# ---------------------------------------------------------------------------
def explore_data(df: pd.DataFrame) -> dict[str, Any]:
    """Realiza analise exploratoria basica e retorna um dicionario de resultados.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.

    Retorna
    -------
    dict[str, Any]
        Dicionario contendo:
          - shape: tupla (linhas, colunas)
          - dtypes: tipos de dados de cada coluna
          - describe: estatisticas descritivas
          - missing_values: contagem de valores ausentes por coluna
          - correlation: matriz de correlacao (apenas colunas numericas)
    """
    numeric_df = df.select_dtypes(include=[np.number])

    report: dict[str, Any] = {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "describe": df.describe(),
        "missing_values": df.isnull().sum(),
        "correlation": numeric_df.corr() if not numeric_df.empty else pd.DataFrame(),
    }

    print("\n" + "=" * 60)
    print("  ANALISE EXPLORATORIA DOS DADOS")
    print("=" * 60)
    print(f"\nFormato: {report['shape'][0]} linhas x {report['shape'][1]} colunas")
    print(f"\nTipos de dados:\n{report['dtypes']}")
    print(f"\nEstatisticas descritivas:\n{report['describe']}")
    print(f"\nValores ausentes:\n{report['missing_values']}")
    print(f"\nMatriz de correlacao:\n{report['correlation']}")
    print("=" * 60)

    return report


# ---------------------------------------------------------------------------
# 3. Limpeza de dados
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa o DataFrame tratando valores ausentes, duplicatas e faixas invalidas.

    Etapas realizadas:
      - Preenchimento de valores ausentes em colunas numericas com a mediana.
      - Remocao de registros duplicados.
      - Validacao da faixa de temperatura (0 a 50 graus Celsius).

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame a ser limpo.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo (copia do original).
    """
    df_clean = df.copy()
    registros_iniciais = len(df_clean)

    # --- Valores ausentes: preencher numericas com mediana ---
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"[clean_data] Coluna '{col}': {missing_count} valores ausentes "
                  f"preenchidos com mediana ({median_val:.2f})")

    # --- Duplicatas ---
    duplicatas = df_clean.duplicated().sum()
    if duplicatas > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[clean_data] {duplicatas} registros duplicados removidos.")

    # --- Validacao de temperatura (0 a 50 C) ---
    if "temperatura" in df_clean.columns:
        fora_faixa = ~df_clean["temperatura"].between(0, 50)
        n_fora = fora_faixa.sum()
        if n_fora > 0:
            df_clean = df_clean[~fora_faixa].reset_index(drop=True)
            print(f"[clean_data] {n_fora} registros com temperatura fora da faixa "
                  f"(0-50 C) removidos.")

    registros_finais = len(df_clean)
    removidos = registros_iniciais - registros_finais
    print(f"[clean_data] Limpeza concluida: {registros_iniciais} -> {registros_finais} "
          f"registros ({removidos} removidos).")

    return df_clean


# ---------------------------------------------------------------------------
# 4. Divisao treino/teste
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target_col: str = "vendas",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide o DataFrame em conjuntos de treino e teste.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame com as features e a variavel alvo.
    target_col : str
        Nome da coluna alvo (variavel dependente). Padrao: 'vendas'.
    test_size : float
        Proporcao dos dados para o conjunto de teste (0 a 1). Padrao: 0.2.
    random_state : int
        Semente para reproducibilidade. Padrao: 42.

    Retorna
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    Levanta
    -------
    ValueError
        Se a coluna alvo nao existir no DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' nao encontrada no DataFrame. "
                         f"Colunas disponiveis: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n[split_data] Divisao treino/teste (test_size={test_size}):")
    print(f"             Treino: {X_train.shape[0]} registros")
    print(f"             Teste:  {X_test.shape[0]} registros")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Geracao de graficos EDA
# ---------------------------------------------------------------------------
def generate_eda_report(df: pd.DataFrame, output_dir: str | Path = None) -> None:
    """Gera graficos de analise exploratoria e salva como imagens PNG.

    Graficos gerados:
      - Distribuicao da temperatura (histograma)
      - Distribuicao das vendas (histograma)
      - Temperatura vs Vendas (dispersao)
      - Mapa de calor da correlacao
      - Vendas por dia da semana (barra) — se a coluna existir
      - Vendas por estacao (boxplot) — se a coluna existir

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame com os dados para gerar os graficos.
    output_dir : str | Path, opcional
        Diretorio onde os graficos serao salvos. Se nao informado,
        utiliza ``outputs/graficos/`` na raiz do projeto.
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "graficos"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("viridis")

    print("\n[EDA] Gerando graficos de analise exploratoria...")

    # --- 1. Distribuicao da temperatura ---
    if "temperatura" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["temperatura"], bins=30, kde=True, color="#2196F3", ax=ax)
        ax.set_title("Distribuicao da Temperatura", fontsize=16, fontweight="bold")
        ax.set_xlabel("Temperatura (°C)", fontsize=12)
        ax.set_ylabel("Frequencia", fontsize=12)
        fig.tight_layout()
        filepath = output_dir / "distribuicao_temperatura.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       Salvo: {filepath.name}")

    # --- 2. Distribuicao das vendas ---
    if "vendas" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["vendas"], bins=30, kde=True, color="#4CAF50", ax=ax)
        ax.set_title("Distribuicao das Vendas", fontsize=16, fontweight="bold")
        ax.set_xlabel("Vendas (R$)", fontsize=12)
        ax.set_ylabel("Frequencia", fontsize=12)
        fig.tight_layout()
        filepath = output_dir / "distribuicao_vendas.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       Salvo: {filepath.name}")

    # --- 3. Temperatura vs Vendas (dispersao) ---
    if "temperatura" in df.columns and "vendas" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df, x="temperatura", y="vendas",
            alpha=0.6, color="#FF9800", edgecolor="white", s=60, ax=ax,
        )
        # Linha de tendencia
        z = np.polyfit(df["temperatura"], df["vendas"], 1)
        p = np.poly1d(z)
        temp_range = np.linspace(df["temperatura"].min(), df["temperatura"].max(), 100)
        ax.plot(temp_range, p(temp_range), "--", color="#E53935", linewidth=2,
                label=f"Tendencia (y = {z[0]:.1f}x + {z[1]:.1f})")
        ax.legend(fontsize=11)
        ax.set_title("Temperatura vs Vendas", fontsize=16, fontweight="bold")
        ax.set_xlabel("Temperatura (°C)", fontsize=12)
        ax.set_ylabel("Vendas (R$)", fontsize=12)
        fig.tight_layout()
        filepath = output_dir / "temperatura_vs_vendas.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       Salvo: {filepath.name}")

    # --- 4. Mapa de calor da correlacao ---
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Mapa de Calor - Correlacao entre Variaveis",
                      fontsize=16, fontweight="bold")
        fig.tight_layout()
        filepath = output_dir / "correlacao_heatmap.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       Salvo: {filepath.name}")

    # --- 5. Vendas por dia da semana (barra) ---
    if "dia_da_semana" in df.columns and "vendas" in df.columns:
        dias_nomes = {
            0: "Segunda", 1: "Terca", 2: "Quarta",
            3: "Quinta", 4: "Sexta", 5: "Sabado", 6: "Domingo",
        }
        df_dias = df.copy()
        df_dias["dia_nome"] = df_dias["dia_da_semana"].map(dias_nomes)
        ordem_dias = ["Segunda", "Terca", "Quarta", "Quinta", "Sexta", "Sabado", "Domingo"]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df_dias, x="dia_nome", y="vendas",
            order=ordem_dias, palette="viridis", errorbar="sd", ax=ax,
        )
        ax.set_title("Vendas Medias por Dia da Semana", fontsize=16, fontweight="bold")
        ax.set_xlabel("Dia da Semana", fontsize=12)
        ax.set_ylabel("Vendas (R$)", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        filepath = output_dir / "vendas_por_dia_semana.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       Salvo: {filepath.name}")

    # --- 6. Vendas por estacao (boxplot) ---
    if "estacao" in df.columns and "vendas" in df.columns:
        ordem_estacoes = ["verao", "outono", "inverno", "primavera"]
        estacoes_presentes = [e for e in ordem_estacoes if e in df["estacao"].values]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=df, x="estacao", y="vendas",
            order=estacoes_presentes, palette="Set2", ax=ax,
        )
        ax.set_title("Distribuicao de Vendas por Estacao", fontsize=16, fontweight="bold")
        ax.set_xlabel("Estacao do Ano", fontsize=12)
        ax.set_ylabel("Vendas (R$)", fontsize=12)
        fig.tight_layout()
        filepath = output_dir / "vendas_por_estacao.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"       Salvo: {filepath.name}")

    print(f"\n[EDA] Todos os graficos salvos em: {output_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Executa o pipeline completo de preparacao e analise exploratoria dos dados."""
    print("=" * 60)
    print("  GELATO MAGICO - Preparacao de Dados e EDA")
    print("=" * 60)

    # 1. Carregar dados
    csv_path = INPUTS_DIR / "gelato_magico_vendas.csv"
    df = load_data(csv_path)

    # 2. Analise exploratoria
    explore_data(df)

    # 3. Limpeza dos dados
    print("\n--- Limpeza dos Dados ---")
    df_clean = clean_data(df)

    # 4. Divisao treino/teste
    # Selecionar apenas colunas numericas para a modelagem
    numeric_cols = ["temperatura", "dia_da_semana", "eh_feriado", "vendas"]
    df_model = df_clean[[c for c in numeric_cols if c in df_clean.columns]]
    X_train, X_test, y_train, y_test = split_data(df_model, target_col="vendas")

    # 5. Gerar graficos EDA
    graficos_dir = OUTPUTS_DIR / "graficos"
    generate_eda_report(df_clean, output_dir=graficos_dir)

    print("\n" + "=" * 60)
    print("  Pipeline de preparacao concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
