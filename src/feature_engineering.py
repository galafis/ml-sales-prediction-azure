"""
feature_engineering.py - Engenharia de features para o projeto Gelato Magico.

Funcionalidades:
  1. Criacao de features baseadas em temperatura (polinomiais e normalizacao).
  2. Criacao de features temporais (fim de semana, mes, dia do ano, semana).
  3. Criacao de features de interacao (temperatura x fim de semana/feriado).
  4. Codificacao one-hot de variaveis categoricas (estacao).
  5. Selecao de features e separacao de variavel alvo.
  6. Pipeline completo de engenharia de features.

Uso:
    python src/feature_engineering.py          # executa a partir da raiz do projeto
    from src.feature_engineering import *      # importavel como modulo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# 1. Features de temperatura
# ---------------------------------------------------------------------------
def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas da temperatura.

    Features adicionadas:
      - temperatura_quadrada: temperatura elevada ao quadrado (captura relacao nao-linear).
      - temperatura_cubica: temperatura elevada ao cubo.
      - temperatura_normalizada: normalizacao z-score (media 0, desvio padrao 1).

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna 'temperatura'.

    Retorna
    -------
    pd.DataFrame
        Copia do DataFrame com as novas colunas adicionadas.
    """
    df = df.copy()

    df["temperatura_quadrada"] = df["temperatura"] ** 2
    df["temperatura_cubica"] = df["temperatura"] ** 3

    temp_mean = df["temperatura"].mean()
    temp_std = df["temperatura"].std()
    df["temperatura_normalizada"] = (df["temperatura"] - temp_mean) / temp_std

    print(f"[create_temperature_features] 3 features de temperatura criadas.")
    print(f"    temperatura media: {temp_mean:.2f} | desvio padrao: {temp_std:.2f}")

    return df


# ---------------------------------------------------------------------------
# 2. Features temporais
# ---------------------------------------------------------------------------
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features temporais a partir das colunas 'data' e 'dia_da_semana'.

    Features adicionadas:
      - eh_fim_de_semana: 1 se dia_da_semana >= 5, 0 caso contrario.
      - mes: mes extraido da coluna 'data'.
      - dia_do_ano: dia do ano extraido da coluna 'data'.
      - semana_do_ano: semana do ano extraida da coluna 'data'.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'data' e 'dia_da_semana'.

    Retorna
    -------
    pd.DataFrame
        Copia do DataFrame com as novas colunas adicionadas.
    """
    df = df.copy()

    df["eh_fim_de_semana"] = (df["dia_da_semana"] >= 5).astype(int)

    data_dt = pd.to_datetime(df["data"])
    df["mes"] = data_dt.dt.month
    df["dia_do_ano"] = data_dt.dt.dayofyear
    df["semana_do_ano"] = data_dt.dt.isocalendar().week.astype(int)

    print(f"[create_temporal_features] 4 features temporais criadas.")

    return df


# ---------------------------------------------------------------------------
# 3. Features de interacao
# ---------------------------------------------------------------------------
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de interacao entre temperatura e variaveis categoricas.

    Features adicionadas:
      - temperatura_x_fds: temperatura multiplicada por eh_fim_de_semana.
      - temperatura_x_feriado: temperatura multiplicada por eh_feriado.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'temperatura', 'eh_fim_de_semana' e 'eh_feriado'.

    Retorna
    -------
    pd.DataFrame
        Copia do DataFrame com as novas colunas adicionadas.

    Observacao
    ----------
    Esta funcao deve ser chamada apos create_temporal_features, pois depende
    da coluna 'eh_fim_de_semana'.
    """
    df = df.copy()

    df["temperatura_x_fds"] = df["temperatura"] * df["eh_fim_de_semana"]
    df["temperatura_x_feriado"] = df["temperatura"] * df["eh_feriado"]

    print(f"[create_interaction_features] 2 features de interacao criadas.")

    return df


# ---------------------------------------------------------------------------
# 4. Codificacao de variaveis categoricas
# ---------------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica codificacao one-hot na coluna 'estacao'.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna 'estacao'.

    Retorna
    -------
    pd.DataFrame
        Copia do DataFrame com as colunas dummy da estacao adicionadas.
        A coluna original 'estacao' e removida.
    """
    df = df.copy()

    dummies = pd.get_dummies(df["estacao"], prefix="estacao", drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    categorias = list(dummies.columns)
    print(f"[encode_categorical] Coluna 'estacao' codificada em {len(categorias)} dummies: {categorias}")

    return df


# ---------------------------------------------------------------------------
# 5. Selecao de features
# ---------------------------------------------------------------------------
def select_features(
    df: pd.DataFrame, target_col: str = "vendas"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features (X) da variavel alvo (y).

    Remove as colunas 'vendas', 'data' e 'estacao' (se ainda presente) do
    conjunto de features, mantendo apenas variaveis numericas e dummies.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame processado contendo as features e a variavel alvo.
    target_col : str
        Nome da coluna alvo. Padrao: 'vendas'.

    Retorna
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) onde X sao as features e y e a variavel alvo.
    """
    cols_to_drop = [target_col]

    if "data" in df.columns:
        cols_to_drop.append("data")
    if "estacao" in df.columns:
        cols_to_drop.append("estacao")

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    print(f"[select_features] Features selecionadas: {list(X.columns)}")
    print(f"                  X shape: {X.shape} | y shape: {y.shape}")

    return X, y


# ---------------------------------------------------------------------------
# 6. Pipeline completo
# ---------------------------------------------------------------------------
def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executa o pipeline completo de engenharia de features.

    Ordem de execucao:
      1. create_temperature_features
      2. create_temporal_features
      3. create_interaction_features
      4. encode_categorical

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame bruto com as colunas originais:
        data, temperatura, dia_da_semana, eh_feriado, estacao, vendas.

    Retorna
    -------
    pd.DataFrame
        DataFrame processado com todas as features adicionadas.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE DE ENGENHARIA DE FEATURES")
    print("=" * 60)

    df = create_temperature_features(df)
    df = create_temporal_features(df)
    df = create_interaction_features(df)
    df = encode_categorical(df)

    print(f"\n[build_feature_pipeline] Pipeline concluido.")
    print(f"    Formato final: {df.shape[0]} registros x {df.shape[1]} colunas")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Executa o pipeline de engenharia de features e salva os dados processados."""
    print("=" * 60)
    print("  GELATO MAGICO - Engenharia de Features")
    print("=" * 60)

    # 1. Carregar dados
    csv_path = INPUTS_DIR / "gelato_magico_vendas.csv"
    print(f"\n[main] Carregando dados de: {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"       {df.shape[0]} registros x {df.shape[1]} colunas carregados.")

    # 2. Executar pipeline de features
    df_processed = build_feature_pipeline(df)

    # 3. Selecionar features
    X, y = select_features(df_processed)

    # 4. Exibir resumo
    print("\n" + "-" * 60)
    print("  RESUMO DAS FEATURES")
    print("-" * 60)
    print(f"\nNomes das features ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"    {i:2d}. {col}")
    print(f"\nShape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    print(f"\nPrimeiras 5 linhas de X:")
    print(X.head().to_string())

    # 5. Salvar dados processados
    output_dir = OUTPUTS_DIR / "metricas"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "features_processadas.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"\n[main] Dados processados salvos em: {output_path}")

    print("\n" + "=" * 60)
    print("  Engenharia de features concluida com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
