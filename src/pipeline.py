"""
pipeline.py - Pipeline end-to-end com sklearn para o projeto Gelato Magico.

Funcionalidades:
  1. Construcao de um ColumnTransformer para pre-processamento de features.
  2. Construcao de um Pipeline sklearn completo (pre-processamento + modelo).
  3. Execucao do pipeline para um modelo especifico com avaliacao.
  4. Comparacao completa de todos os modelos via pipeline.

Uso:
    python src/pipeline.py                  # executa a partir da raiz do projeto
    from src.pipeline import *              # importavel como modulo
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Features numericas que serao padronizadas com StandardScaler
NUMERIC_FEATURES = [
    "temperatura",
    "temperatura_quadrada",
    "temperatura_cubica",
    "dia_da_semana",
    "dia_do_ano",
    "mes",
    "semana_do_ano",
    "temperatura_x_fds",
    "temperatura_x_feriado",
]

# Features binarias que passam sem transformacao
BINARY_FEATURES = [
    "eh_feriado",
    "eh_fim_de_semana",
]

# Features de estacao (one-hot) que passam sem transformacao
SEASON_FEATURES = [
    "estacao_verao",
    "estacao_outono",
    "estacao_inverno",
    "estacao_primavera",
]


# ---------------------------------------------------------------------------
# 1. Pre-processador (ColumnTransformer)
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Retorna um ColumnTransformer para pre-processamento das features.

    Transformacoes aplicadas:
      - Features numericas (temperatura, polinomiais, temporais, interacoes):
        padronizacao com StandardScaler.
      - Features binarias (eh_feriado, eh_fim_de_semana): passthrough.
      - Features de estacao (dummies one-hot): passthrough.

    Retorna
    -------
    ColumnTransformer
        Transformador configurado com as tres etapas de pre-processamento.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("season", "passthrough", SEASON_FEATURES),
        ],
        remainder="drop",
    )

    print(f"[build_preprocessor] ColumnTransformer configurado:")
    print(f"    Numericas (StandardScaler): {len(NUMERIC_FEATURES)} features")
    print(f"    Binarias (passthrough):     {len(BINARY_FEATURES)} features")
    print(f"    Estacao (passthrough):      {len(SEASON_FEATURES)} features")

    return preprocessor


# ---------------------------------------------------------------------------
# 2. Pipeline completo (pre-processamento + modelo)
# ---------------------------------------------------------------------------
def build_pipeline(model) -> Pipeline:
    """Constroi um Pipeline sklearn com pre-processamento e modelo.

    Parametros
    ----------
    model : estimador sklearn
        Modelo de regressao (ou pipeline interno) a ser usado como etapa final.

    Retorna
    -------
    Pipeline
        Pipeline sklearn com duas etapas: 'preprocessor' e 'model'.
    """
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    print(f"[build_pipeline] Pipeline construido com modelo: {type(model).__name__}")

    return pipeline


# ---------------------------------------------------------------------------
# 3. Execucao do pipeline para um modelo
# ---------------------------------------------------------------------------
def run_pipeline(
    df: pd.DataFrame,
    model_name: str = "Gradient Boosting",
) -> dict:
    """Executa o pipeline completo para um modelo especifico.

    Etapas:
      1. Engenharia de features.
      2. Selecao de features e variavel alvo.
      3. Divisao treino/teste.
      4. Construcao do pipeline com o modelo escolhido.
      5. Treinamento.
      6. Avaliacao no conjunto de teste.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame bruto com as colunas originais do dataset.
    model_name : str
        Nome do modelo a ser utilizado, conforme definido em get_models().
        Padrao: 'Gradient Boosting'.

    Retorna
    -------
    dict
        Dicionario com as chaves:
          - model_name: nome do modelo
          - pipeline: pipeline treinado
          - metrics: dict de metricas (MAE, MSE, RMSE, R2, MAPE)
          - predictions: array de previsoes no teste
    """
    # Garantir imports do projeto
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_preparation import split_data
    from src.feature_engineering import build_feature_pipeline, select_features
    from src.model_training import get_models

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE - {model_name}")
    print(f"{'=' * 60}")

    # 1. Engenharia de features
    print("\n[run_pipeline] Executando engenharia de features...")
    df_processed = build_feature_pipeline(df)

    # 2. Selecao de features
    X, y = select_features(df_processed)

    # 3. Divisao treino/teste
    df_model = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = split_data(df_model, target_col="vendas")

    # 4. Selecionar modelo
    models = get_models()
    if model_name not in models:
        raise ValueError(
            f"Modelo '{model_name}' nao encontrado. "
            f"Modelos disponiveis: {list(models.keys())}"
        )
    model = models[model_name]

    # 5. Construir e treinar pipeline
    print(f"\n[run_pipeline] Construindo pipeline para '{model_name}'...")
    pipeline = build_pipeline(model)

    print(f"[run_pipeline] Treinando pipeline...")
    pipeline.fit(X_train, y_train)

    # 6. Avaliar
    y_pred = pipeline.predict(X_test)
    metrics = _calculate_pipeline_metrics(y_test, y_pred)

    print(f"\n[run_pipeline] Resultados para '{model_name}':")
    print(f"    MAE:  {metrics['MAE']:.2f}")
    print(f"    MSE:  {metrics['MSE']:.2f}")
    print(f"    RMSE: {metrics['RMSE']:.2f}")
    print(f"    R2:   {metrics['R2']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.2f}%")

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "predictions": y_pred,
    }


# ---------------------------------------------------------------------------
# 4. Comparacao completa de todos os modelos
# ---------------------------------------------------------------------------
def run_full_comparison_pipeline(df: pd.DataFrame) -> dict[str, dict]:
    """Executa o pipeline para todos os modelos e retorna resultados comparativos.

    Para cada modelo definido em get_models(), constroi um pipeline completo
    com pre-processamento, treina e avalia no conjunto de teste.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame bruto com as colunas originais do dataset.

    Retorna
    -------
    dict[str, dict]
        Dicionario mapeando nome do modelo -> dicionario com:
          - pipeline: pipeline treinado
          - metrics: dict de metricas (MAE, MSE, RMSE, R2, MAPE)
          - predictions: array de previsoes no teste
    """
    # Garantir imports do projeto
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_preparation import split_data
    from src.feature_engineering import build_feature_pipeline, select_features
    from src.model_training import get_models

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETO - COMPARACAO DE TODOS OS MODELOS")
    print("=" * 60)

    # 1. Engenharia de features
    print("\n[run_full_comparison_pipeline] Executando engenharia de features...")
    df_processed = build_feature_pipeline(df)

    # 2. Selecao de features
    X, y = select_features(df_processed)

    # 3. Divisao treino/teste
    df_model = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = split_data(df_model, target_col="vendas")

    # 4. Treinar e avaliar todos os modelos
    models = get_models()
    results: dict[str, dict] = {}

    for name, model in models.items():
        print(f"\n--- Pipeline: {name} ---")

        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = _calculate_pipeline_metrics(y_test, y_pred)

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "predictions": y_pred,
        }

        print(f"    MAE:  {metrics['MAE']:.2f}")
        print(f"    RMSE: {metrics['RMSE']:.2f}")
        print(f"    R2:   {metrics['R2']:.4f}")

    # 5. Tabela comparativa
    print("\n" + "=" * 70)
    print("  TABELA COMPARATIVA - PIPELINE COMPLETO")
    print("=" * 70)

    comparison_data = []
    for name, res in results.items():
        m = res["metrics"]
        comparison_data.append({
            "Modelo": name,
            "MAE": m["MAE"],
            "MSE": m["MSE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
            "MAPE (%)": m["MAPE"],
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values("MAE").reset_index(drop=True)
    df_comparison.index += 1
    df_comparison.index.name = "Rank"

    print(f"\n{df_comparison.to_string()}")

    # Melhor modelo
    best_name = df_comparison.iloc[0]["Modelo"]
    print(f"\n[run_full_comparison_pipeline] Melhor modelo: {best_name}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Funcao auxiliar de metricas
# ---------------------------------------------------------------------------
def _calculate_pipeline_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas de regressao entre valores reais e previstos.

    Parametros
    ----------
    y_true : array-like
        Valores reais da variavel alvo.
    y_pred : array-like
        Valores previstos pelo modelo.

    Retorna
    -------
    dict[str, float]
        Dicionario com MAE, MSE, RMSE, R2 e MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE — evitar divisao por zero
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    mask = y_true_arr != 0
    if mask.sum() > 0:
        mape = np.mean(
            np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])
        ) * 100
    else:
        mape = float("inf")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Executa o pipeline completo de comparacao de todos os modelos."""

    # Garantir que o projeto raiz esteja no sys.path
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_preparation import load_data

    print("=" * 60)
    print("  GELATO MAGICO - Pipeline End-to-End")
    print("=" * 60)

    # 1. Carregar dados
    csv_path = INPUTS_DIR / "gelato_magico_vendas.csv"
    print(f"\n[main] Carregando dados de: {csv_path.name}")
    df = load_data(csv_path)

    # 2. Executar pipeline completo com todos os modelos
    results = run_full_comparison_pipeline(df)

    # 3. Resumo final
    print("\n" + "=" * 60)
    print(f"  {len(results)} modelos avaliados via pipeline end-to-end.")
    print("  Pipeline concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
