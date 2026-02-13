"""
mlflow_tracking.py - Integracao com MLflow para rastreamento de experimentos no projeto Gelato Magico.

Funcionalidades:
  1. Configuracao de experimentos MLflow.
  2. Registro de execucoes individuais de modelos (parametros, metricas, artefatos).
  3. Registro de todos os 8 modelos em um unico experimento.
  4. Consulta da melhor execucao com base no menor MAE.
  5. Registro do melhor modelo no MLflow Model Registry.

Uso:
    python src/mlflow_tracking.py              # executa a partir da raiz do projeto
    from src.mlflow_tracking import *          # importavel como modulo
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_EXPERIMENT_NAME = "gelato-magico-vendas"


# ---------------------------------------------------------------------------
# 1. Configuracao do experimento
# ---------------------------------------------------------------------------
def setup_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> mlflow.entities.Experiment:
    """Configura e retorna um experimento MLflow.

    Define o URI de rastreamento local (file:./mlruns) e cria ou recupera
    o experimento com o nome especificado.

    Parametros
    ----------
    experiment_name : str
        Nome do experimento MLflow. Padrao: 'gelato-magico-vendas'.

    Retorna
    -------
    mlflow.entities.Experiment
        Objeto do experimento MLflow configurado.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    print(f"[setup_experiment] Experimento configurado:")
    print(f"    Nome:          {experiment.name}")
    print(f"    ID:            {experiment.experiment_id}")
    print(f"    Tracking URI:  {mlflow.get_tracking_uri()}")

    return experiment


# ---------------------------------------------------------------------------
# 2. Registro de uma execucao de modelo
# ---------------------------------------------------------------------------
def log_model_run(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> str:
    """Executa e registra um modelo no MLflow.

    Realiza o treinamento do modelo, calcula metricas de avaliacao e
    registra tudo em uma execucao (run) do MLflow, incluindo:
      - Parametros do modelo (nome, hiperparametros via get_params())
      - Metricas (MAE, MSE, RMSE, R2, MAPE)
      - Artefato do modelo (mlflow.sklearn.log_model)

    Parametros
    ----------
    model : estimador sklearn
        Modelo ou pipeline sklearn (nao treinado).
    model_name : str
        Nome descritivo do modelo para identificacao no MLflow.
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Variavel alvo de treino.
    X_test : pd.DataFrame
        Features de teste.
    y_test : pd.Series
        Variavel alvo de teste.
    params : dict, opcional
        Parametros adicionais para registrar no MLflow. Se None, utiliza
        apenas model.get_params().

    Retorna
    -------
    str
        ID da execucao (run_id) registrada no MLflow.
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # --- Parametros ---
        mlflow.log_param("model_name", model_name)

        # Hiperparametros do modelo
        try:
            model_params = model.get_params()
            for key, value in model_params.items():
                # MLflow nao aceita valores None ou objetos complexos
                param_value = str(value) if value is not None else "None"
                # Truncar valores muito longos
                if len(param_value) > 250:
                    param_value = param_value[:250]
                mlflow.log_param(key, param_value)
        except Exception:
            print(f"    [aviso] Nao foi possivel extrair todos os hiperparametros.")

        # Parametros adicionais fornecidos pelo usuario
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # --- Treinamento ---
        print(f"    Treinando '{model_name}'...")
        model.fit(X_train, y_train)

        # --- Predicoes ---
        y_pred = model.predict(X_test)

        # --- Metricas ---
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # MAPE — evitar divisao por zero
        y_true_arr = np.array(y_test, dtype=float)
        y_pred_arr = np.array(y_pred, dtype=float)
        mask = y_true_arr != 0
        if mask.sum() > 0:
            mape = np.mean(
                np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])
            ) * 100
        else:
            mape = float("inf")

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)

        # --- Artefato do modelo ---
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"    [log_model_run] Run ID: {run_id}")
        print(f"        MAE:  {mae:.2f}")
        print(f"        MSE:  {mse:.2f}")
        print(f"        RMSE: {rmse:.2f}")
        print(f"        R2:   {r2:.4f}")
        print(f"        MAPE: {mape:.2f}%")

    return run_id


# ---------------------------------------------------------------------------
# 3. Registro de todos os modelos
# ---------------------------------------------------------------------------
def log_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, str]:
    """Registra todos os 8 modelos em um experimento MLflow.

    Utiliza get_models() do modulo model_training para obter a lista
    de modelos e registra cada um em uma execucao separada.

    Parametros
    ----------
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Variavel alvo de treino.
    X_test : pd.DataFrame
        Features de teste.
    y_test : pd.Series
        Variavel alvo de teste.
    experiment_name : str
        Nome do experimento MLflow. Padrao: 'gelato-magico-vendas'.

    Retorna
    -------
    dict[str, str]
        Dicionario mapeando nome do modelo -> run_id.
    """
    # Garantir imports do projeto
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.model_training import get_models

    # Configurar experimento
    setup_experiment(experiment_name)

    models = get_models()
    run_ids: dict[str, str] = {}

    print("\n" + "=" * 60)
    print("  MLFLOW - REGISTRO DE TODOS OS MODELOS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        run_id = log_model_run(
            model=model,
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        run_ids[name] = run_id

    print("\n" + "=" * 60)
    print(f"  {len(run_ids)} modelos registrados no MLflow.")
    print("=" * 60)

    return run_ids


# ---------------------------------------------------------------------------
# 4. Consulta da melhor execucao
# ---------------------------------------------------------------------------
def get_best_run(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> mlflow.entities.Run:
    """Consulta o MLflow e retorna a execucao com o menor MAE.

    Parametros
    ----------
    experiment_name : str
        Nome do experimento MLflow. Padrao: 'gelato-magico-vendas'.

    Retorna
    -------
    mlflow.entities.Run
        Objeto Run do MLflow correspondente a melhor execucao.

    Levanta
    -------
    ValueError
        Se nenhum experimento ou execucao for encontrado.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experimento '{experiment_name}' nao encontrado. "
            f"Execute log_all_models() primeiro."
        )

    # Buscar todas as execucoes ordenadas por MAE ascendente
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.MAE ASC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"Nenhuma execucao encontrada no experimento '{experiment_name}'."
        )

    best_run_id = runs.iloc[0]["run_id"]
    best_run = mlflow.get_run(best_run_id)

    # Extrair informacoes
    best_model_name = best_run.data.params.get("model_name", "N/A")
    best_mae = best_run.data.metrics.get("MAE", float("inf"))
    best_rmse = best_run.data.metrics.get("RMSE", float("inf"))
    best_r2 = best_run.data.metrics.get("R2", 0.0)

    print(f"\n[get_best_run] Melhor execucao encontrada:")
    print(f"    Run ID:  {best_run_id}")
    print(f"    Modelo:  {best_model_name}")
    print(f"    MAE:     {best_mae:.2f}")
    print(f"    RMSE:    {best_rmse:.2f}")
    print(f"    R2:      {best_r2:.4f}")

    return best_run


# ---------------------------------------------------------------------------
# 5. Registro do melhor modelo no Model Registry
# ---------------------------------------------------------------------------
def register_best_model(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str = "gelato-magico-best-model",
) -> mlflow.entities.model_registry.ModelVersion:
    """Registra o melhor modelo no MLflow Model Registry.

    Busca a execucao com o menor MAE no experimento e registra o
    artefato do modelo no Model Registry com o nome especificado.

    Parametros
    ----------
    experiment_name : str
        Nome do experimento MLflow. Padrao: 'gelato-magico-vendas'.
    model_name : str
        Nome para registrar o modelo no Model Registry.
        Padrao: 'gelato-magico-best-model'.

    Retorna
    -------
    mlflow.entities.model_registry.ModelVersion
        Objeto ModelVersion com informacoes do modelo registrado.
    """
    best_run = get_best_run(experiment_name)
    run_id = best_run.info.run_id

    # URI do artefato do modelo
    model_uri = f"runs:/{run_id}/model"

    print(f"\n[register_best_model] Registrando modelo no Model Registry...")
    print(f"    Run ID:     {run_id}")
    print(f"    Model URI:  {model_uri}")
    print(f"    Nome:       {model_name}")

    # Registrar no Model Registry
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(f"    Versao:     {model_version.version}")
    print(f"    Status:     {model_version.status}")
    print(f"\n[register_best_model] Modelo registrado com sucesso!")

    return model_version


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Executa o pipeline completo de rastreamento MLflow.

    Carrega os dados, realiza engenharia de features, divide em treino/teste,
    registra todos os modelos no MLflow e exibe a melhor execucao.
    """
    # Garantir que o projeto raiz esteja no sys.path
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_preparation import load_data, split_data
    from src.feature_engineering import build_feature_pipeline, select_features

    print("=" * 60)
    print("  GELATO MAGICO - MLflow Tracking")
    print("=" * 60)

    # 1. Carregar dados
    csv_path = INPUTS_DIR / "gelato_magico_vendas.csv"
    print(f"\n[main] Carregando dados de: {csv_path.name}")
    df = load_data(csv_path)

    # 2. Engenharia de features
    print("\n[main] Executando engenharia de features...")
    df_processed = build_feature_pipeline(df)

    # 3. Selecao de features e variavel alvo
    X, y = select_features(df_processed)

    # 4. Divisao treino/teste
    df_model = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = split_data(df_model, target_col="vendas")

    # 5. Registrar todos os modelos no MLflow
    print("\n[main] Registrando modelos no MLflow...")
    run_ids = log_all_models(X_train, y_train, X_test, y_test)

    # 6. Buscar melhor execucao
    print("\n[main] Buscando melhor execucao...")
    best_run = get_best_run()

    best_model_name = best_run.data.params.get("model_name", "N/A")
    best_mae = best_run.data.metrics.get("MAE", float("inf"))

    # 7. Resumo final
    print("\n" + "=" * 60)
    print("  RESUMO DO RASTREAMENTO MLFLOW")
    print("=" * 60)
    print(f"\n  Modelos registrados:  {len(run_ids)}")
    print(f"  Melhor modelo:        {best_model_name}")
    print(f"  Melhor MAE:           {best_mae:.2f}")
    print(f"  Run ID:               {best_run.info.run_id}")
    print(f"\n  Para visualizar os resultados, execute:")
    print(f"    mlflow ui --backend-store-uri file:./mlruns")
    print("\n" + "=" * 60)
    print("  Pipeline MLflow concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
