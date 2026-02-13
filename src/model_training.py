"""
model_training.py - Treinamento e comparacao de modelos para o projeto Gelato Magico.

Funcionalidades:
  1. Definicao de 8 modelos de regressao (linear, polinomial, Ridge, Lasso, RF, GB, SVR).
  2. Treinamento individual de modelos.
  3. Validacao cruzada com metricas MAE, RMSE e R2.
  4. Treinamento e avaliacao de todos os modelos no conjunto de teste.
  5. Selecao do melhor modelo com base no MAE.
  6. Salvamento de modelos com joblib.

Uso:
    python src/model_training.py          # executa a partir da raiz do projeto
    from src.model_training import *      # importavel como modulo
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# 1. Definicao dos modelos
# ---------------------------------------------------------------------------
def get_models() -> OrderedDict:
    """Retorna um OrderedDict com os modelos de regressao a serem comparados.

    Os modelos incluidos sao:
      - Regressao Linear
      - Polinomial Grau 2
      - Polinomial Grau 3
      - Ridge
      - Lasso
      - Random Forest
      - Gradient Boosting
      - SVR (com escalonamento)

    Retorna
    -------
    OrderedDict
        Dicionario ordenado mapeando nome do modelo -> estimador sklearn.
    """
    models = OrderedDict()

    models["Regressao Linear"] = LinearRegression()

    models["Polinomial Grau 2"] = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression()),
    ])

    models["Polinomial Grau 3"] = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("lr", LinearRegression()),
    ])

    models["Ridge"] = Ridge(alpha=1.0)

    models["Lasso"] = Lasso(alpha=1.0, max_iter=10000)

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100, random_state=42,
    )

    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=100, random_state=42,
    )

    models["SVR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=100, epsilon=0.1)),
    ])

    print(f"[get_models] {len(models)} modelos definidos: {list(models.keys())}")

    return models


# ---------------------------------------------------------------------------
# 2. Treinamento individual
# ---------------------------------------------------------------------------
def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Treina um modelo de regressao nos dados de treino.

    Parametros
    ----------
    model : estimador sklearn
        Modelo ou pipeline sklearn a ser treinado.
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Variavel alvo de treino.

    Retorna
    -------
    estimador sklearn
        O modelo treinado (fitted).
    """
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Validacao cruzada
# ---------------------------------------------------------------------------
def cross_validate_model(
    model, X: pd.DataFrame, y: pd.Series, cv: int = 5,
) -> dict[str, float]:
    """Realiza validacao cruzada e retorna metricas agregadas.

    Utiliza cross_val_score com scoring='neg_mean_absolute_error' para
    calcular o MAE medio e desvio padrao. Tambem calcula RMSE e R2 por
    validacao cruzada.

    Parametros
    ----------
    model : estimador sklearn
        Modelo ou pipeline sklearn (nao precisa estar treinado).
    X : pd.DataFrame
        Features completas (treino + validacao).
    y : pd.Series
        Variavel alvo completa.
    cv : int
        Numero de folds para validacao cruzada. Padrao: 5.

    Retorna
    -------
    dict[str, float]
        Dicionario com as chaves:
          - cv_mae_mean: media do MAE nos folds
          - cv_mae_std: desvio padrao do MAE nos folds
          - cv_rmse_mean: media do RMSE nos folds
          - cv_rmse_std: desvio padrao do RMSE nos folds
          - cv_r2_mean: media do R2 nos folds
          - cv_r2_std: desvio padrao do R2 nos folds
    """
    # MAE (neg_mean_absolute_error retorna valores negativos)
    mae_scores = -cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error",
    )

    # RMSE (neg_mean_squared_error retorna valores negativos)
    mse_scores = -cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_squared_error",
    )
    rmse_scores = np.sqrt(mse_scores)

    # R2
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    return {
        "cv_mae_mean": mae_scores.mean(),
        "cv_mae_std": mae_scores.std(),
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "cv_r2_mean": r2_scores.mean(),
        "cv_r2_std": r2_scores.std(),
    }


# ---------------------------------------------------------------------------
# 4. Metricas de avaliacao
# ---------------------------------------------------------------------------
def _calculate_metrics(
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
        mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
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
# 5. Treinar e avaliar todos os modelos
# ---------------------------------------------------------------------------
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """Treina todos os modelos e avalia no conjunto de teste.

    Para cada modelo definido em get_models():
      1. Treina no conjunto de treino.
      2. Gera previsoes no conjunto de teste.
      3. Calcula metricas de avaliacao (MAE, MSE, RMSE, R2, MAPE).

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

    Retorna
    -------
    dict[str, dict]
        Dicionario mapeando nome do modelo -> dicionario com:
          - model: estimador treinado
          - predictions: array de previsoes no teste
          - metrics: dict de metricas (MAE, MSE, RMSE, R2, MAPE)
    """
    models = get_models()
    results: dict[str, dict] = {}

    print("\n" + "=" * 60)
    print("  TREINAMENTO E AVALIACAO DE MODELOS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Treinar
        trained_model = train_model(model, X_train, y_train)

        # Prever
        y_pred = trained_model.predict(X_test)

        # Metricas
        metrics = _calculate_metrics(y_test, y_pred)

        results[name] = {
            "model": trained_model,
            "predictions": y_pred,
            "metrics": metrics,
        }

        print(f"    MAE:  {metrics['MAE']:.2f}")
        print(f"    MSE:  {metrics['MSE']:.2f}")
        print(f"    RMSE: {metrics['RMSE']:.2f}")
        print(f"    R2:   {metrics['R2']:.4f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")

    print("\n" + "=" * 60)
    print(f"  {len(results)} modelos treinados e avaliados com sucesso.")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# 6. Selecao do melhor modelo
# ---------------------------------------------------------------------------
def find_best_model(results: dict[str, dict]) -> str:
    """Identifica o melhor modelo com base no menor MAE no conjunto de teste.

    Parametros
    ----------
    results : dict[str, dict]
        Dicionario retornado por train_all_models().

    Retorna
    -------
    str
        Nome do modelo com o menor MAE.
    """
    best_name = min(results, key=lambda name: results[name]["metrics"]["MAE"])

    best_metrics = results[best_name]["metrics"]
    print(f"\n[find_best_model] Melhor modelo: {best_name}")
    print(f"    MAE:  {best_metrics['MAE']:.2f}")
    print(f"    RMSE: {best_metrics['RMSE']:.2f}")
    print(f"    R2:   {best_metrics['R2']:.4f}")
    print(f"    MAPE: {best_metrics['MAPE']:.2f}%")

    return best_name


# ---------------------------------------------------------------------------
# 7. Salvamento de modelo
# ---------------------------------------------------------------------------
def save_model(model, filepath: str | Path) -> None:
    """Salva um modelo treinado em disco usando joblib.

    Parametros
    ----------
    model : estimador sklearn
        Modelo treinado a ser salvo.
    filepath : str | Path
        Caminho completo do arquivo de destino (.pkl).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"[save_model] Modelo salvo em: {filepath}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Executa o pipeline completo de treinamento, comparacao e salvamento do melhor modelo."""

    # Garantir que o projeto raiz esteja no sys.path para imports
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_preparation import load_data, split_data
    from src.feature_engineering import build_feature_pipeline, select_features

    print("=" * 60)
    print("  GELATO MAGICO - Treinamento de Modelos")
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
    # Reconstituir DataFrame completo para usar split_data
    df_model = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = split_data(df_model, target_col="vendas")

    # 5. Treinar e avaliar todos os modelos
    results = train_all_models(X_train, y_train, X_test, y_test)

    # 6. Tabela comparativa
    print("\n" + "=" * 70)
    print("  TABELA COMPARATIVA DE MODELOS")
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
    df_comparison.index += 1  # Ranking comeca em 1
    df_comparison.index.name = "Rank"

    print(f"\n{df_comparison.to_string()}")

    # 7. Selecionar melhor modelo
    best_name = find_best_model(results)

    # 8. Salvar melhor modelo
    model_dir = OUTPUTS_DIR / "modelo_final"
    model_path = model_dir / "melhor_modelo.pkl"
    save_model(results[best_name]["model"], model_path)

    # 9. Salvar tabela comparativa em CSV
    metrics_dir = OUTPUTS_DIR / "metricas"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = metrics_dir / "comparacao_modelos.csv"
    df_comparison.to_csv(comparison_path)
    print(f"[main] Tabela comparativa salva em: {comparison_path}")

    print("\n" + "=" * 60)
    print(f"  Melhor modelo: {best_name}")
    print(f"  Salvo em: {model_path}")
    print("  Pipeline de treinamento concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
