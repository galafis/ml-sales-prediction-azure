"""
model_evaluation.py - Avaliacao e visualizacao de modelos para o projeto Gelato Magico.

Funcionalidades:
  1. Grafico de comparacao de MAE entre modelos (barras horizontais).
  2. Grafico de predicoes vs valores reais (dispersao).
  3. Grafico de residuos (dispersao + histograma).
  4. Grafico de importancia de features (barras horizontais, top 15).
  5. Curva de aprendizado (train/test scores vs tamanho do treino).
  6. Box plot de scores de validacao cruzada.
  7. Relatorio completo com todos os graficos e CSV de metricas.

Uso:
    python src/model_evaluation.py          # executa a partir da raiz do projeto
    from src.model_evaluation import *      # importavel como modulo
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# 1. Comparacao de modelos (barras horizontais por MAE)
# ---------------------------------------------------------------------------
def plot_model_comparison(results: dict[str, dict], output_dir: str | Path) -> None:
    """Gera grafico de barras horizontais comparando o MAE de todos os modelos.

    Os modelos sao ordenados pelo MAE em ordem crescente, de modo que o
    melhor modelo aparece no topo do grafico.

    Parametros
    ----------
    results : dict[str, dict]
        Dicionario retornado por train_all_models(), mapeando
        nome do modelo -> dict com chaves 'model', 'predictions', 'metrics'.
    output_dir : str | Path
        Diretorio onde o grafico sera salvo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extrair nomes e MAEs, ordenar por MAE ascendente
    model_names = []
    mae_values = []
    for name, res in results.items():
        model_names.append(name)
        mae_values.append(res["metrics"]["MAE"])

    # Ordenar por MAE ascendente (melhor no topo)
    sorted_pairs = sorted(zip(mae_values, model_names), reverse=True)
    mae_sorted = [pair[0] for pair in sorted_pairs]
    names_sorted = [pair[1] for pair in sorted_pairs]

    # Gerar paleta de cores
    palette = sns.color_palette("viridis", n_colors=len(names_sorted))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names_sorted, mae_sorted, color=palette, edgecolor="white", height=0.6)

    # Adicionar valores nas barras
    for bar, val in zip(bars, mae_sorted):
        ax.text(
            bar.get_width() + max(mae_sorted) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_title("Comparacao de Modelos - Erro Absoluto Medio (MAE)", fontsize=16, fontweight="bold")
    ax.set_xlabel("MAE (Erro Absoluto Medio)", fontsize=12)
    ax.set_ylabel("Modelo", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()

    filepath = output_dir / "comparacao_modelos.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_model_comparison] Grafico salvo em: {filepath.name}")


# ---------------------------------------------------------------------------
# 2. Predicoes vs Valores Reais (dispersao)
# ---------------------------------------------------------------------------
def plot_predictions_vs_actual(
    y_test: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Gera grafico de dispersao de predicoes vs valores reais.

    Inclui a linha de predicao perfeita (y = x) como referencia visual.

    Parametros
    ----------
    y_test : array-like
        Valores reais da variavel alvo no conjunto de teste.
    y_pred : array-like
        Valores previstos pelo modelo.
    model_name : str
        Nome do modelo (usado no titulo e no nome do arquivo).
    output_dir : str | Path
        Diretorio onde o grafico sera salvo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_test_arr, y_pred_arr,
        alpha=0.6, color="#2196F3", edgecolor="white", s=50,
        label="Predicoes",
    )

    # Linha de predicao perfeita (y = x)
    lim_min = min(y_test_arr.min(), y_pred_arr.min()) * 0.9
    lim_max = max(y_test_arr.max(), y_pred_arr.max()) * 1.1
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "--", color="#E53935", linewidth=2, label="Predicao Perfeita (y = x)",
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title(f"Predicoes vs Valores Reais - {model_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Valores Reais (Vendas)", fontsize=12)
    ax.set_ylabel("Valores Previstos (Vendas)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    filepath = output_dir / f"predicoes_vs_real_{safe_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_predictions_vs_actual] Grafico salvo em: {filepath.name}")


# ---------------------------------------------------------------------------
# 3. Residuos (dispersao + histograma)
# ---------------------------------------------------------------------------
def plot_residuals(
    y_test: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Gera grafico de residuos com dois subplots.

    Subplot esquerdo: dispersao dos residuos vs valores previstos.
    Subplot direito: histograma dos residuos com curva KDE.

    Parametros
    ----------
    y_test : array-like
        Valores reais da variavel alvo no conjunto de teste.
    y_pred : array-like
        Valores previstos pelo modelo.
    model_name : str
        Nome do modelo (usado no titulo e no nome do arquivo).
    output_dir : str | Path
        Diretorio onde o grafico sera salvo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    residuos = y_test_arr - y_pred_arr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Residuos vs Valores Previstos ---
    ax1.scatter(
        y_pred_arr, residuos,
        alpha=0.6, color="#FF9800", edgecolor="white", s=50,
    )
    ax1.axhline(y=0, color="#E53935", linestyle="--", linewidth=2)
    ax1.set_title(f"Residuos vs Valores Previstos", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Valores Previstos (Vendas)", fontsize=12)
    ax1.set_ylabel("Residuos", fontsize=12)

    # --- Subplot 2: Histograma dos Residuos ---
    sns.histplot(residuos, bins=30, kde=True, color="#4CAF50", ax=ax2)
    ax2.axvline(x=0, color="#E53935", linestyle="--", linewidth=2)
    ax2.set_title(f"Distribuicao dos Residuos", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Residuo", fontsize=12)
    ax2.set_ylabel("Frequencia", fontsize=12)

    fig.suptitle(f"Analise de Residuos - {model_name}", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    filepath = output_dir / f"residuos_{safe_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_residuals] Grafico salvo em: {filepath.name}")


# ---------------------------------------------------------------------------
# 4. Importancia de Features (barras horizontais, top 15)
# ---------------------------------------------------------------------------
def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Gera grafico de barras horizontais com a importancia das features.

    Funciona apenas para modelos baseados em arvore que possuem o atributo
    ``feature_importances_``. Para pipelines, tenta acessar o atributo no
    ultimo step. Se o atributo nao existir, o grafico nao e gerado.

    Sao exibidas as 15 features mais importantes.

    Parametros
    ----------
    model : estimador sklearn
        Modelo treinado (ou pipeline).
    feature_names : list[str]
        Lista com os nomes das features.
    model_name : str
        Nome do modelo (usado no titulo e no nome do arquivo).
    output_dir : str | Path
        Diretorio onde o grafico sera salvo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tentar obter feature_importances_ diretamente ou do ultimo step do pipeline
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline: verificar ultimo step
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "feature_importances_"):
            importances = last_step.feature_importances_

    if importances is None:
        print(f"[plot_feature_importance] Modelo '{model_name}' nao possui "
              f"atributo feature_importances_. Grafico nao gerado.")
        return

    # Criar DataFrame e ordenar
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importancia": importances,
    })
    df_imp = df_imp.sort_values("Importancia", ascending=True)

    # Selecionar top 15
    df_top = df_imp.tail(15)

    palette = sns.color_palette("viridis", n_colors=len(df_top))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        df_top["Feature"], df_top["Importancia"],
        color=palette, edgecolor="white", height=0.6,
    )

    ax.set_title(f"Importancia das Features - {model_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Importancia", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    filepath = output_dir / f"importancia_features_{safe_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_feature_importance] Grafico salvo em: {filepath.name}")


# ---------------------------------------------------------------------------
# 5. Curva de Aprendizado
# ---------------------------------------------------------------------------
def plot_learning_curve(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_name: str,
    output_dir: str | Path,
    cv: int = 5,
) -> None:
    """Gera a curva de aprendizado de um modelo.

    Utiliza sklearn.model_selection.learning_curve para calcular os scores
    de treino e validacao em funcao do tamanho do conjunto de treino.

    Parametros
    ----------
    model : estimador sklearn
        Modelo ou pipeline sklearn (nao precisa estar treinado).
    X : pd.DataFrame | np.ndarray
        Features de treino.
    y : pd.Series | np.ndarray
        Variavel alvo de treino.
    model_name : str
        Nome do modelo (usado no titulo e no nome do arquivo).
    output_dir : str | Path
        Diretorio onde o grafico sera salvo.
    cv : int
        Numero de folds para validacao cruzada. Padrao: 5.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_learning_curve] Calculando curva de aprendizado para '{model_name}'...")

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    # Converter scores negativos para positivos (MAE)
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Faixa de desvio padrao - Treino
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2, color="#2196F3",
    )
    ax.plot(
        train_sizes, train_scores_mean,
        "o-", color="#2196F3", linewidth=2, markersize=6,
        label="Score de Treino (MAE)",
    )

    # Faixa de desvio padrao - Validacao
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2, color="#E53935",
    )
    ax.plot(
        train_sizes, test_scores_mean,
        "o-", color="#E53935", linewidth=2, markersize=6,
        label="Score de Validacao (MAE)",
    )

    ax.set_title(f"Curva de Aprendizado - {model_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Tamanho do Conjunto de Treino", fontsize=12)
    ax.set_ylabel("MAE (Erro Absoluto Medio)", fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    filepath = output_dir / f"curva_aprendizado_{safe_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_learning_curve] Grafico salvo em: {filepath.name}")


# ---------------------------------------------------------------------------
# 6. Box Plot de Scores de Validacao Cruzada
# ---------------------------------------------------------------------------
def plot_cross_validation_scores(
    cv_results: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """Gera box plot dos scores de validacao cruzada (MAE) para todos os modelos.

    Parametros
    ----------
    cv_results : dict[str, np.ndarray]
        Dicionario mapeando nome do modelo -> array de scores MAE por fold.
    output_dir : str | Path
        Diretorio onde o grafico sera salvo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construir DataFrame longo para seaborn
    records = []
    for name, scores in cv_results.items():
        for score in scores:
            records.append({"Modelo": name, "MAE": score})

    df_cv = pd.DataFrame(records)

    # Ordenar modelos pela mediana do MAE
    ordem = (
        df_cv.groupby("Modelo")["MAE"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_cv, x="Modelo", y="MAE",
        order=ordem, palette="viridis", ax=ax,
    )

    ax.set_title("Validacao Cruzada - Distribuicao do MAE por Modelo", fontsize=16, fontweight="bold")
    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("MAE (Erro Absoluto Medio)", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    fig.tight_layout()

    filepath = output_dir / "cv_scores.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_cross_validation_scores] Grafico salvo em: {filepath.name}")


# ---------------------------------------------------------------------------
# 7. Relatorio Completo
# ---------------------------------------------------------------------------
def generate_full_report(
    results: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    output_dir: str | Path,
) -> None:
    """Gera todos os graficos de avaliacao e salva um CSV com as metricas.

    Para cada modelo em ``results``:
      - Grafico de predicoes vs valores reais.
      - Grafico de residuos.
      - Grafico de importancia de features (se aplicavel).
      - Curva de aprendizado.

    Alem disso:
      - Grafico de comparacao de MAE entre modelos.
      - Box plot de validacao cruzada.
      - CSV com metricas de todos os modelos.

    Parametros
    ----------
    results : dict[str, dict]
        Dicionario retornado por train_all_models(), mapeando
        nome do modelo -> dict com chaves 'model', 'predictions', 'metrics'.
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Variavel alvo de treino.
    X_test : pd.DataFrame
        Features de teste.
    y_test : pd.Series
        Variavel alvo de teste.
    feature_names : list[str]
        Lista com os nomes das features.
    output_dir : str | Path
        Diretorio base para salvar graficos (subdiretorio 'graficos') e
        metricas (subdiretorio 'metricas').
    """
    output_dir = Path(output_dir)
    graficos_dir = output_dir / "graficos"
    metricas_dir = output_dir / "metricas"
    graficos_dir.mkdir(parents=True, exist_ok=True)
    metricas_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  GERACAO DO RELATORIO COMPLETO DE AVALIACAO")
    print("=" * 60)

    # --- 1. Comparacao de modelos ---
    print("\n--- 1. Comparacao de modelos (MAE) ---")
    plot_model_comparison(results, graficos_dir)

    # --- 2. Graficos por modelo ---
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    cv_results: dict[str, np.ndarray] = {}

    for name, res in results.items():
        model = res["model"]
        y_pred = res["predictions"]

        print(f"\n--- Modelo: {name} ---")

        # Predicoes vs Real
        print("    Gerando grafico de predicoes vs real...")
        plot_predictions_vs_actual(y_test, y_pred, name, graficos_dir)

        # Residuos
        print("    Gerando grafico de residuos...")
        plot_residuals(y_test, y_pred, name, graficos_dir)

        # Importancia de features
        print("    Verificando importancia de features...")
        plot_feature_importance(model, feature_names, name, graficos_dir)

        # Curva de aprendizado
        print("    Gerando curva de aprendizado...")
        plot_learning_curve(model, X_full, y_full, name, graficos_dir)

        # Validacao cruzada para box plot
        print("    Calculando validacao cruzada...")
        mae_scores = -cross_val_score(
            model, X_full, y_full, cv=5,
            scoring="neg_mean_absolute_error",
        )
        cv_results[name] = mae_scores

    # --- 3. Box plot de validacao cruzada ---
    print("\n--- Box plot de validacao cruzada ---")
    plot_cross_validation_scores(cv_results, graficos_dir)

    # --- 4. Salvar metricas em CSV ---
    print("\n--- Salvando metricas em CSV ---")
    comparison_data = []
    for name, res in results.items():
        m = res["metrics"]
        row = {"Modelo": name}
        row.update(m)
        comparison_data.append(row)

    df_metrics = pd.DataFrame(comparison_data)
    df_metrics = df_metrics.sort_values("MAE").reset_index(drop=True)
    df_metrics.index += 1
    df_metrics.index.name = "Rank"

    csv_path = metricas_dir / "resultados_modelos.csv"
    df_metrics.to_csv(csv_path)
    print(f"[generate_full_report] Metricas salvas em: {csv_path}")

    print("\n" + "=" * 60)
    print("  RELATORIO COMPLETO GERADO COM SUCESSO!")
    print(f"  Graficos: {graficos_dir}")
    print(f"  Metricas: {metricas_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Executa o pipeline completo de avaliacao: carrega dados, features, treina e gera relatorio."""

    # Garantir que o projeto raiz esteja no sys.path para imports
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_preparation import load_data, split_data
    from src.feature_engineering import build_feature_pipeline, select_features
    from src.model_training import train_all_models

    print("=" * 60)
    print("  GELATO MAGICO - Avaliacao de Modelos")
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
    feature_names = list(X.columns)

    # 4. Divisao treino/teste
    df_model = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = split_data(df_model, target_col="vendas")

    # 5. Treinar todos os modelos
    print("\n[main] Treinando modelos...")
    results = train_all_models(X_train, y_train, X_test, y_test)

    # 6. Gerar relatorio completo
    print("\n[main] Gerando relatorio completo de avaliacao...")
    generate_full_report(
        results=results,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=OUTPUTS_DIR,
    )

    print("\n" + "=" * 60)
    print("  Pipeline de avaliacao concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
