"""
predict.py - Modulo de inferencia via CLI para o projeto Gelato Magico.

Permite realizar previsoes de vendas a partir de parametros informados
via linha de comando (temperatura, dia da semana, feriado, estacao).

Funcionalidades:
  1. Carregamento de modelo salvo com joblib.
  2. Preparacao de entrada com todas as features de engenharia.
  3. Predicao de vendas.
  4. Interface CLI com argparse.

Uso:
    python src/predict.py --temperatura 32 --dia_da_semana 5 --feriado --estacao verao
    python src/predict.py --temperatura 18
    python src/predict.py --help
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_PATH = OUTPUTS_DIR / "modelo_final" / "melhor_modelo.pkl"

# Features esperadas pelo modelo treinado (mesma ordem do treinamento)
EXPECTED_FEATURES = [
    "temperatura",
    "dia_da_semana",
    "eh_feriado",
    "temperatura_quadrada",
    "temperatura_cubica",
    "temperatura_normalizada",
    "eh_fim_de_semana",
    "mes",
    "dia_do_ano",
    "semana_do_ano",
    "temperatura_x_fds",
    "temperatura_x_feriado",
    "estacao_inverno",
    "estacao_outono",
    "estacao_primavera",
    "estacao_verao",
]


# ---------------------------------------------------------------------------
# 1. Carregamento do modelo
# ---------------------------------------------------------------------------
def load_model(filepath: str | Path):
    """Carrega um modelo treinado salvo em disco com joblib.

    Parametros
    ----------
    filepath : str | Path
        Caminho para o arquivo do modelo (.pkl).

    Retorna
    -------
    estimador sklearn
        Modelo treinado carregado do disco.

    Levanta
    -------
    FileNotFoundError
        Se o arquivo do modelo nao for encontrado.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {filepath}\n"
            f"Execute 'python src/model_training.py' primeiro para treinar e salvar o modelo."
        )

    model = joblib.load(filepath)
    print(f"[load_model] Modelo carregado de: {filepath}")

    return model


# ---------------------------------------------------------------------------
# 2. Preparacao da entrada
# ---------------------------------------------------------------------------
def prepare_input(
    temperatura: float,
    dia_da_semana: int = 2,
    eh_feriado: bool = False,
    estacao: str = "verao",
) -> pd.DataFrame:
    """Constroi um DataFrame de uma unica linha com todas as features de engenharia.

    Replica as transformacoes aplicadas em feature_engineering.py para
    garantir que a entrada tenha o mesmo formato esperado pelo modelo.

    Parametros
    ----------
    temperatura : float
        Temperatura em graus Celsius.
    dia_da_semana : int
        Dia da semana (0=segunda a 6=domingo). Padrao: 2 (quarta).
    eh_feriado : bool
        Se o dia eh feriado. Padrao: False.
    estacao : str
        Estacao do ano: 'verao', 'outono', 'inverno' ou 'primavera'.
        Padrao: 'verao'.

    Retorna
    -------
    pd.DataFrame
        DataFrame com uma unica linha contendo todas as features esperadas
        pelo modelo treinado.
    """
    feriado_int = int(eh_feriado)
    eh_fim_de_semana = int(dia_da_semana >= 5)

    # Features de temperatura
    temperatura_quadrada = temperatura ** 2
    temperatura_cubica = temperatura ** 3

    # Normalizacao z-score usando valores tipicos do dataset de Sao Paulo
    # Media e desvio padrao aproximados do dataset de treinamento
    temp_mean = 22.5
    temp_std = 5.5
    temperatura_normalizada = (temperatura - temp_mean) / temp_std

    # Features temporais aproximadas a partir da estacao
    # Usamos valores medios representativos de cada estacao
    estacao_to_month = {
        "verao": 1,
        "outono": 4,
        "inverno": 7,
        "primavera": 10,
    }
    mes = estacao_to_month.get(estacao, 1)

    # Dia do ano e semana do ano aproximados pelo mes
    hoje = datetime.date.today()
    data_aprox = datetime.date(hoje.year, mes, 15)
    dia_do_ano = data_aprox.timetuple().tm_yday
    semana_do_ano = data_aprox.isocalendar()[1]

    # Features de interacao
    temperatura_x_fds = temperatura * eh_fim_de_semana
    temperatura_x_feriado = temperatura * feriado_int

    # Codificacao one-hot da estacao
    estacoes = ["inverno", "outono", "primavera", "verao"]
    estacao_values = {f"estacao_{e}": int(estacao == e) for e in estacoes}

    # Montar dicionario com todas as features
    row = {
        "temperatura": temperatura,
        "dia_da_semana": dia_da_semana,
        "eh_feriado": feriado_int,
        "temperatura_quadrada": temperatura_quadrada,
        "temperatura_cubica": temperatura_cubica,
        "temperatura_normalizada": temperatura_normalizada,
        "eh_fim_de_semana": eh_fim_de_semana,
        "mes": mes,
        "dia_do_ano": dia_do_ano,
        "semana_do_ano": semana_do_ano,
        "temperatura_x_fds": temperatura_x_fds,
        "temperatura_x_feriado": temperatura_x_feriado,
    }
    row.update(estacao_values)

    input_df = pd.DataFrame([row])

    # Garantir a ordem correta das colunas
    input_df = input_df[EXPECTED_FEATURES]

    print(f"[prepare_input] Entrada preparada:")
    print(f"    Temperatura:      {temperatura} C")
    print(f"    Dia da semana:    {dia_da_semana} ({'fim de semana' if eh_fim_de_semana else 'dia util'})")
    print(f"    Feriado:          {'Sim' if eh_feriado else 'Nao'}")
    print(f"    Estacao:          {estacao}")

    return input_df


# ---------------------------------------------------------------------------
# 3. Predicao
# ---------------------------------------------------------------------------
def predict_sales(model, input_df: pd.DataFrame) -> float:
    """Realiza a predicao de vendas usando o modelo carregado.

    Parametros
    ----------
    model : estimador sklearn
        Modelo treinado (ou pipeline).
    input_df : pd.DataFrame
        DataFrame com uma unica linha contendo as features.

    Retorna
    -------
    float
        Valor previsto de vendas em reais (R$).
    """
    prediction = model.predict(input_df)
    valor = float(prediction[0])

    # Garantir que a predicao nao seja negativa
    valor = max(valor, 0.0)

    print(f"[predict_sales] Predicao: R$ {valor:.2f}")

    return valor


# ---------------------------------------------------------------------------
# 4. Interface CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Interface de linha de comando para previsao de vendas do Gelato Magico.

    Argumentos CLI:
      --temperatura   (float, obrigatorio): Temperatura em graus Celsius.
      --dia_da_semana (int, padrao 2):      Dia da semana (0=seg a 6=dom).
      --feriado       (flag):               Marcar o dia como feriado.
      --estacao       (str, padrao 'verao'): Estacao do ano.
      --modelo        (str, padrao caminho do melhor modelo salvo): Caminho do modelo.
    """
    parser = argparse.ArgumentParser(
        description="Gelato Magico - Previsao de Vendas de Sorvete",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos de uso:\n"
            "  python src/predict.py --temperatura 32 --estacao verao\n"
            "  python src/predict.py --temperatura 25 --dia_da_semana 5 --feriado\n"
            "  python src/predict.py --temperatura 18 --estacao inverno\n"
        ),
    )

    parser.add_argument(
        "--temperatura",
        type=float,
        required=True,
        help="Temperatura em graus Celsius (ex: 28.5)",
    )

    parser.add_argument(
        "--dia_da_semana",
        type=int,
        default=2,
        help="Dia da semana: 0=segunda a 6=domingo (padrao: 2 = quarta)",
    )

    parser.add_argument(
        "--feriado",
        action="store_true",
        default=False,
        help="Indica que o dia eh feriado",
    )

    parser.add_argument(
        "--estacao",
        type=str,
        default="verao",
        choices=["verao", "outono", "inverno", "primavera"],
        help="Estacao do ano (padrao: verao)",
    )

    parser.add_argument(
        "--modelo",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Caminho para o arquivo do modelo (padrao: {DEFAULT_MODEL_PATH})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GELATO MAGICO - Previsao de Vendas")
    print("=" * 60)

    # 1. Carregar modelo
    print(f"\n[main] Carregando modelo...")
    model = load_model(args.modelo)

    # 2. Preparar entrada
    print(f"\n[main] Preparando entrada...")
    input_df = prepare_input(
        temperatura=args.temperatura,
        dia_da_semana=args.dia_da_semana,
        eh_feriado=args.feriado,
        estacao=args.estacao,
    )

    # 3. Realizar predicao
    print(f"\n[main] Realizando predicao...")
    valor_previsto = predict_sales(model, input_df)

    # 4. Exibir resultado formatado
    dias_semana = {
        0: "Segunda-feira",
        1: "Terca-feira",
        2: "Quarta-feira",
        3: "Quinta-feira",
        4: "Sexta-feira",
        5: "Sabado",
        6: "Domingo",
    }

    estacoes_display = {
        "verao": "Verao",
        "outono": "Outono",
        "inverno": "Inverno",
        "primavera": "Primavera",
    }

    print("\n" + "=" * 60)
    print("  RESULTADO DA PREVISAO")
    print("=" * 60)
    print(f"\n  Condicoes informadas:")
    print(f"    Temperatura:    {args.temperatura:.1f} C")
    print(f"    Dia da semana:  {dias_semana.get(args.dia_da_semana, 'N/A')}")
    print(f"    Feriado:        {'Sim' if args.feriado else 'Nao'}")
    print(f"    Estacao:        {estacoes_display.get(args.estacao, args.estacao)}")
    print(f"\n  Vendas previstas: R$ {valor_previsto:,.2f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
