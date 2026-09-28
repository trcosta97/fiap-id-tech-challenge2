"""
Carregamento e pré-processamento do dataset Breast Cancer.

Reproduz exatamente o pipeline do Módulo 1 (fiap-id-tech-challenge1),
garantindo rastreabilidade e comparabilidade dos resultados.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Caminho padrão do dataset (relativo à raiz do projeto)
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CSV = _DATA_DIR / "Breast_Cancer.csv"

# Colunas conforme definidas no Módulo 1
NUMERIC_COLS: list[str] = [
    "Age",
    "Tumor Size",
    "Regional Node Examined",
    "Reginol Node Positive",
]

CATEGORICAL_COLS: list[str] = [
    "Race",
    "Marital Status",
    "T Stage ",   # nota: o CSV tem espaço trailing nesta coluna
    "N Stage",
    "6th Stage",
    "differentiate",
    "Grade",
    "A Stage",
    "Estrogen Status",
    "Progesterone Status",
]

TARGET_COL = "Status"
DROP_COLS = ["Survival Months"]  # data leakage – não disponível no diagnóstico


def load_raw(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """
    Carrega o CSV bruto e valida colunas mínimas esperadas.

    Parâmetros
    ----------
    csv_path : Path
        Caminho para o arquivo Breast_Cancer.csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame bruto sem qualquer transformação.
    """
    logger.info(f"Carregando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Shape: {df.shape} | Colunas: {list(df.columns)}")

    expected = set(NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Monta o ColumnTransformer idêntico ao do Módulo 1.

    - Numéricas  → StandardScaler
    - Categóricas → OrdinalEncoder (handle_unknown='use_encoded_value')

    Retorna
    -------
    ColumnTransformer
        Pré-processador ainda não ajustado.
    """
    num_transformer = StandardScaler()
    cat_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_COLS),
            ("cat", cat_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(
    csv_path: Path = DEFAULT_CSV,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Pipeline completo: carrega, remove colunas de leakage, codifica o alvo
    e divide em treino/teste estratificado.

    Parâmetros
    ----------
    csv_path : Path
        Caminho do CSV.
    test_size : float
        Proporção do conjunto de teste (padrão: 0.2).
    random_state : int
        Seed para reprodutibilidade (padrão: 42).

    Retorna
    -------
    X_train, X_test : pd.DataFrame
        Features de treino e teste.
    y_train, y_test : np.ndarray
        Rótulos codificados (0 = Alive, 1 = Dead).
    le : LabelEncoder
        Encoder treinado (para recuperar os nomes das classes).
    """
    df = load_raw(csv_path)

    # Remove coluna de data leakage
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Codifica o alvo: Alive=0, Dead=1
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    logger.info(f"Classes: {list(le.classes_)} | Distribuição: {np.bincount(y)}")

    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras"
    )
    logger.info(
        f"Dead no treino: {y_train.sum()} ({y_train.mean():.1%}) | "
        f"Dead no teste: {y_test.sum()} ({y_test.mean():.1%})"
    )

    return X_train, X_test, y_train, y_test, le
