"""
Testes para src/models/dataset.py

Cobrem o carregamento do CSV real, o pré-processador e o pipeline
de preparação de dados, garantindo rastreabilidade do Módulo 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.models.dataset import (
    CATEGORICAL_COLS,
    DEFAULT_CSV,
    NUMERIC_COLS,
    TARGET_COL,
    build_preprocessor,
    load_raw,
    prepare_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Carrega o CSV real uma vez por módulo de teste."""
    if not DEFAULT_CSV.exists():
        pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
    return load_raw()


@pytest.fixture(scope="module")
def prepared_data():
    """Chama prepare_data() uma vez por módulo de teste."""
    if not DEFAULT_CSV.exists():
        pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
    return prepare_data()


# ---------------------------------------------------------------------------
# Testes de load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_retorna_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_shape_minimo(self, raw_df):
        """Esperamos pelo menos 4000 linhas e as colunas conhecidas."""
        assert raw_df.shape[0] >= 4000
        assert raw_df.shape[1] >= 16

    def test_colunas_numericas_presentes(self, raw_df):
        for col in NUMERIC_COLS:
            assert col in raw_df.columns, f"Coluna numérica ausente: {col}"

    def test_colunas_categoricas_presentes(self, raw_df):
        for col in CATEGORICAL_COLS:
            assert col in raw_df.columns, f"Coluna categórica ausente: {col}"

    def test_coluna_target_presente(self, raw_df):
        assert TARGET_COL in raw_df.columns

    def test_target_valores_validos(self, raw_df):
        """Target deve conter apenas 'Alive' e 'Dead'."""
        valores = set(raw_df[TARGET_COL].unique())
        assert valores == {"Alive", "Dead"}

    def test_sem_nan_nas_colunas_principais(self, raw_df):
        cols_principais = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]
        for col in cols_principais:
            n_nulos = raw_df[col].isna().sum()
            assert n_nulos == 0, f"Coluna {col!r} tem {n_nulos} NaN(s)"

    def test_erro_csv_invalido(self, tmp_path):
        """Deve levantar ValueError se o CSV não tiver as colunas esperadas."""
        csv_ruim = tmp_path / "vazio.csv"
        csv_ruim.write_text("col1,col2\n1,2\n")
        with pytest.raises(ValueError, match="Colunas ausentes"):
            load_raw(csv_ruim)


# ---------------------------------------------------------------------------
# Testes de build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_retorna_column_transformer(self):
        pre = build_preprocessor()
        assert isinstance(pre, ColumnTransformer)

    def test_nomes_transformers(self):
        pre = build_preprocessor()
        nomes = [t[0] for t in pre.transformers]
        assert "num" in nomes
        assert "cat" in nomes

    def test_colunas_numericas_corretas(self):
        pre = build_preprocessor()
        _, _, cols = pre.transformers[0]  # transformer "num"
        assert list(cols) == NUMERIC_COLS

    def test_colunas_categoricas_corretas(self):
        pre = build_preprocessor()
        _, _, cols = pre.transformers[1]  # transformer "cat"
        assert list(cols) == CATEGORICAL_COLS

    def test_fit_transform_nao_produz_nan(self, raw_df):
        """Pré-processador ajustado no raw_df não deve produzir NaN."""
        pre = build_preprocessor()
        X = raw_df.drop(columns=[TARGET_COL, "Survival Months"], errors="ignore")
        resultado = pre.fit_transform(X)
        assert not np.isnan(resultado).any()

    def test_saida_shape(self, raw_df):
        """Shape de saída deve ter n_colunas = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)."""
        pre = build_preprocessor()
        X = raw_df.drop(columns=[TARGET_COL, "Survival Months"], errors="ignore")
        resultado = pre.fit_transform(X)
        assert resultado.shape[1] == len(NUMERIC_COLS) + len(CATEGORICAL_COLS)


# ---------------------------------------------------------------------------
# Testes de prepare_data
# ---------------------------------------------------------------------------

class TestPrepareData:
    def test_retorna_5_elementos(self, prepared_data):
        assert len(prepared_data) == 5

    def test_tipos_corretos(self, prepared_data):
        X_train, X_test, y_train, y_test, le = prepared_data
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_split_80_20(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        total = len(X_train) + len(X_test)
        ratio_teste = len(X_test) / total
        assert abs(ratio_teste - 0.2) < 0.01

    def test_y_binario(self, prepared_data):
        _, _, y_train, y_test, _ = prepared_data
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})

    def test_label_encoder_classes(self, prepared_data):
        _, _, _, _, le = prepared_data
        assert list(le.classes_) == ["Alive", "Dead"]

    def test_estratificacao(self, prepared_data):
        """Proporção de Dead no treino e teste deve ser similar (±5 pp)."""
        _, _, y_train, y_test, _ = prepared_data
        proporcao_treino = y_train.mean()
        proporcao_teste = y_test.mean()
        assert abs(proporcao_treino - proporcao_teste) < 0.05

    def test_sem_sobreposicao_indices(self, prepared_data):
        """Conjuntos de treino e teste não devem ter índices em comum."""
        X_train, X_test, _, _, _ = prepared_data
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_reprodutibilidade(self):
        """Duas chamadas com mesmo random_state devem produzir resultados idênticos."""
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        X_train_1, _, y_train_1, _, _ = prepare_data(random_state=42)
        X_train_2, _, y_train_2, _, _ = prepare_data(random_state=42)
        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        np.testing.assert_array_equal(y_train_1, y_train_2)
