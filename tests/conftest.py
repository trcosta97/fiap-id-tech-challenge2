"""
Fixtures compartilhadas entre os testes.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def small_dataset():
    """
    Dataset sintético pequeno para testes rápidos.
    Não depende do CSV real – testes rodam sem os dados.
    """
    rng = np.random.default_rng(42)
    # 200 amostras, 14 features (4 numéricas + 10 categóricas, mimetizando o Breast Cancer)
    n = 200
    X_num = rng.standard_normal((n, 4))

    # Simula features categóricas como inteiros (já ordinalmente codificadas)
    X_cat = rng.integers(0, 4, size=(n, 10))

    import pandas as pd
    from src.models.dataset import CATEGORICAL_COLS, NUMERIC_COLS

    X = pd.DataFrame(
        np.hstack([X_num, X_cat.astype(float)]),
        columns=NUMERIC_COLS + CATEGORICAL_COLS,
    )
    y = rng.integers(0, 2, size=n)
    return X, y
