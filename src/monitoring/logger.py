"""
Módulo de logging estruturado com loguru.

Centraliza a configuração do logger para todo o projeto.
Uso:
    from src.monitoring.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Mensagem de log")
"""

import os
import sys
from pathlib import Path

from loguru import logger as _logger

# Diretório de logs na raiz do projeto
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# Remove o handler padrão do loguru (console sem formatação estruturada)
_logger.remove()

# ----- Configuração do handler de console -----
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> – "
        "<level>{message}</level>"
    ),
    colorize=True,
    enqueue=True,   # thread-safe
)

# ----- Configuração do handler de arquivo (rotação diária) -----
_LOG_FILE = os.getenv("LOG_FILE", "")
_log_path = Path(_LOG_FILE) if _LOG_FILE else _LOG_DIR / "app_{time:YYYY-MM-DD}.log"

_logger.add(
    str(_log_path),
    level=_LOG_LEVEL,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} – {message}"
    ),
    rotation="00:00",   # novo arquivo a cada meia-noite
    retention="7 days", # mantém 7 dias de histórico
    encoding="utf-8",
    enqueue=True,
)


def get_logger(name: str):
    """
    Retorna um logger com o contexto (nome do módulo) já vinculado.

    Parâmetros
    ----------
    name : str
        Normalmente ``__name__`` do módulo que chama a função.

    Retorna
    -------
    loguru.Logger
        Instância configurada do logger.
    """
    return _logger.bind(name=name)
