import logging
import os
import sys

from logging.handlers import TimedRotatingFileHandler
from typing import Optional

def get_logger(name: Optional[str] = "faiss_app", log_to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        file_handler = TimedRotatingFileHandler(
            filename=f"{log_dir}/{name}",
            when="midnight",
            interval=1,
            backupCount=7,
            encoding='utf-8',
            utc=False
        )

        # Append date suffix to filename like faiss_app.2025-07-25.log
        file_handler.suffix = "%Y-%m-%d.log"

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
