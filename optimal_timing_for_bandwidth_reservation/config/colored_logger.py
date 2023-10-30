import os
import logging
import coloredlogs
import time


def setup_logger(log_directory="./logs"):
    os.makedirs(log_directory, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(log_directory, f"log_{time.strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"
    log_formatter = logging.Formatter(log_format, datefmt=log_date_format)

    file_handler.setFormatter(log_formatter)
    console_handler.setFormatter(log_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    coloredlogs.install(
        level="INFO", logger=logger, fmt=log_format, datefmt=log_date_format
    )

    return logger
