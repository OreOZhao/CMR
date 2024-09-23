import logging
from config import args


def _setup_logger(log_path):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding='UTF-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler, file_handler]
    return logger


logger = _setup_logger(args.log_path)
