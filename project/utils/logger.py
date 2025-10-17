import logging
import os

def setup_logger(rank, log_file=None):
    """
    Sets up a logger for each worker, ensuring no duplicate handlers.
    Optionally logs to a file.
    """
    logger_name = f"Worker-{rank}"
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers if they already exist
    if logger.hasHandlers():
        # Clear existing handlers to ensure clean setup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f'[%(asctime)s][RANK {rank}][%(levelname)s] %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Add file handler if a path is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False
    
    return logger
