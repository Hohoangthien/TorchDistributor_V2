import logging

def setup_logger(rank):
    """Sets up a logger for each worker, ensuring no duplicate handlers."""
    logger_name = f"Worker-{rank}"
    logger = logging.getLogger(logger_name)
    
    # Avoid adding handlers if they already exist (e.g., in interactive sessions)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'[%(asctime)s][RANK {rank}][%(levelname)s] %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent log messages from propagating to the root logger
    logger.propagate = False
    
    return logger
