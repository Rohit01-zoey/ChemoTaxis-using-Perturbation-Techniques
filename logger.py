import logging

def setup_logger(log_file):
    """
    Set up a logger with a file handler.

    Parameters:
    -----------
    log_file : str
        Path to the log file.

    Returns:
    --------
    logger : logging.Logger
        Logger object.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create file handler and set its level to DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
