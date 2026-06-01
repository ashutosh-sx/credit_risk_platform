import logging
import sys
import os

def setup_logger(name: str, log_level: str = None) -> logging.Logger:
    """
    Sets up a standardized logger with color formatting for terminal output.
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # ANSI Colors
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        green = "\x1b[32;20m"
        reset = "\x1b[0m"

        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

        class CustomFormatter(logging.Formatter):
            FORMATS = {
                logging.DEBUG: grey + format_str + reset,
                logging.INFO: green + format_str + reset,
                logging.WARNING: yellow + format_str + reset,
                logging.ERROR: red + format_str + reset,
                logging.CRITICAL: bold_red + format_str + reset
            }

            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
                return formatter.format(record)

        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

    return logger
