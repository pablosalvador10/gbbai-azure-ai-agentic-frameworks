import functools
import logging
import time
from typing import Callable, Optional

# Define a new logging level named "KEYINFO" with a level of 25
KEYINFO_LEVEL_NUM = 25
logging.addLevelName(KEYINFO_LEVEL_NUM, "KEYINFO")

def keyinfo(self: logging.Logger, message, *args, **kws):
    """
    Log 'msg % args' with severity 'KEYINFO'.
    """
    if self.isEnabledFor(KEYINFO_LEVEL_NUM):
        self._log(KEYINFO_LEVEL_NUM, message, args, **kws)

logging.Logger.keyinfo = keyinfo

class CustomFormatter(logging.Formatter):
    """
    CustomFormatter overrides 'funcName' and 'filename' attributes in the log record.

    When a decorator is used to log function calls in a different file, this formatter helps
    preserve the correct file and function name in the log records.

    - 'funcName' is overridden with 'func_name_override', if present in the record.
    - 'filename' is overridden with 'file_name_override', if present in the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.funcName = getattr(record, "func_name_override", record.funcName)
        record.filename = getattr(record, "file_name_override", record.filename)
        return super().format(record)

def get_logger(
    name: str = "micro",
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    include_stream_handler: bool = True,
) -> logging.Logger:
    """
    Returns a configured logger with a custom name, level, and formatter.

    Parameters:
    name (str): Name of the logger.
    level (int, optional): Initial logging level. Defaults to INFO if not provided.
    log_file (str, optional): File path for logging. If provided, logs will be written to this file.
    include_stream_handler (bool): Whether to include a stream handler. Defaults to True.

    Returns:
    logging.Logger: Configured logger instance.
    """
    formatter = CustomFormatter(
        "%(asctime)s - %(name)s - %(processName)-10s - "
        "%(levelname)-8s %(message)s (%(filename)s:%(funcName)s:%(lineno)d)"
    )
    logger = logging.getLogger(name)

    if level is not None or logger.level == 0:
        logger.setLevel(level or logging.INFO)

    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if include_stream_handler and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger

def log_function_call(
    logger_name: str, log_inputs: bool = True, log_output: bool = True
) -> Callable:
    """
    Decorator to log function calls, input arguments, output, execution duration, and completion message.

    Parameters:
    logger_name (str): The name for the logger.
    log_inputs (bool): Whether to log input arguments. Defaults to True.
    log_output (bool): Whether to log the function's output. Defaults to True.

    Returns:
    Callable: The decorated function.
    """

    def decorator_log_function_call(func):
        @functools.wraps(func)
        def wrapper_log_function_call(*args, **kwargs):
            logger = get_logger(logger_name)
            func_name = func.__name__

            if log_inputs:
                args_str = ", ".join(map(str, args))
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                logger.info(
                    f"Function {func_name} called with arguments: {args_str} and keyword arguments: {kwargs_str}"
                )
            else:
                logger.info(f"Function {func_name} called")

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            if log_output:
                logger.info(f"Function {func_name} output: {result}")

            logger.info(f"Function {func_name} executed in {duration:.2f} seconds")
            logger.info(f"Function {func_name} completed")

            return result

        return wrapper_log_function_call

    return decorator_log_function_call
