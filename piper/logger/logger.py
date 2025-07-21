import logging
import logging.config
import os
import sys
from datetime import datetime
from logging import Logger
from typing import Optional
import tomllib


_logger_configured = False


def _get_logging_config() -> dict:
    """Get logging configuration from pyproject.toml"""
    try:
        pyproject_path = os.path.join(os.path.dirname(__file__), "../../pyproject.toml")
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("logging", {}).get("output", {})
    except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
        sys.stderr.write(f"Warning: Could not read pyproject.toml: {e}. Using defaults.\n")
        return {"log_to_console": True, "log_to_file": False}


def configure_logger() -> None:
    """Configure the logger based on pyproject.toml settings"""
    global _logger_configured
    if _logger_configured:
        return

    config = _get_logging_config()
    log_to_console = config.get("log_to_console", True)
    log_to_file = config.get("log_to_file", False)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_defaults = {}

    if log_to_file:
        log_dir = os.path.join(base_dir, "../../log")
        
        if "pytest" in sys.modules:
            log_file_name = "pytest_log.log"
        else:
            log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_piper_log.log"
        
        log_file_path = os.path.join(log_dir, log_file_name)
        
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "a", encoding="utf-8"):
                pass
            config_defaults["logfilename"] = log_file_path
        except OSError as e:
            sys.stderr.write(f"Warning: Could not create/access log file {log_file_path}: {e}\n")
            log_to_file = False

    # Use existing logging.conf if file logging is enabled, otherwise configure programmatically
    if log_to_file:
        log_conf_file_path = os.path.join(base_dir, "logging.conf")
        
        if os.path.exists(log_conf_file_path):
            try:
                logging.config.fileConfig(
                    log_conf_file_path, 
                    defaults=config_defaults, 
                    disable_existing_loggers=False
                )
            except Exception as e:
                sys.stderr.write(f"Error configuring logging from {log_conf_file_path}: {e}. Using basic config.\n")
                _setup_basic_logging(log_to_console)
        else:
            _setup_basic_logging(log_to_console)
    else:
        _setup_basic_logging(log_to_console)

    _logger_configured = True


def _setup_basic_logging(log_to_console: bool = True) -> None:
    """Setup basic logging configuration"""
    if log_to_console:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(module)-30s %(lineno)-4d - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout
        )
    else:
        # Minimal logging if console is disabled and no file logging
        logging.basicConfig(level=logging.WARNING)


def get_logger(name: Optional[str] = None) -> Logger:
    """
    Get a logger instance.

    Parameters
    ----------
    name : str, optional
        The name of the logger. If None, uses __name__ of the calling module.
        For most cases, pass __name__ as the name.

    Returns
    -------
    Logger
        The configured logger object.
    """
    if not _logger_configured:
        configure_logger()
    
    config = _get_logging_config()
    log_to_console = config.get("log_to_console", True)
    log_to_file = config.get("log_to_file", False)
    
    if log_to_file and log_to_console:
        # Use aux.console logger from logging.conf for both console and file output
        return logging.getLogger("aux.console")
    elif log_to_file:
        # Use aux logger from logging.conf for file-only output
        return logging.getLogger("aux")
    else:
        name = "aux.console" if name else "root"
        return logging.getLogger(name)