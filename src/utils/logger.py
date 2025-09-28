"""
Logger utility module for DeepDTA-Pro
"""

import logging
import sys
from pathlib import Path
from typing import Optional

class Logger:
    """Simple logger utility for DeepDTA-Pro examples."""
    
    def __init__(self, name: str = "DeepDTA-Pro", level_or_dir: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level_or_dir: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or log directory
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        
        # Handle case where second parameter might be a directory instead of level
        if level_or_dir and any(char in level_or_dir for char in ['/','\\','.']):
            # It's likely a directory path, use default level
            level = "INFO"
            if not log_file:
                log_file = str(Path(level_or_dir) / f"{name}.log")
        else:
            # It's a logging level
            level = level_or_dir
        
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)