"""
Centralized Activity Logger for Crypto Filter Shiny App
Tracks all operations, errors, and debug messages across modules.
"""
import pandas as pd
from datetime import datetime
from typing import Literal

class ActivityLogger:
    """Singleton logger to track all app activities"""
    _instance = None
    _logs = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def log(self, module: str, level: Literal["INFO", "WARNING", "ERROR", "DEBUG"], message: str):
        """Add a log entry"""
        entry = {
            'timestamp': datetime.now(),
            'module': module,
            'level': level,
            'message': message
        }
        self._logs.append(entry)
        # Also print to console
        print(f"[{level}] [{module}] {message}")
        
    def info(self, message: str, module: str = "System"):
        self.log(module, "INFO", message)

    def warning(self, message: str, module: str = "System"):
        self.log(module, "WARNING", message)

    def error(self, message: str, module: str = "System"):
        self.log(module, "ERROR", message)

    def debug(self, message: str, module: str = "System"):
        self.log(module, "DEBUG", message)
    
    def get_logs(self, limit: int = 100) -> pd.DataFrame:
        """Get recent logs as DataFrame"""
        if not self._logs:
            return pd.DataFrame(columns=['timestamp', 'module', 'level', 'message'])
        
        recent = self._logs[-limit:]
        return pd.DataFrame(recent)
    
    def clear(self):
        """Clear all logs"""
        self._logs.clear()

# Global logger instance
logger = ActivityLogger()
