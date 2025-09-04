from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from pipeline.app_state import AppState


class BaseLayer(ABC):
    """
    Abstract base class for all pipeline processing layers.
    Provides common interface and utilities for dependency injection and execution.
    """
    
    def __init__(self, app_state: AppState, layer_name: Optional[str] = None):
        self.app_state = app_state
        self.layer_name = layer_name or self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.layer_name}")
        
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """
        Execute the layer's processing logic.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        pass
    
    def _log_execution_start(self, input_summary: str = ""):
        """Log the start of layer execution."""
        self.logger.info(f"Starting {self.layer_name} execution. {input_summary}")
        
    def _log_execution_end(self, output_summary: str = ""):
        """Log the end of layer execution."""
        self.logger.info(f"Completed {self.layer_name} execution. {output_summary}")
        
    def _handle_error(self, error: Exception, context: str = ""):
        """Handle and log errors in a consistent manner."""
        error_msg = f"Error in {self.layer_name}: {str(error)}"
        if context:
            error_msg += f" Context: {context}"
        self.logger.error(error_msg, exc_info=True)
        raise error