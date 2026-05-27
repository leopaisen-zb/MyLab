# adapters/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseAdapter(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the adapter (load model, etc.)"""
        pass

    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """Execute the main forward pass"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input before processing"""
        pass

    def is_initialized(self) -> bool:
        return self._initialized

    def reload(self, config: Optional[Dict[str, Any]] = None):
        """Reload the adapter with new config"""
        if config:
            self.config.update(config)
        self._initialized = False
        return self.initialize()
