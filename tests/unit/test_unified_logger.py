"""Unit tests for unified logger (utils.logger)

Tests the Logger class with dependency injection and source tracking.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from utils.logger import Logger, set_default_logger, get_default_logger, log, warn, error


class TestLogger:
    """Test Logger class functionality."""
    
    def test_logger_initialization(self):
        """Test logger can be initialized with different sources."""
        logger = Logger(name="test", source="swing")
        assert logger.name == "test"
        assert logger.source == "swing"
        assert logger.dt_brain is None
    
    def test_logger_with_dt_brain(self):
        """Test logger can be initialized with DT brain."""
        mock_brain = Mock()
        logger = Logger(name="test", source="dt", dt_brain=mock_brain)
        assert logger.dt_brain is mock_brain
    
    def test_logger_custom_log_dir(self):
        """Test logger can use custom log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(name="test", source="backend", log_dir=Path(tmpdir))
            assert logger.log_dir == Path(tmpdir)
    
    def test_info_logging(self, tmp_path):
        """Test info level logging."""
        logger = Logger(name="test_component", source="swing", log_dir=tmp_path)
        logger.info("Test message")
        
        # Check log file was created
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        
        # Check log content
        content = log_files[0].read_text()
        assert "test_component" in content
        assert "swing" in content
        assert "INFO" in content
        assert "Test message" in content
    
    def test_warn_logging(self, tmp_path):
        """Test warning level logging."""
        logger = Logger(name="test_component", source="dt", log_dir=tmp_path)
        logger.warn("Warning message")
        
        # Check log file was created in dt_backend subdirectory
        log_files = list((tmp_path / "dt_backend").glob("*.log"))
        assert len(log_files) == 1
        
        # Check log content
        content = log_files[0].read_text()
        assert "test_component" in content
        assert "dt" in content
        assert "WARN" in content
        assert "Warning message" in content
    
    def test_error_logging(self, tmp_path):
        """Test error level logging."""
        logger = Logger(name="test_component", source="backend", log_dir=tmp_path)
        logger.error("Error message")
        
        # Check log file was created
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        
        # Check log content
        content = log_files[0].read_text()
        assert "ERROR" in content
        assert "Error message" in content
    
    def test_error_with_exception(self, tmp_path):
        """Test error logging with exception."""
        logger = Logger(name="test_component", source="backend", log_dir=tmp_path)
        
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("Error occurred", exc=e)
        
        # Check log file contains traceback
        log_files = list(tmp_path.glob("*.log"))
        content = log_files[0].read_text()
        assert "ERROR" in content
        assert "Error occurred" in content
        assert "ValueError" in content
        assert "Test exception" in content
    
    def test_log_with_context(self, tmp_path):
        """Test logging with context parameters."""
        logger = Logger(name="test_component", source="swing", log_dir=tmp_path)
        logger.info("Trade executed", symbol="AAPL", qty=100, price=150.50)
        
        # Check log contains context
        log_files = list(tmp_path.glob("*.log"))
        content = log_files[0].read_text()
        assert "symbol=AAPL" in content
        assert "qty=100" in content
        assert "price=150.5" in content
    
    def test_dt_brain_update_without_brain(self, tmp_path):
        """Test dt_brain_update without dt_brain configured."""
        logger = Logger(name="test_component", source="dt", log_dir=tmp_path)
        logger.dt_brain_update("test_knob", 0.5, 0.7, "test reason")
        
        # Should log warning
        log_files = list((tmp_path / "dt_backend").glob("*.log"))
        content = log_files[0].read_text()
        assert "WARN" in content
        assert "no dt_brain configured" in content
    
    def test_dt_brain_update_with_brain(self, tmp_path):
        """Test dt_brain_update with dt_brain configured."""
        mock_brain = Mock()
        logger = Logger(name="test_component", source="dt", dt_brain=mock_brain, log_dir=tmp_path)
        logger.dt_brain_update("test_knob", 0.5, 0.7, "test reason")
        
        # Should log info with brain emoji
        log_files = list((tmp_path / "dt_backend").glob("*.log"))
        content = log_files[0].read_text()
        assert "INFO" in content
        assert "test_knob" in content
        assert "0.5" in content
        assert "0.7" in content
    
    def test_log_alias(self, tmp_path):
        """Test log() is an alias for info()."""
        logger = Logger(name="test_component", source="backend", log_dir=tmp_path)
        logger.log("Test log message")
        
        log_files = list(tmp_path.glob("*.log"))
        content = log_files[0].read_text()
        assert "INFO" in content
        assert "Test log message" in content
    
    def test_warning_alias(self, tmp_path):
        """Test warning() is an alias for warn()."""
        logger = Logger(name="test_component", source="backend", log_dir=tmp_path)
        logger.warning("Test warning message")
        
        log_files = list(tmp_path.glob("*.log"))
        content = log_files[0].read_text()
        assert "WARN" in content
        assert "Test warning message" in content


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""
    
    def test_default_logger(self):
        """Test default logger can be retrieved."""
        logger = get_default_logger()
        assert logger is not None
        assert isinstance(logger, Logger)
    
    def test_set_default_logger(self):
        """Test default logger can be set."""
        custom_logger = Logger(name="custom", source="swing")
        set_default_logger(custom_logger)
        
        retrieved = get_default_logger()
        assert retrieved is custom_logger
        assert retrieved.name == "custom"
        assert retrieved.source == "swing"
    
    def test_module_level_log(self):
        """Test module-level log function works."""
        # Should not raise
        log("Test message")
    
    def test_module_level_warn(self):
        """Test module-level warn function works."""
        # Should not raise
        warn("Test warning")
    
    def test_module_level_error(self):
        """Test module-level error function works."""
        # Should not raise
        error("Test error")


class TestDTLoggerCompatibility:
    """Test dt_backend.core.logger_dt compatibility wrapper."""
    
    def test_dt_logger_imports(self):
        """Test dt_backend logger functions can be imported."""
        from dt_backend.core.logger_dt import log, info, warn, error
        
        # Should not raise
        info("Test info")
        warn("Test warn")
        error("Test error")
        log("Test log")
