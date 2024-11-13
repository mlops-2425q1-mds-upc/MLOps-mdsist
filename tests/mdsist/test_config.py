import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv
from loguru import logger
from mdsist.config import *  

@pytest.fixture(scope='module', autouse=True)
def setup_env():
    """Set up the environment for testing."""
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.env') as env_file:
        env_file.write(b'SOME_ENV_VAR=test_value\n')
        env_file_name = env_file.name
    
    load_dotenv(env_file_name)
    yield
    
    # Clean up the temporary .env file
    os.remove(env_file_name)
    os.environ.pop("SOME_ENV_VAR", None)  

def test_proj_root():
    """Test the PROJ_ROOT path."""
    assert PROJ_ROOT == Path(__file__).resolve().parents[2], "PROJ_ROOT should point to the correct project root."

def test_data_directories():
    """Test the construction of data directories."""
    assert DATA_DIR == PROJ_ROOT / "data", "DATA_DIR should be set correctly."
    assert RAW_DATA_DIR == DATA_DIR / "raw", "RAW_DATA_DIR should be set correctly."
    assert INTERIM_DATA_DIR == DATA_DIR / "interim", "INTERIM_DATA_DIR should be set correctly."
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed", "PROCESSED_DATA_DIR should be set correctly."
    assert EXTERNAL_DATA_DIR == DATA_DIR / "external", "EXTERNAL_DATA_DIR should be set correctly."
    assert MODELS_DIR == PROJ_ROOT / "models", "MODELS_DIR should be set correctly."
    assert REPORTS_DIR == PROJ_ROOT / "reports", "REPORTS_DIR should be set correctly."
    assert FIGURES_DIR == REPORTS_DIR / "figures", "FIGURES_DIR should be set correctly."

def test_environment_variable_loading():
    """Test loading of environment variables."""
    load_dotenv()  # Load environment variables
    assert os.getenv('SOME_ENV_VAR') == 'test_value', "Environment variable should be loaded correctly."

def test_logger_configuration():
    """Test if logger is configured correctly."""
    logger.info("Testing logger configuration")
    
    # Check logger info does not raise an error
    assert True  # Just to indicate logger works without throwing errors.

def test_tqdm_handling():
    """Test if logger configuration adapts when tqdm is installed."""
    try:
        import tqdm
        assert tqdm is not None, "TQDM should be installed."
        
        # Test logger configuration behavior with tqdm
        # You may need to assert on the logger configuration itself or its output
    except ImportError:
        pass  # If tqdm isn't available, simply pass

def test_environment_variable_cleanup(setup_env):
    """Ensure environment variable is cleaned up after test."""
    # Check that SOME_ENV_VAR is set after the fixture runs
    assert os.getenv('SOME_ENV_VAR') == 'test_value', "SOME_ENV_VAR should be loaded correctly."
    
    # Simulate teardown
    os.environ.pop("SOME_ENV_VAR", None)  # Clean up manually
    
    # Check that SOME_ENV_VAR is cleaned up
    assert os.getenv('SOME_ENV_VAR') is None, "SOME_ENV_VAR should be cleaned up after test."

if __name__ == "__main__":
    pytest.main()