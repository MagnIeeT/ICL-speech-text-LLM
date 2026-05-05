import os
import logging

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

def setup_environment():
    """Load environment variables from .env file if available."""
    if HAS_DOTENV:
        # Look for .env in the current directory or parent directory
        dotenv_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            logging.debug(f"Loaded environment variables from {dotenv_path}")
        else:
            # Try project root if we are in a subfolder
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dotenv_path = os.path.join(project_root, '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                logging.debug(f"Loaded environment variables from {dotenv_path}")
    else:
        logging.debug("python-dotenv not installed, skipping .env loading")

def get_env_path(key, default=None):
    """Get a path from environment variable."""
    path = os.getenv(key, default)
    if path is None:
        logging.warning(f"Environment variable {key} is not set.")
    return path
