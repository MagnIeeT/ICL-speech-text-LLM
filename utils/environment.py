import os
import sys
import logging

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

def setup_environment():
    """Load environment variables and setup Python path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add project root to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logging.debug(f"Added {project_root} to sys.path")

    if HAS_DOTENV:
        # Look for .env in the current directory or parent directory
        dotenv_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            logging.debug(f"Loaded environment variables from {dotenv_path}")
        else:
            # Try project root
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

setup_environment()
