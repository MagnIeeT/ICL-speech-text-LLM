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

    # Load environment variables early to check for custom paths
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

    # Add parent directory to sys.path to support sibling repositories like SALMONN
    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        logging.debug(f"Added parent directory {parent_dir} to sys.path")

    # Explicitly support SALMONN_SOURCE_PATH if provided
    salmonn_path = os.getenv("SALMONN_SOURCE_PATH")
    if salmonn_path and os.path.exists(salmonn_path):
        if salmonn_path not in sys.path:
            sys.path.insert(0, salmonn_path)
            logging.debug(f"Added SALMONN_SOURCE_PATH {salmonn_path} to sys.path")

def get_env_path(key, default=None):
    """Get a path from environment variable."""
    path = os.getenv(key, default)
    if path is None:
        logging.warning(f"Environment variable {key} is not set.")
    return path

setup_environment()
