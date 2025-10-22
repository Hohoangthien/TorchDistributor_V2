import yaml
import argparse
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    """Loads the YAML config file, resolving path placeholders."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Resolve path placeholders --- 
    # This allows defining a base_path and reusing it in other path definitions.
    
    # Resolve for data_source
    if 'data_source' in config and 'base_path' in config['data_source']:
        base_path = config['data_source']['base_path']
        for key, value in config['data_source'].items():
            if isinstance(value, str):
                config['data_source'][key] = value.format(base_path=base_path)

    # Resolve for artifact_storage (if it also uses placeholders in the future)
    if 'artifact_storage' in config and 'base_path' in config['artifact_storage']:
        base_path = config['artifact_storage']['base_path']
        for key, value in config['artifact_storage'].items():
            if isinstance(value, str):
                config['artifact_storage'][key] = value.format(base_path=base_path)

    return config

def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments to get the config file path."""
    parser = argparse.ArgumentParser(description="Run the Spark ML training pipeline based on a config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the override the output directory e.g., hdfs://master:9000/usr/ubuntu/my_results."
    )
    return parser.parse_args()