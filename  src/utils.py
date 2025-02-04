import json

def load_config(config_path):
    """
    Load configuration settings from a JSON file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration settings
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def save_config(config, config_path):
    """
    Save configuration settings to a JSON file.

    Args:
        config (dict): Configuration settings
        config_path (str): Path to save the configuration file
    """
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    print(f"Configuration saved to {config_path}")
