import json

def load_config(filepath):
    """
    Load configuration from a JSON file.

    Args:
        filepath (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{filepath}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{filepath}': {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the config: {e}")