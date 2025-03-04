# tearing_mode_extractor
    The `tearing_mode_extractor` package is designed for extracting information about tearing modes from diagnostic signals. It includes tools for determining the presence of tearing modes, identifying their frequencies, amplitudes, and coupling characteristics.

# File Structure
    ├── tearing_mode_extractor/
    │   ├── __init__.py
    │   ├── processor_registry.py
    │   ├── tmextractor.py
    │   └── default_config_json.json

# File Descriptions
## `default_config_json.json`

    This JSON file is an essential input to the TMExtractor class (found in tmextractor.py). It contains:

    Tag names of input and output signals.
    Various processing parameters required for extracting tearing mode information.
## `processor_registry.py`

    This file maintains a dictionary that maps processor class names to their corresponding implementations. It serves as a registry for managing different processing components within the extraction pipeline.

## `tmextractor.py`

   This file defines the TMExtractor class, which inherits from jddb.extractor. 
   TMExtractor is responsible for extracting tearing mode characteristics, including:
   Whether a tearing mode is present.
   The frequency and amplitude of different tearing mode types.
   Coupling behavior of tearing modes.
   This package provides a structured way to analyze tearing modes using predefined configurations and modular processing components.

## `tmextarctor.py`

`TMExtractor` is the smallest unit of data processing, corresponding to a single signal in the data

    extarctor.TMExtractor(
        config_file_path: str
    )

Arguments:

* `config_file_path`：Path to the configuration file, which contains parameters required for feature extraction.

Properties:

* `config_file_path`：Stores the path of the configuration file,`str`.
* `config`：A dictionary containing the extracted configuration settings, `dict`.

Methods:

* `load_config`
    
      load_config(self) -> dict

`load_config` Loads the configuration file from the specified path.

  Returns:
  * `dict`：The loaded configuration as a dictionary.

* `extract_steps`

      extract_steps(self) -> List[Step]

    Returns:
    * `List[Step]`：A list of Step objects representing the extraction steps.

* `make_pipeline`

      make_pipeline(self) -> Pipeline

     Returns:
    * `jddb.processor.Pipeline`：a specific processing pipeline.

Example:
    
    >>> from jddb.processor import extractor
    >>>
    >>> # Initialize an extractor with a config file
    >>> tm_extractor = TMExtractor("default_config_json.json")
    >>> # Loaded configuration
    >>> tm_extractor_config = tm_extractor.config  
    >>> tm_extractor_steps = tm_extractor.extract_steps() 
    >>> # Created pipeline
    >>> tm_extractor_pipeline = tm_extractor.make_pipeline() 