# tm_extractor

`tm_extractor` is a command-line tool for extracting tearing mode (TM) features from diagnostic signals. It supports configuration through JSON or YAML and can process input data to extract the presence, frequency, amplitude, and coupling characteristics of the tearing mode.

## Installation

Make sure all necessary dependencies are installed. If you are using tm_extractor as a Python package, follow these steps:

First, navigate to the tm_extractor directory, where the `setup.py` file is located, using the cd command in your terminal:

```bash
cd /path/to/tm_extractor
```

Then, run the following command to install the package:

```bash
pip install .
```
This will install tm_extractor and its dependencies in the current Python environment. After installation, you can use tm_extractor as a command-line tool.

## Usage

Run the following command in the terminal (replace the path with your actual data location):

```bash
    tm_extractor --from_json_or_yaml "from json" \
                 --input_file_path "example\raw_data" \
                 --output_file_path "example\tearing_mode_data" \
                 --processes 1
```

### Command-Line Arguments

| Argument                 | Type   | Required | Default Value            | Description                                      |
|--------------------------|--------|----------|--------------------------|--------------------------------------------------|
| `--from_json_or_yaml`    | `str`  | ✅ Yes   | `"from json"`            | Specify the configuration format: `"from json"` or `"from yaml"` |
| `--from_json_to_yaml`    | `bool` | ❌ No    | `True`                   | Whether to convert JSON configuration to YAML    |
| `--json_file_path`       | `str`  | ❌ No    | `"default_path"`         | Path to the JSON configuration file              |
| `--to_yaml_file_path`    | `str`  | ❌ No    | `"pipeline_config.yaml"` | Path to save the converted YAML configuration   |
| `--input_file_path`      | `str`  | ✅ Yes   | `"example_shotset"`      | Path to the input data                          |
| `--output_file_path`     | `str`  | ✅ Yes   | `"save_shotset"`         | Path to save the processed output data           |
| `--final_plt_path`       | `str`  | ❌ No    | `"mode_amp_plt"`         | Path to save processed plots                     |
| `--processes`            | `int`  | ❌ No    | `0`                      | Number of processes for parallel execution       |
| `--save_updated_only`    | `bool` | ❌ No    | `False`                  | Save only updated results                        |
| `--shot_filter_config`   | `str`  | ❌ No    | `"all_shot"`             | Configuration to filter shots                    |
| `--ext_name`             | `str`  | ❌ No    | `"tm_extractor"`         | Name of the extractor                            |

## Workflow

1. **Load Configuration**  
   - If `--from_json_or_yaml "from json"`, load configuration from `default_config_json.json` or the specified JSON file.
   - If `--from_json_or_yaml "from yaml"`, load configuration from `pipeline_config.yaml`.

2. **Initialize Extractor and Processing Pipeline**  
   - Create a `TMExtractor` instance from the configuration file.
   - Generate the processing pipeline.

3. **Process Shotset Data**  
   - Load the input data files.
   - Run the processing pipeline to extract tearing mode information and save the results.

4. **Save Processing Results**  
   - If `--from_json_to_yaml` is enabled, convert and save the JSON configuration as YAML.
   - The processed data is saved in the directory specified by `--output_file_path`.
   - The path for saving the tearing mode extraction plot, specified by `--final_plt_path`. If not specified, the plot will be saved in the current directory under the default name `mode_amp_plt`.
   - the exceptions are saved in the cuurent directory named `process_exceptions.log`.
