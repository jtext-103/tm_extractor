# -*- coding: utf-8 -*-
import argparse
import os
import pkg_resources
import warnings
import tm_extractor.utils as utils
from tm_extractor.tearing_mode_extractor.processor_registry import processor_registry
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet, Pipeline
from tm_extractor.tearing_mode_extractor.tmextractor import TMExtractor

def main_function():
    parser = argparse.ArgumentParser(description='tm_extractor_running')
    parser.add_argument('--from_json_or_yaml', type=str, default="from json",
                        help='whether to load config from json or yaml, options:[from json, from yaml]')
    parser.add_argument('--from_json_to_yaml', type=bool, required=False, default=True, help='whether to save config to yaml')
    parser.add_argument('--json_file_path', type=str, required=False, default='default_path', help='file path of json config')
    parser.add_argument('--to_yaml_file_path', type=str, required=False, default='pipeline_config.yaml',
                        help='file path of yaml config')
    parser.add_argument('--input_file_path', type=str,  default="example_shotset", help='process input file path')
    parser.add_argument('--output_file_path', type=str,  default="save_shotset", help='process output file path')
    parser.add_argument('--final_plt_path', type=str, required=False, default="mode_amp_plt", help='saving plot file path')
    parser.add_argument('--processes', type=int, required=False, default=0, help='processes number')
    parser.add_argument('--save_updated_only', type=bool, required=False, default=False, help='save updated only')
    parser.add_argument('--shot_filter_config', type=str, required=False, default="all_shot", help='shot filter config')
    # iTransformer
    parser.add_argument('--ext_name', type=str, required=False, default='tm_extractor',
                        help='extractor name, options:[extractor,...]')
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore", UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="jddb.processor.signal")
    args = parser.parse_args()
    print('Args in processing:')
    # set extractor
    # if args.ext_name == '...':
    #     Exp = TMExtractor
    # else: #
    #     Exp = ...
    input_file_path =os.path.join(f"{args.input_file_path}", "$shot_2$00")

    # Check if the configuration is not coming from YAML
    if args.from_json_or_yaml != "from yaml":
        # If a custom JSON file path is provided, use it; otherwise, use the default path
        if args.json_file_path != "default_path":
            json_path = args.json_file_path
        else:
            json_path = pkg_resources.resource_filename(__name__, os.path.join('tearing_mode_extractor', 'default_config_json.json'))

        # Initialize the TMExtractor with the provided or default JSON config file
        extractor = TMExtractor(config_file_path=json_path,plt_mode_amp_path=args.final_plt_path)

        # Create an extractor pipeline from the config
        extractor_pipeline = extractor.make_pipeline()

        # Convert the pipeline to a config
        extractor_pipeline.to_config()

        # If the user wants to save the pipeline configuration to YAML, do it
        if args.from_json_to_yaml:
            extractor_pipeline.to_yaml(args.to_yaml_file_path)

        # Initialize the input shotset from the file repository
        input_shotset = ShotSet(FileRepo(input_file_path))

        # If an output file path is provided, create the output repo; otherwise, use the input file path
        if args.output_file_path:
            output_file_path = os.path.join(f"{args.output_file_path}", "$shot_2$00")
            output_repo = FileRepo(output_file_path)
        else:
            output_repo = FileRepo(input_file_path)

        # Determine the shot filter based on the config
        if args.shot_filter_config == "all_shot":
            shot_filter = sorted(input_shotset.shot_list)
        else:
            shot_filter = utils.load_config('shot_filter.json')["shot_filter"]

        # Process the shots using the pipeline with the selected filter and save the results
        extractor_pipeline.process_by_shotset(
            shotset=input_shotset,
            processes=args.processes,
            save_repo=output_repo,
            shot_filter=shot_filter,
            save_updated_only=args.save_updated_only
        )
    else:
        # If the configuration is coming from YAML, initialize a pipeline from the YAML file
        pipeline = Pipeline([])

        # Load the pipeline configuration from the YAML file
        pipeline = pipeline.from_yaml('./pipeline_config.yaml', processor_registry=processor_registry)

        # Initialize the input shotset from the file repository
        input_shotset = ShotSet(FileRepo(os.path.join(f"{args.input_file_path}", "$shot_2$00")))

        # If an output file path is provided, create the output repo; otherwise, use the input file path
        if args.output_file_path:
            output_file_path = os.path.join(f"{args.output_file_path}", "$shot_2$00")
            output_repo = FileRepo(output_file_path)
        else:
            output_repo = FileRepo(input_file_path)

        # Determine the shot filter based on the config
        if args.shot_filter_config == "all_shot":
            shot_filter = sorted(input_shotset.shot_list)
        else:
            shot_filter = utils.load_config('shot_filter.json')["shot_filter"]

        # Process the shots using the pipeline with the selected filter and save the results
        pipeline.process_by_shotset(
            shotset=input_shotset,
            processes=args.processes,
            save_repo=output_repo,
            shot_filter=shot_filter,
            save_updated_only=args.save_updated_only
        )
if __name__ == '__main__':
    main_function()