import os
import sys
import argparse

import yaml

from utils.trainer import Trainer


DESCRIPTION = """Evaluate a BERT for address extraction model."""

def check_args(args):
    post_processing = args.post_processing
    if not post_processing:
        string = ("`post_processing` is set to False. This is NOT recommended. Do you want to continue (y: yes, continue, "
                  "n: no, change `post_processing` to `True`): ")
        choice = input(string)
        while choice not in ["y", "n"]:
            choice = input(string)
        if choice == "n":
            args.post_processing = True


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    config["action"] = "evaluation"
    config["config_path"] = args.config_path
    config["load_from"] = args.load_from
    config["data_path"] = args.data_path
    config["resume_from"] = None
    # Overwrite values
    check_args(args)
    config["evaluation"]["post_processing"] = args.post_processing
    if args.confidence_threshold is not None:
        config["evaluation"]["confidence_threshold"] = args.confidence_threshold

    if args.save_csv_path is None:
        model_dir, _ = os.path.split(args.load_from)
        args.save_csv_path = os.path.join(model_dir, "prediction_results.csv")
    config["save_csv_path"] = args.save_csv_path

    # Initializer trainer
    trainer = Trainer(config)

    # Start training
    trainer.eval()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='Path to full config that the model is trained with.')
    parser.add_argument(
        '-m', '--load-from', type=str, required=True, help='Path to trained model.')
    parser.add_argument(
        '-d', '--data-path', type=str, required=False, default=None,
        help='Path to load the data from. If not specified, load the `val` dataset specified in the config file.')
    parser.add_argument(
        '-s', '--save-csv-path', type=str, required=False, default=None,
        help='Path to save the result csv file. If not specified, results will '
             'be saved to directory where the model was saved.')
    parser.add_argument(
        '-p', '--post-processing', action="store_true", default=False,
        help='Whether to post process predictions. Overwrite `post_processing` value in the config file.')
    parser.add_argument(
        '-t', '--confidence-threshold', type=float, default=None,
        help='Confidence threshold to make predictions. Predictions with score lower than this will be discarded. '
             'Overwrite `confidence_threshold` value in the config file, if specified.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
