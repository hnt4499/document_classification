import sys
import argparse

import yaml

from utils.trainer import Trainer


DESCRIPTION = """Train and evaluate a BERT for address extraction model."""


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    config["action"] = "training"
    config["resume_from"] = args.resume_from
    config["load_from"] = args.load_from
    config["config_path"] = args.config_path

    # Initializer trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='Path to full config.')
    parser.add_argument(
        '-r', '--resume-from', type=str, required=False, default=None,
        help='Directory to resume from.')
    parser.add_argument(
        '-l', '--load-from', type=str, required=False, default=None,
        help='Path to pretrained model to load from.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
