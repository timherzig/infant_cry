from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )

    return parser.parse_args()
