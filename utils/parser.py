from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )

    return parser.parse_args()
