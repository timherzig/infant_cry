from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="/netscratch/herzig/shared_projects/infant_cry/config/trill1_noaug.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--gen_examples",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--train_loso",
        type=bool,
        default=False,
    )

    return parser.parse_args()
