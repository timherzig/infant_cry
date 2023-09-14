import os
import wandb

from omegaconf import OmegaConf

from data.babycry import BabyCry
from utils.parser import parse_arguments
from utils.training import train_single


def main(args):
    # Load config & initialize wandb
    config = OmegaConf.load(args.config)
    os.makedirs("checkpoints", exist_ok=True)
    name = (
        config.model.name
        + f'_{len([x for x in os.listdir("checkpoints") if config.model.name in x]) + 1}'
    )
    wandb.init(
        project="trill_babycry",
        config=config.train,
        name=name,
    )
    save_dir = "checkpoints/" + name
    os.makedirs(save_dir)
    OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

    train_single(None, None, args, config, save_dir)

    print(f"Done! Model saved to {save_dir}")


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
