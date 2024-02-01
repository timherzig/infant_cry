import os
import shutil

from omegaconf import OmegaConf

from data.babycry import BabyCry
from utils.parser import parse_arguments
from utils.training import train_single
from utils.gen_ir_examples import gen_examples


def main(args):
    # Load config & initialize wandb
    config = OmegaConf.load(args.config)
    os.makedirs("./trained_models/", exist_ok=True)
    name = args.config.split("/")[-1].split(".")[0]

    save_dir = (
        f"./trained_models/{config.data.dir.split('/')[-1]}/{config.model.name}/{name}"
    )

    if name == "config":
        save_dir = f"./trained_models/debug"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving to {save_dir}")
    OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

    loss, f1, acc = train_single(None, None, args, config, save_dir)

    with open(os.path.join(save_dir, "results.txt"), "w") as f:
        f.write(f"Model: {name} test set results: \n")
        f.write(f"Loss: {loss}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"Acc: {acc}\n")
        f.close()

    print(f"Done! Model saved to {save_dir}")
    print(f"Loss: {loss}")
    print(f"F1: {f1}")
    print(f"Acc: {acc}")


if __name__ == "__main__":
    args = parse_arguments()

    if args.gen_examples:
        gen_examples()
    else:
        main(args)
