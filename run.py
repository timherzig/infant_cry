import os
import shutil

from omegaconf import OmegaConf

from data.babycry import BabyCry
from utils.parser import parse_arguments
from utils.training import train_single, train_loso
from utils.gen_ir_examples import gen_examples


def main(args):
    # Load config & initialize wandb
    config = OmegaConf.load(args.config)
    os.makedirs("./trained_models/", exist_ok=True)
    name = args.config.split("/")[-1].split(".")[0]

    save_dir = (
        f"./trained_models/{config.data.dir.split('/')[-1]}/{config.model.name}/{name}"
        + ("/loso" if args.train_loso else "")
    )

    if name == "config":
        save_dir = f"./trained_models/debug" + ("/loso" if args.train_loso else "")

    if args.train_loso:
        config.train.epochs = 10

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving to {save_dir}")
    OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

    if args.train_loso is True:
        (
            loss,
            f1,
            acc,
            dev_loss,
            dev_f1,
            dev_acc,
            max_f1,
            max_val_f1,
            min_f1,
            min_val_f1,
        ) = train_loso(args, config, save_dir)
    else:
        loss, f1, acc, dev_loss, dev_f1, dev_acc = train_single(
            None, None, args, config, save_dir
        )

    with open(os.path.join(save_dir, "results.txt"), "w") as f:
        f.write(f"Model: {name} test set results: \n")
        f.write(f"Loss: {loss}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"Acc: {acc}\n")
        f.write(f"Dev Loss: {dev_loss}\n")
        f.write(f"Dev F1: {dev_f1}\n")
        f.write(f"Dev Acc: {dev_acc}\n")
        if args.train_loso:
            f.write(f"Max F1: {max_f1}\n")
            f.write(f"Max Val F1: {max_val_f1}\n")
            f.write(f"Min F1: {min_f1}\n")
            f.write(f"Min Val F1: {min_val_f1}\n")
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
