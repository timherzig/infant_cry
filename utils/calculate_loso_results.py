import pandas as pd

from argparse import ArgumentParser


def calc_loso_results(csv_file):
    data = pd.read_csv(csv_file)
    data = data.dropna()
    print(f"Length of data before removing duplicates: {len(data)}")
    data = data.groupby("val_speakers", group_keys=False).apply(
        lambda x: x.loc[x.f1.idxmax()]
    )
    print(f"Length of data after removing duplicates: {len(data)}")

    max_f1 = data["f1"].max()
    min_f1 = data["f1"].min()

    max_val_f1 = data["dev_f1"].max()
    min_val_f1 = data["dev_f1"].min()

    avg_f1 = data["f1"].mean()
    avg_acc = data["acc"].mean()
    avg_dev_f1 = data["dev_f1"].mean()
    avg_dev_acc = data["dev_acc"].mean()

    return (
        avg_f1,
        avg_acc,
        avg_dev_f1,
        avg_dev_acc,
        max_f1,
        max_val_f1,
        min_f1,
        min_val_f1,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str)

    args = parser.parse_args()

    avg_f1, avg_acc, avg_dev_f1, avg_dev_acc, max_f1, max_val_f1, min_f1, min_val_f1 = (
        calc_loso_results(args.csv)
    )

    model_name = args.csv.split("/")[-2]

    with open((f'{"/".join(args.csv.split("/")[:-1])}/results.txt'), "w") as f:
        f.write(f"Model: {model_name} test set results: \n")
        f.write(f"Loss: N/A\n")
        f.write(f"F1: {avg_f1}\n")
        f.write(f"Acc: {avg_acc}\n")
        f.write(f"Dev Loss: N/A\n")
        f.write(f"Dev F1: {avg_dev_f1}\n")
        f.write(f"Dev Acc: {avg_dev_acc}\n")
        f.write(f"Max F1: {max_f1}\n")
        f.write(f"Max Val F1: {max_val_f1}\n")
        f.write(f"Min F1: {min_f1}\n")
        f.write(f"Min Val F1: {min_val_f1}\n")
        f.close()
