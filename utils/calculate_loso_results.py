import pandas as pd


def calc_loso_results(csv_file):
    data = pd.read_csv(csv_file)
    data = data.dropna()

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
