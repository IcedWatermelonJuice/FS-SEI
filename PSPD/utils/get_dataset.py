import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader


def default_normalize_fn(x):
    return x


def sample_max_min(x):
    max_value = x.max(axis=2)
    min_value = x.min(axis=2)
    max_value = np.expand_dims(max_value, 2)
    min_value = np.expand_dims(min_value, 2)
    max_value = max_value.repeat(x.shape[2], axis=2)
    min_value = min_value.repeat(x.shape[2], axis=2)
    x = (x - min_value) / (max_value - min_value)
    return x


def entire_max_min(x):
    max_value = x.max()
    min_value = x.min()
    x = (x - min_value) / (max_value - min_value)
    return x


def power_normalize_fn(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i, 0, :], 2) + np.power(x[i, 1, :], 2)).max()
        x[i] = x[i] / np.power(max_power, 1 / 2)
    return x

def load_data(dataset_root, num_class, suffix, ch_type=None):
    ch_type = f"_{ch_type}" if ch_type else ""
    x = np.load(os.path.expanduser(os.path.join(dataset_root, f"X_{suffix}_{num_class}Class{ch_type}.npy")))
    y = np.load(os.path.expanduser(os.path.join(dataset_root, f"Y_{suffix}_{num_class}Class.npy")))

    return x, y


def pt_train_data(dataset_root, num_class, normalize_dataX=default_normalize_fn):
    x, y = load_data(dataset_root, num_class, "train")

    x = normalize_dataX(x)
    y = y.astype(np.uint8)
    return x, y


def ft_train_data(random_seed, dataset_root, num_class, k_shot, normalize_dataX=default_normalize_fn):
    x, y = load_data(dataset_root, num_class, "train")

    train_index_shot = []
    random.seed(random_seed)
    for i in range(num_class):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += random.sample(index_classi, k_shot)
    x = x[train_index_shot]
    y = y[train_index_shot]

    x = normalize_dataX(x)
    y = y.astype(np.uint8)
    return x, y


def ft_test_data(dataset_root, num_class, channel, normalize_dataX=default_normalize_fn):
    x, y = load_data(dataset_root, num_class, "test", channel)

    x = normalize_dataX(x)
    y = y.astype(np.uint8)

    return x, y


def get_pretrain_dataloader(opt):
    opt_dataset = opt["dataset"]

    if opt_dataset["normalize"] == "sample":
        normalize_fn = sample_max_min
    elif opt_dataset["normalize"] == "dataset":
        normalize_fn = entire_max_min
    elif opt_dataset["normalize"] == "power":
        normalize_fn = power_normalize_fn
    else:
        normalize_fn = default_normalize_fn

    X_train, Y_train = pt_train_data(opt_dataset["root"], opt_dataset["num_classes"], normalize_fn)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=opt_dataset["ratio"],
                                                      random_state=opt["random_seed"])

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    train_dataloader = DataLoader(train_dataset, batch_size=opt_dataset['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt_dataset['batch_size'], shuffle=True)
    return train_dataloader, val_dataloader


def get_finetune_dataloader(opt):
    opt_dataset = opt["dataset"]

    if opt_dataset["normalize"] == "sample":
        normalize_fn = sample_max_min
    elif opt_dataset["normalize"] == "dataset":
        normalize_fn = entire_max_min
    elif opt_dataset["normalize"] == "power":
        normalize_fn = power_normalize_fn
    else:
        normalize_fn = default_normalize_fn

    X_train, Y_train = ft_train_data(opt["random_seed"], opt_dataset["root"], opt_dataset["num_classes"],
                                     opt_dataset["shot"], normalize_fn)
    X_test, Y_test = ft_test_data(opt_dataset["root"], opt_dataset["num_classes"], opt_dataset["channel"], normalize_fn)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

    train_dataloader = DataLoader(train_dataset, batch_size=opt_dataset['train_batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt_dataset['test_batch_size'], shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == "__main__":

    from config import finetune_config
    ft_conf = finetune_config()
    ft_train_dataloader, ft_test_dataloader = get_finetune_dataloader(ft_conf)

    print("end")

