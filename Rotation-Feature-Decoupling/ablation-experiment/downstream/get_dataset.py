import numpy as np
from sklearn.model_selection import train_test_split
import random

def normalize_dataX(x):
    max_value = x.max(axis=2)
    min_value = x.min(axis=2)
    max_value = np.expand_dims(max_value, 2)
    min_value = np.expand_dims(min_value, 2)
    max_value = max_value.repeat(x.shape[2], axis=2)
    min_value = min_value.repeat(x.shape[2], axis=2)
    x = (x - min_value) / (max_value - min_value)
    return x

def TrainDataset(random_seed, dataset_root, num_class, k_shot):
    x = np.load(f'{dataset_root}X_train_{num_class}Class.npy')
    y = np.load(f'{dataset_root}Y_train_{num_class}Class.npy')
    y = y.astype(np.uint8)
    train_index_shot = []
    random.seed(random_seed)
    for i in range(num_class):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += random.sample(index_classi, k_shot)

    x = x[train_index_shot]
    y = y[train_index_shot]

    x=normalize_dataX(x)
    return x, y


def TestDataset(dataset_root, num_class):
    x = np.load(f'{dataset_root}X_test_{num_class}Class.npy')
    y = np.load(f'{dataset_root}Y_test_{num_class}Class.npy')
    y = y.astype(np.uint8)

    x=normalize_dataX(x)
    return x, y


def TrainDataset_prepared(random_seed, opt):
    X_train, Y_train = TrainDataset(random_seed, opt["dataset_root"], opt["num_class"], opt["trainer"]["k_shot"])
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=30)

    return X_train, X_val, Y_train, Y_val


def TestDataset_prepared(random_seed, opt):
    X_test, Y_test = TestDataset(opt["dataset_root"], opt["num_class"])
    return X_test, Y_test
