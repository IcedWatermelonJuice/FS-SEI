import argparse
import os
import sys
import torch

dataset_path_dict = {
    "wifi_ft62": {
        "linux": "~/Datasets/WiFi_ft62",
        "windows": "E:/Datasets/WiFi_ft62",
        "pt_class": 10,
        "ft_class": 6
    },
    "wifi_ft32": {
        "linux": "~/Datasets/WiFi_ft32",
        "windows": "E:/Datasets/WiFi_ft32",
        "pt_class": 10,
        "ft_class": 6
    },
    "wifi_ft20": {
        "linux": "~/Datasets/WiFi_ft20",
        "windows": "E:/Datasets/WiFi_ft20",
        "pt_class": 10,
        "ft_class": 6
    },
    "wifi_ft44": {
        "linux": "~/Datasets/WiFi_ft44",
        "windows": "E:/Datasets/WiFi_ft44",
        "pt_class": 10,
        "ft_class": 6
    },
    "wifi_ft50": {
        "linux": "~/Datasets/WiFi_ft50",
        "windows": "E:/Datasets/WiFi_ft50",
        "pt_class": 10,
        "ft_class": 6
    },
    "radar_00db": {
        "linux": "~/Datasets/Radar/npy/00dB",
        "pt_class": 13,
        "ft_class": 13
    },
    "radar_10db": {
        "linux": "~/Datasets/Radar/npy/10dB",
        "pt_class": 13,
        "ft_class": 13
    },
    "radar_20db": {
        "linux": "~/Datasets/Radar/npy/20dB",
        "pt_class": 13,
        "ft_class": 13
    },
    "bt_day1": {
        "linux": "~/Datasets/Day1BT",
        "windows": "E:/Datasets/Day1BT",
        "pt_class": 6,
        "ft_class": 4
    },
    "ads-b": {
        "linux": "~/Datasets/ADS-B",
        "windows": "E:/Datasets/ADS-B",
        "pt_class": 90,
        "ft_class": 30
    },
    "zigbee": {
        "linux": "~/Datasets/ZigBee_Indor_LOS",
        "windows": "E:/Datasets/ZigBee_Indor_LOS",
        "pt_class": 40,
        "ft_class": 20
    }
}
model_path_dict = {
    "CVCNN": "models/OnlyCVCNNFeature.py",
    "CNN": "models/OnlyCNNFeature.py",
    "ResNet18": "models/ResNet18Feature.py",
    "AlexNet": "models/AlexNetFeature.py",
    "EfficientNet": "models/EfficientNetFeature.py",
    "MobileNet": "models/MobileNetFeature.py",
    "ShuffleNet": "models/ShuffleNetFeature.py",
    "DenseNet": "models/DenseNetFeature.py",
    "VGG": "models/VGG11Feature.py",
    "Classifier": "models/RotationClassifier.py",
    "FS-Classifier": "models/DropoutClassifier.py",
    "NCE": "models/NCELoss.py",
    "MTL": "models/AutomaticWeightedLoss.py",
    "Rotator": "models/Rotator.py"
}


# The input param 'feature_dim' must be a valid integer and a positive even number.
def feature_dim_type(value):
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

    if int_value <= 0 or int_value % 2 != 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive even number")

    return int_value


def pretrain_config(encoder_name="CNN", dataset_name="wifi_ft44", input_type="stft", normalize_fn="power",
                    batch_size=32, max_epoch=250, lr=0.001, lr_step=50, lr_gamma=0.1, momentum=0.9, weight_decay=5e-4,
                    feature_dim=4096, nce_dim=128, nce_n=1281167, nce_t=0.07, nce_m=0.5, RANDOM_SEED=2024, ablate=""):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", type=str, default=encoder_name)
    parser.add_argument("--dataset", "-d", type=str, default=dataset_name)
    parser.add_argument("--input_type", "-t", type=str, default=input_type)
    parser.add_argument("--normalize_fn", "-n", type=str, default=normalize_fn)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epoch", type=int, default=max_epoch)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--lr_step", type=int, default=lr_step)
    parser.add_argument("--lr_gamma", type=float, default=lr_gamma)
    parser.add_argument("--momentum", type=float, default=momentum)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--feature_dim", type=feature_dim_type, default=feature_dim)
    parser.add_argument("--nce_dim", type=int, default=nce_dim)
    parser.add_argument("--nce_n", type=int, default=nce_n)
    parser.add_argument("--nce_t", type=float, default=nce_t)
    parser.add_argument("--nce_m", type=float, default=nce_m)
    parser.add_argument("--random_seed", "-r", type=int, default=RANDOM_SEED)
    parser.add_argument("--ablate", "-a", type=str, default=ablate)
    opt = parser.parse_args()

    loss_item = ["cls", "mse", "nce"]
    exp_suffix = f"_{opt.ablate}Ablate" if opt.ablate and opt.ablate in loss_item else ""
    exp = f"{opt.encoder}_{opt.dataset}_{opt.input_type}_{opt.normalize_fn}Norm{exp_suffix}"
    platform = "windows" if sys.platform.startswith("win") else "linux"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rot_angle = (0, 90, 180, 270)
    half_dim = opt.feature_dim // 2
    if opt.ablate and opt.ablate in loss_item:
        loss_item.remove(opt.ablate)
    loss_item = tuple(loss_item)
    dataset_root = dataset_path_dict[opt.dataset][platform]

    return {
        "exp_name": exp,
        "random_seed": opt.random_seed,
        "platform": platform,
        "device": device,
        "epoch": opt.epoch,
        "dataset": {
            "name": opt.dataset,
            "root": dataset_root,
            "type": opt.input_type,
            "normalize": opt.normalize_fn,
            "batch_size": opt.batch_size,
            "ratio": 0.2,
            "num_classes": dataset_path_dict[opt.dataset]["pt_class"]
        },
        "encoder": {
            "name": opt.encoder,
            "root": model_path_dict[opt.encoder],
            "feature_dim": opt.feature_dim
        },
        "classifier": {
            "root": model_path_dict["Classifier"],
            "in_dim": half_dim,
            "num_classes": len(rot_angle)
        },
        "rotator": {
            "root": model_path_dict["Rotator"],
            "rot_angle": rot_angle
        },
        "optimizer": {
            "lr": opt.lr,
            "momentum": opt.momentum,
            "weight_decay": opt.weight_decay,
            "step_size": opt.lr_step,
            "gamma": opt.lr_gamma
        },
        "nce": {
            "root": model_path_dict["NCE"],
            "low_dim": opt.nce_dim,
            "n": opt.nce_n,
            "k": opt.feature_dim,
            "t": opt.nce_t,
            "m": opt.nce_m
        },
        "mtl": {
            "root": model_path_dict["MTL"],
            "item": loss_item,
            "num": len(loss_item)
        }
    }


def finetune_config(encoder_name="CNN", dataset_name="wifi_ft62", ch_type="", input_type="stft", normalize_fn="power", train_batch_size=32,
                    test_batch_size=32, shot=5, max_epoch=50, max_iteration=100, lr=0.001, weight_decay=0, feature_dim=4096, RANDOM_SEED=2024,
                    pretrain_normalize_fn="same", pretrain_batch_size=32, pretrain_epoch=250, ablate="", snr=""):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", type=str, default=encoder_name)
    parser.add_argument("--dataset", "-d", type=str, default=dataset_name)
    parser.add_argument("--channel", "-c", type=str, default=ch_type)
    parser.add_argument("--input_type", "-t", type=str, default=input_type)
    parser.add_argument("--normalize_fn", "-n", type=str, default=normalize_fn)
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--test_batch_size", type=int, default=test_batch_size)
    parser.add_argument("--shot", "-s", type=int, nargs="+", default=shot)
    parser.add_argument("--epoch", type=int, default=max_epoch)
    parser.add_argument("--iteration", "-i", type=int, default=max_iteration)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--feature_dim", type=feature_dim_type, default=feature_dim)
    parser.add_argument("--random_seed", "-r", type=int, default=RANDOM_SEED)
    parser.add_argument("--pretrain_normalize_fn", type=str, default=normalize_fn if pretrain_normalize_fn == "same" else pretrain_normalize_fn)
    parser.add_argument("--pretrain_batch_size", type=int, default=pretrain_batch_size)
    parser.add_argument("--pretrain_epoch", type=int, default=pretrain_epoch)
    parser.add_argument("--ablate", "-a", type=str, default=ablate)
    parser.add_argument("--snr", type=str, default=snr)
    opt = parser.parse_args()

    loss_item = ["cls", "mse", "nce"]
    exp_suffix = f"_{opt.ablate}Ablate" if opt.ablate and opt.ablate in loss_item else ""
    dataset_fullname = opt.dataset + (f"_{opt.channel}_" if opt.channel else "_") + opt.input_type
    exp = f"{opt.encoder if opt.pretrain_epoch else 'FineZero'}_{dataset_fullname}_PT_{opt.pretrain_normalize_fn}Norm_FT_{opt.normalize_fn}Norm{exp_suffix}"
    pretrain_exp = f"{opt.encoder}_{opt.dataset}_{opt.input_type}_{opt.pretrain_normalize_fn}Norm{exp_suffix}"
    platform = "windows" if sys.platform.startswith("win") else "linux"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = dataset_path_dict[opt.dataset]["ft_class"]
    dataset_root = dataset_path_dict[opt.dataset][platform]
    return {
        "exp_name": exp,
        "random_seed": opt.random_seed,
        "platform": platform,
        "device": device,
        "iteration": opt.iteration,
        "epoch": opt.epoch,
        "dataset": {
            "name": opt.dataset,
            "channel": opt.channel,
            "root": dataset_root,
            "type": opt.input_type,
            "normalize": opt.normalize_fn,
            "train_batch_size": opt.train_batch_size,
            "test_batch_size": opt.test_batch_size,
            "ratio": 0.2,
            "num_classes": num_classes,
            "shot": opt.shot,
            "snr": opt.snr.split(",") if opt.snr else [None]
        },
        "encoder": {
            "name": opt.encoder,
            "root": model_path_dict[opt.encoder],
            "feature_dim": opt.feature_dim,
            "pretrain_path": os.path.join("./runs/pretext", pretrain_exp,
                                          "best_encoder.pth") if opt.pretrain_epoch else ""
        },
        "classifier": {
            "root": model_path_dict["Classifier"],
            "in_dim": opt.feature_dim,
            "ratio": 0.2,
            "num_classes": num_classes
        },
        "optimizer": {
            "lr": opt.lr,
            "weight_decay": opt.weight_decay
        }
    }


if __name__ == "__main__":
    import json

    pretrain_conf = pretrain_config()
    finetune_conf = finetune_config()
    print("==> Pretrain Config:", json.dumps(pretrain_conf))
    print("==> Finetune Config:", json.dumps(finetune_conf))
