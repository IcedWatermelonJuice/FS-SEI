import os
import shutil

import pandas as pd
import torch
from torch.optim import Adam

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils.config import finetune_config
from utils.utils import set_seed, get_logger_and_writer, create_model, ListApply, RecordTime, accuracy
from utils.get_dataset import get_finetune_dataloader
from tqdm import tqdm
from models.Rotator import Rotator


def run_step(inputs, labels, device, encoder, classifier, lossfn, dtype):
    if len(inputs.shape) != 4:
        inputs = Rotator.to_data_type(inputs, dtype)
    inputs = inputs.to(device)
    labels = labels.to(device)

    features = encoder(inputs)
    preds = classifier(features)

    loss_item = lossfn(preds, labels.long())

    return loss_item, accuracy(preds, labels)


def train(writer, config, iteration, epoch, train_dataloader, device, encoder, classifier, lossfn, optimizer):
    # encoder.train()
    encoder.eval()
    classifier.train()
    acc_sum = 0
    loss_sum = 0
    for inputs, labels in train_dataloader:
        loss_item, acc_item = run_step(inputs, labels, device, encoder, classifier, lossfn, config["dataset"]["type"])
        optimizer.zero_grad()
        loss_item.backward()
        optimizer.step()
        acc_sum += acc_item
        loss_sum += loss_item.item()
    acc_sum /= len(train_dataloader)
    loss_sum /= len(train_dataloader)

    writer.add_scalar(f"Train Set {iteration} - Acc", acc_sum, epoch)
    writer.add_scalar(f"Train Set {iteration} - Loss", loss_sum, epoch)

    return acc_sum, loss_sum


def val(writer, config, iteration, epoch, val_dataloader, device, encoder, classifier, lossfn, writer_enable=True):
    encoder.eval()
    classifier.eval()
    acc_sum = 0
    loss_sum = 0
    for inputs, labels in val_dataloader:
        with torch.no_grad():
            loss_item, acc_item = run_step(inputs, labels, device, encoder, classifier, lossfn,
                                           config["dataset"]["type"])
            acc_sum += acc_item
            loss_sum += loss_item.item()
    acc_sum /= len(val_dataloader)
    loss_sum /= len(val_dataloader)

    if writer_enable:
        writer.add_scalar(f"Val Set {iteration} - Acc", acc_sum, epoch)
        writer.add_scalar(f"Val Set {iteration} - Loss", loss_sum, epoch)

    return acc_sum, loss_sum


def test(logger, writer, config, iteration, test_dataloader, device, encoder, classifier, lossfn):
    acc_sum, loss_sum = val(writer, config, iteration, None, test_dataloader, device, encoder, classifier, lossfn,
                            writer_enable=False)
    logger.info("==> Test Set: Acc: {:.2f}%, Loss: {:.8f}".format(acc_sum, loss_sum))
    return acc_sum


def train_and_val(logger, writer, config, iteration, train_dataloader, device, encoder, classifier, lossfn, opts):
    best_record = {
        "epoch": 0,
        "acc": 0,
        "loss": 0,
        "encoder": None,
        "classifier": None
    }
    for epoch in tqdm(range(config["epoch"]), desc=f'Iteration: {iteration + 1}/{config["iteration"]}'):
        acc, loss = train(writer, config, iteration, epoch, train_dataloader, device, encoder, classifier, lossfn, opts)
        # acc, loss = val(writer, config, iteration, epoch, val_dataloader, device, encoder, classifier, lossfn)
        if epoch == 0 or (epoch > 0 and loss < best_record["loss"]):
            best_record["epoch"] = epoch
            best_record["acc"] = acc
            best_record["loss"] = loss
            best_record["encoder"] = encoder
            best_record["classifier"] = classifier
    logger.info("==> Train Set: Acc: {:.2f}%, Loss: {:.8f}".format(best_record["acc"], best_record["loss"]))
    return best_record


def finetune(config, logger, writer, iteration=0):
    set_seed(config["random_seed"])

    # train_dataloader, val_dataloader, test_dataloader = get_finetune_dataloader(config)
    train_dataloader, test_dataloader = get_finetune_dataloader(config)

    device = config["device"]
    encoder = create_model(config["encoder"]["root"], feature_dim=config["encoder"]["feature_dim"],
                           dtype=config["dataset"]["type"])
    if config["encoder"]["pretrain_path"]:
        encoder.load_state_dict(torch.load(config["encoder"]["pretrain_path"]))
    encoder = encoder.to(device)
    classifier = create_model(config["classifier"]["root"], in_dim=config["classifier"]["in_dim"],
                              num_classes=config["classifier"]["num_classes"])
    classifier = classifier.to(device)

    loss = torch.nn.CrossEntropyLoss()

    conf_opt = config["optimizer"]
    optimizers = ListApply([Adam(encoder.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"]),
                            Adam(classifier.parameters(), lr=conf_opt["lr"], weight_decay=conf_opt["weight_decay"])])
    best_record = train_and_val(logger, writer, config, iteration, train_dataloader, device, encoder, classifier, loss,
                                optimizers)
    acc = test(logger, writer, config, iteration, test_dataloader, device, best_record["encoder"],
               best_record["classifier"], loss)

    return acc


def downstream(config=None):
    config = finetune_config() if config is None else config
    logger, _, exp_path, save_path = get_logger_and_writer(
        os.path.join("runs/downstream", config["exp_name"], f"{config['dataset']['shot']}Shot"))
    config["exp_path"] = exp_path
    config["save_path"] = os.path.dirname(save_path)
    logger.info(f"==> Config: {config}")
    acc_list = []
    for i in range(config['iteration']):
        writer = SummaryWriter(os.path.join(exp_path, "writer", str(i)))
        logger.info("--------------------------------------------")
        logger.info(f"Iteration {i + 1}/{config['iteration']}")
        config["random_seed"] = config["random_seed"] + i
        acc = finetune(config, logger, writer, i)
        acc_list.append(acc)
        writer.close()
    df = pd.DataFrame(acc_list)
    df.to_excel(os.path.join(config["exp_path"], "test_result.xlsx"))
    shutil.copy(os.path.join(config["exp_path"], "test_result.xlsx"),
                os.path.join(config["save_path"], f"{config['exp_name']}_{config['dataset']['shot']}Shot.xlsx"))


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # for s in [5, 10, 15, 20, 25, 30]:
    #     ft_config = finetune_config(shot=s, dataset_name="wifi_ft62_run2", ablate="")
    #     downstream(ft_config)
    # ft_config = finetune_config(shot=10, encoder_name="DenseNet")
    # downstream(ft_config)
    # downstream()
    from copy import deepcopy

    def_ft_config = finetune_config(dataset_name="wifi_ft50")
    shots = [def_ft_config['dataset']['shot']] if isinstance(def_ft_config['dataset']['shot'], int) else def_ft_config['dataset']['shot']

    for s in shots:
        ft_config = deepcopy(def_ft_config)
        ft_config['dataset']['shot'] = s
        downstream(ft_config)
