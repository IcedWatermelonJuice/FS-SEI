import os
import shutil
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from utils.config import pretrain_config
from utils.utils import set_seed, get_logger_and_writer, create_model, ListApply, RecordTime, accuracy
from utils.get_dataset import get_pretrain_dataloader
from tqdm import tqdm


def run_step(config, inputs, rotator, device, encoder, classifier, cls, mse, nce, mtl):
    if len(inputs[0].shape) == 5:
        rot_inputs = inputs[0].reshape([-1, inputs[0].shape[2], inputs[0].shape[3], inputs[0].shape[4]])
        rot_labels = inputs[1].reshape(-1).long()
        inputs_len = len(inputs[0])
        ins_labels = torch.arange(inputs_len)
        rot_num = 4
    else:
        rot_inputs, rot_labels, ins_labels, rot_num, inputs_len = rotator(inputs[0])
    rot_inputs = rot_inputs.to(device)
    rot_labels = rot_labels.to(device)
    ins_labels = ins_labels.to(device)

    rot_features, inv_features = torch.split(encoder(rot_inputs), config["classifier"]["in_dim"], dim=1)
    rot_preds = classifier(rot_features)
    inv_preds = classifier(inv_features)
    inv_features_mean = torch.zeros((inputs_len, config["classifier"]["in_dim"]), device=device)
    for i in range(rot_num):
        inv_features_mean += inv_features[i * inputs_len:i * inputs_len + inputs_len, :]
    inv_features_mean = torch.mul(inv_features_mean, 1 / rot_num)
    inv_features_mean_full = inv_features_mean.repeat(4, 1)

    loss_items = []
    for loss_name in config["mtl"]["item"]:
        if loss_name == "cls":
            loss_items.append(cls(rot_preds, rot_labels))
        elif loss_name == "mse":
            loss_items.append(mse(inv_features, inv_features_mean_full))
        elif loss_name == "nce":
            loss_items.append(nce(inv_features_mean, ins_labels))
    loss_items.append(mtl(*loss_items))
    return loss_items, [accuracy(rot_preds, rot_labels), accuracy(inv_preds, rot_labels)]


def train(logger, writer, config, epoch, train_dataloader, rotator, device, encoder, classifier, cls, mse, nce, mtl,
          optimizers, schedulers):
    encoder.train()
    classifier.train()
    acc_sum = [0, 0]
    loss_sum = [0, 0, 0, 0]
    logger.info(f"==> lr = {optimizers[0].param_groups[0]['lr']}")
    for inputs in tqdm(train_dataloader, desc=f'Epoch: {epoch + 1}/{config["epoch"]}'):
        loss_items, acc_items = run_step(config, inputs, rotator, device, encoder, classifier, cls, mse, nce, mtl)
        optimizers.zero_grad()
        loss_items[-1].backward()
        optimizers.step()
        for idx, item in enumerate(acc_items):
            acc_sum[idx] += item
        for idx, item in enumerate(loss_items):
            loss_sum[idx] += item.item()
    schedulers.step()
    acc_sum = [v / len(train_dataloader) for v in acc_sum]
    loss_sum = [v / len(train_dataloader) for v in loss_sum]
    loss_sum_name = [n.upper() for n in config["mtl"]["item"]] + ["Total"]

    info_str = "==> Train Set: Rot-Acc: {:.2f}%, Inv-Acc: {:.2f}%, " + ": {:.8f}, ".join(loss_sum_name) + ": {:8f}"
    logger.info(info_str.format(*acc_sum, *loss_sum))

    writer.add_scalar("Train Set - Rot Acc", acc_sum[0], epoch)
    writer.add_scalar("Train Set - Inv Acc", acc_sum[1], epoch)
    for idx, loss_name in enumerate(loss_sum_name):
        writer.add_scalar(f"Train Set - {loss_name} Loss", loss_sum[idx], epoch)
    writer.add_scalar("Train Set - Total Loss (no MTL)", sum(loss_sum[0:-1]), epoch)


def val(logger, writer, config, epoch, val_dataloader, rotator, device, encoder, classifier, cls, mse, nce, mtl):
    encoder.eval()
    classifier.eval()
    acc_sum = [0, 0]
    loss_sum = [0, 0, 0, 0]
    for inputs in tqdm(val_dataloader, desc=f'Epoch: {epoch + 1}/{config["epoch"]}'):
        with torch.no_grad():
            loss_items, acc_items = run_step(config, inputs, rotator, device, encoder, classifier, cls, mse, nce, mtl)
            for idx, item in enumerate(acc_items):
                acc_sum[idx] += item
            for idx, item in enumerate(loss_items):
                loss_sum[idx] += item.item()

    acc_sum = [v / len(val_dataloader) for v in acc_sum]
    loss_sum = [v / len(val_dataloader) for v in loss_sum]
    loss_sum_name = [n.upper() for n in config["mtl"]["item"]] + ["Total"]

    info_str = "==> Val Set: Rot-Acc: {:.2f}%, Inv-Acc: {:.2f}%, " + ": {:.8f}, ".join(loss_sum_name) + ": {:8f}"
    logger.info(info_str.format(*acc_sum, *loss_sum))

    writer.add_scalar("Val Set - Rot Acc", acc_sum[0], epoch)
    writer.add_scalar("Val Set - Inv Acc", acc_sum[1], epoch)
    for idx, loss_name in enumerate(loss_sum_name):
        writer.add_scalar(f"Val Set - {loss_name} Loss", loss_sum[idx], epoch)
    writer.add_scalar("Val Set - Total Loss (no MTL)", sum(loss_sum[0:-1]), epoch)

    return acc_sum, loss_sum


def train_and_val(logger, writer, config, train_dataloader, val_dataloader, rotator, device, encoder, classifier, cls,
                  mse, nce, mtl, opts, schs):
    record_time = RecordTime(config["epoch"])
    best_record = {
        "epoch": 0,
        "acc": [0] * 2,
        "loss": [0] * (config["mtl"]["num"] + 1)
    }
    for epoch in range(config["epoch"]):
        logger.info("--------------------------------------------")
        logger.info(f"Epoch {epoch + 1}/{config['epoch']}")
        record_time.start()
        train(logger, writer, config, epoch, train_dataloader, rotator, device, encoder, classifier, cls, mse, nce, mtl,
              opts, schs)
        acc, loss = val(logger, writer, config, epoch, val_dataloader, rotator, device, encoder, classifier, cls, mse,
                        nce, mtl)
        if epoch == 0 or (epoch > 0 and sum(loss[0:-1]) < sum(best_record["loss"][0:-1])):
            best_record["epoch"] = epoch
            best_record["acc"] = acc
            best_record["loss"] = loss
            torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "best_encoder.pth"))
            logger.info(f"==> Best model is saved in epoch {epoch + 1}.")
        torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "final_encoder.pth"))
        logger.info("==> Time spend (current/mean/total/remain): {}/{}/{}/{}".format(*record_time.step()))
    shutil.copy(os.path.join(config["exp_path"], "best_encoder.pth"),
                os.path.join(config["save_path"], "best_encoder.pth"))
    shutil.copy(os.path.join(config["exp_path"], "final_encoder.pth"),
                os.path.join(config["save_path"], "final_encoder.pth"))
    logger.info("--------------------------------------------")
    logger.info(f"End. Best Record: {best_record}")


def pretext(config=None):
    config = pretrain_config() if config is None else config
    set_seed(config["random_seed"])
    logger, writer, exp_path, save_path = get_logger_and_writer(os.path.join("runs/pretext", config["exp_name"]))
    config["exp_path"] = exp_path
    config["save_path"] = save_path
    logger.info(f"==> Config: {config}")

    logger.info(f"==> Loading pretrain datasets.")
    train_dataloader, val_dataloader = get_pretrain_dataloader(config)

    logger.info(f"==> Generating new models.")
    rotator = create_model(config["rotator"]["root"], rot_angle=config["rotator"]["rot_angle"],
                           data_type=config["dataset"]["type"])

    device = config["device"]

    conf_en = config["encoder"]
    encoder = create_model(conf_en["root"], feature_dim=conf_en["feature_dim"], dtype=config["dataset"]["type"])
    encoder = encoder.to(device)

    conf_cl = config["classifier"]
    classifier = create_model(conf_cl["root"], in_dim=conf_cl["in_dim"], num_classes=conf_cl["num_classes"])
    classifier = classifier.to(device)

    logger.info(f"==> Creating loss and optimizers.")
    cls = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    nce = create_model(config["nce"]["root"], low_dim=config["nce"]["low_dim"], ndata=config["nce"]["n"],
                       nce_k=config["nce"]["k"], nce_t=config["nce"]["t"], nce_m=config["nce"]["m"]).to(device)
    mtl = create_model(config["mtl"]["root"], num=config["mtl"]["num"]).to(device)

    conf_opt = config["optimizer"]
    optimizers = ListApply([SGD(encoder.parameters(), lr=conf_opt["lr"], momentum=conf_opt["momentum"],
                                weight_decay=conf_opt["weight_decay"]),
                            SGD(classifier.parameters(), lr=conf_opt["lr"], momentum=conf_opt["momentum"],
                                weight_decay=conf_opt["weight_decay"]),
                            SGD(nce.norm.parameters(), lr=conf_opt["lr"], momentum=conf_opt["momentum"],
                                weight_decay=conf_opt["weight_decay"]),
                            SGD(mtl.parameters(), lr=conf_opt["lr"], momentum=conf_opt["momentum"],
                                weight_decay=conf_opt["weight_decay"])])

    schedulers = ListApply([StepLR(op, step_size=conf_opt["step_size"], gamma=conf_opt["gamma"]) for op in optimizers])

    logger.info(f"==> Starting tasks.")
    train_and_val(logger, writer, config, train_dataloader, val_dataloader, rotator, device, encoder, classifier, cls,
                  mse, nce, mtl, optimizers, schedulers)
    writer.close()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    pretext(pretrain_config(dataset_name="wifi_ft50"))
    # for loss in ["", "cls", "mse", "nce"]:
    #     config = pretrain_config(ablate=loss)
    #     pretext(config)
