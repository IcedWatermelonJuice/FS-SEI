# 只同时考虑分类损失
import torch
import torch.nn as nn
import torch.nn.functional as F
from get_dataset import TrainDataset_prepared, TestDataset_prepared
import argparse
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import _create_model_training_folder
import pandas as pd
import random
from utils import set_log_file_handler, get_pretrain_encoder
from config.config import get_config


def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = get_config("./config/config.yaml")
logger = set_log_file_handler(f"logs/{config['full_name']}")


def train(encoder, classifier, loss_nll, train_dataloader, optimizer_encoder, optimizer_classifier, epoch, device,
          writer, freeze_encoder):
    if freeze_encoder:
        encoder.eval()  # 冻结模型参数
    else:
        encoder.train()  # 允许更新模型参数
    classifier.train()  # 允许更新模型参数
    correct = 0
    nll_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optimizer_encoder.zero_grad()  # 清空优化器中梯度信息
        optimizer_classifier.zero_grad()

        # 分类损失反向生成encoder和classifier的梯度
        features = encoder(data)
        output = F.log_softmax(classifier(features), dim=1)
        nll_loss_batch = loss_nll(output, target)
        nll_loss_batch.backward()

        optimizer_encoder.step()
        optimizer_classifier.step()

        nll_loss += nll_loss_batch.item()

        output2 = F.softmax(classifier(features), dim=1)
        pred = output2.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()  # 求pred和target中对应位置元素相等的个数

    nll_loss /= len(train_dataloader)

    logger.info('Train Epoch: {} \tClass_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        nll_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', nll_loss, epoch)


def evaluate(encoder, classifier, loss_nll, val_dataloader, epoch, device, writer):
    encoder.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = classifier(encoder(data))
            output = F.log_softmax(output, dim=1)
            test_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    logger.info(
        fmt.format(
            test_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    writer.add_scalar('Accuracy/val', 100.0 * correct / len(val_dataloader.dataset), epoch)
    writer.add_scalar('Loss/val', test_loss, epoch)
    return test_loss


def test(encoder, classifier, loss_nll, test_dataloader, device):
    encoder.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = classifier(encoder(data))
            output = F.log_softmax(output, dim=1)
            test_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    logger.info(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)


def train_and_test(encoder, classifier, loss_nll, train_dataloader, val_dataloader, optim_encoder, optim_classifier,
                   epochs, device, writer, freeze_encoder):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    current_min_eval_loss = 1000000
    for epoch in range(1, epochs + 1):
        train(encoder, classifier, loss_nll, train_dataloader, optim_encoder, optim_classifier, epoch, device, writer,
              freeze_encoder)
        eval_loss = evaluate(encoder, classifier, loss_nll, val_dataloader, epoch, device, writer)
        if eval_loss < current_min_eval_loss:
            logger.info("The validation loss is falled from {} to {}, new model weight is saved.".format(
                current_min_eval_loss, eval_loss))
            current_min_eval_loss = eval_loss
            torch.save(encoder, os.path.join(model_checkpoints_folder, 'encoder.pth'))
            torch.save(classifier, os.path.join(model_checkpoints_folder, 'classifier.pth'))
        else:
            logger.info("The validation loss is not fall.")
        logger.info("------------------------------------------------")


def run(train_dataloader, val_dataloader, test_dataloader, epochs, device, writer, encoder_name, freeze_encoder,
        is_pretrained, pretrained_path, **opt_params):
    if encoder_name == "AlexNet":
        from AlexNetFeature import AlexNetFeature as Encoder
    else:
        from model_complexcnn import CVCNN as Encoder
    from model_complexcnn import Classifier
    encoder = Encoder()
    encoder = get_pretrain_encoder(encoder, logger, pretrained_path) if is_pretrained else encoder
    classifier = Classifier(6)

    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    loss_nll = nn.NLLLoss()
    if torch.cuda.is_available():
        loss_nll = loss_nll.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=opt_params['lr_encoder'], weight_decay=0)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=opt_params['lr_classifier'], weight_decay=0)

    train_and_test(encoder, classifier, loss_nll, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                   optim_encoder=optim_encoder, optim_classifier=optim_classifier, epochs=epochs, device=device,
                   writer=writer, freeze_encoder=freeze_encoder)

    # test
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    encoder = torch.load(os.path.join(model_checkpoints_folder, 'encoder.pth'))
    classifier = torch.load(os.path.join(model_checkpoints_folder, 'classifier.pth'))
    test_acc = test(encoder, classifier, loss_nll, test_dataloader, device)
    return test_acc


def main():
    logger.info("Experiment configuration: %s" % config)
    params = config['trainer']

    test_acc_all = []

    for i in range(config['iteration']):
        logger.info(f"iteration: {i}--------------------------------------------------------")
        set_seed(2023 + i)
        writer = SummaryWriter(f"runs/{config['full_name']}")
        _create_model_training_folder(writer, files_to_same=["./config/config.yaml", "train.py", "get_dataset.py"])
        device = torch.device("cuda:0")

        X_train, X_val, Y_train, Y_val = TrainDataset_prepared(2023 + i, config)
        X_test, Y_test = TestDataset_prepared(2023 + i, config)

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=True)
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=params['test_batch_size'], shuffle=True)

        # train
        test_acc = run(train_dataloader, val_dataloader, test_dataloader, epochs=params['max_epochs'], device=device,
                       writer=writer, encoder_name=params['encoder'], freeze_encoder=params['freeze'],
                       is_pretrained=params["pretrain"],
                       pretrained_path=params['pretrained_path'], **config['optimizer']['params'])
        test_acc_all.append(test_acc)
        writer.close()

    df = pd.DataFrame(test_acc_all)
    df.to_excel(f"test_result/{config['full_name']}.xlsx")


if __name__ == '__main__':
    main()
