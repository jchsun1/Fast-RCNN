import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from Utils import FastRCNN, GenDataSet
from torch import Tensor
from Config import *


def multitask_loss(output: tuple, labels: Tensor, offsets: Tensor, criterion: list, alpha: float = 1.0):
    """
    计算多任务损失
    :param output: 模型输出
    :param labels: 边界框标签
    :param offsets: 边界框偏移值
    :param criterion: 损失函数
    :param alpha: 损失函数
    :return:
    """
    output_cls, output_reg = output
    # 计算分类损失
    loss_cls = criterion[0](output_cls, labels)
    # 计算正样本的回归损失
    output_reg_valid = output_reg[labels != 0]
    offsets_valid = offsets[labels != 0]
    loss_reg = criterion[1](output_reg_valid, offsets_valid)
    # 损失加权
    loss = loss_cls + alpha * loss_reg
    return loss


def train(data_set, network, num_epochs, optimizer, scheduler, criterion, device, train_rate=0.8):
    """
    模型训练
    :param data_set: 训练数据集
    :param network: 网络结构
    :param num_epochs: 训练轮次
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param criterion: 损失函数
    :param device: CPU/GPU
    :param train_rate: 训练集比例
    :return: None
    """
    os.makedirs('./model', exist_ok=True)
    network = network.to(device)
    best_loss = np.inf
    print("=" * 8 + "开始训练模型" + "=" * 8)
    # 计算训练batch数量
    batch_num = len(data_set)
    train_batch_num = round(batch_num * train_rate)
    # 记录训练过程中每一轮损失和准确率
    train_loss_all, val_loss_all, train_acc_all, val_acc_all = [], [], [], []

    for epoch in range(num_epochs):
        # 记录train/val分类准确率和总损失
        num_train_acc = num_val_acc = num_train_loss = num_val_loss = 0
        train_loss = val_loss = 0.0
        train_corrects = val_corrects = 0

        for step, batch_data in enumerate(data_set):
            # 读取数据
            ims, labels, ss_boxes, offsets = batch_data
            ims = ims.to(device)
            labels = labels.to(device)
            ss_boxes = [ss.to(device) for ss in ss_boxes]
            offsets = offsets.to(device)
            # 模型输入为全图和推荐区域边界框, 即[ims: Tensor, ss_boxes: list[Tensor]]
            inputs = [ims, ss_boxes]
            if step < train_batch_num:
                # train
                network.train()
                output = network(inputs)
                loss = multitask_loss(output=output, labels=labels, offsets=offsets, criterion=criterion)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算每个batch分类正确的数量和loss
                label_hat = torch.argmax(output[0], dim=1)
                train_corrects += (label_hat == labels).sum().item()
                num_train_acc += labels.size(0)
                # 计算每个batch总损失
                train_loss += loss.item() * ims.size(0)
                num_train_loss += ims.size(0)

            else:
                # validation
                network.eval()
                with torch.no_grad():
                    output = network(inputs)
                    loss = multitask_loss(output=output, labels=labels, offsets=offsets, criterion=criterion)

                    # 计算每个batch分类正确的数量和loss和
                    label_hat = torch.argmax(output[0], dim=1)
                    val_corrects += (label_hat == labels).sum().item()
                    num_val_acc += labels.size(0)
                    val_loss += loss.item() * ims.size(0)
                    num_val_loss += ims.size(0)

        scheduler.step()
        # 记录loss和acc变化曲线
        train_loss_all.append(train_loss / num_train_loss)
        val_loss_all.append(val_loss / num_val_loss)
        train_acc_all.append(100 * train_corrects / num_train_acc)
        val_acc_all.append(100 * val_corrects / num_val_acc)
        print("Epoch:[{:0>3}|{}]  train_loss:{:.3f}  train_acc:{:.2f}%  val_loss:{:.3f}  val_acc:{:.2f}%".format(
            epoch + 1, num_epochs,
            train_loss_all[-1], train_acc_all[-1],
            val_loss_all[-1], val_acc_all[-1]
        ))

        # 保存模型
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            save_path = os.path.join("./model", "model.pth")
            torch.save(network, save_path)

    # 绘制训练曲线
    fig_path = os.path.join("./model/",  "train_curve.png")
    plt.subplot(121)
    plt.plot(range(num_epochs), train_loss_all, "r-", label="train")
    plt.plot(range(num_epochs), val_loss_all, "b-", label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(range(num_epochs), train_acc_all, "r-", label="train")
    plt.plot(range(num_epochs), val_acc_all, "b-", label="val")
    plt.title("Acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return None


if __name__ == "__main__":
    if not os.path.exists("./data/ss"):
        raise FileNotFoundError("数据不存在, 请先运行SelectiveSearch.py生成目标区域")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastRCNN(num_classes=CLASSES, in_size=IM_SIZE, pool_size=POOL_SIZE, spatial_scale=SCALE, device=device)
    exit(0)
    criterion = [nn.CrossEntropyLoss(), nn.SmoothL1Loss()]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP, gamma=GAMMA)

    model_root = "./model"
    os.makedirs(model_root, exist_ok=True)

    # 在生成的ss数据上进行预训练
    train_root = "./data/ss"
    train_set = GenDataSet(root=train_root, im_width=IM_SIZE, im_height=IM_SIZE, batch_size=BATCH_SIZE, shuffle=True)

    train(data_set=train_set, network=model, num_epochs=EPOCHS, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device)


