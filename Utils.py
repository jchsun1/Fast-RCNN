import os
import cv2 as cv
import numpy as np
import pandas as pd
import random
import torch
from skimage import io
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import Tensor
from typing import Union


def cal_IoU(boxes: np.ndarray, gt_box) -> np.ndarray:
    """
    计算推荐区域与真值的IoU
    :param boxes: 推荐区域边界框, n*4维数组, 列对应左上和右下两个点坐标[x1, y1, w, h]
    :param gt_box: 真值, 对应左上和右下两个点坐标[x1, y1, w, h]
    :return: iou, 推荐区域boxes与真值的IoU结果
    """
    # 复制矩阵防止直接引用修改原始值
    bbox = boxes.copy()
    gt = gt_box.copy()

    # 将宽度转换成坐标
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    gt[2] = gt[0] + gt[2]
    gt[3] = gt[1] + gt[3]

    box_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    inter_w = np.minimum(bbox[:, 2], gt[2]) - np.maximum(bbox[:, 0], gt[0])
    inter_h = np.minimum(bbox[:, 3], gt[3]) - np.maximum(bbox[:, 1], gt[1])

    inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
    union = box_area + gt_area - inter
    iou = inter / union
    return iou


def draw_box(img, boxes=None, save_name: str = None):
    """
    在图像上绘制边界框
    :param img: 输入图像
    :param boxes: bbox坐标, 列分别为[x, y, w, h]
    :param save_name: 保存bbox图像名称, None-不保存
    :return: None
    """
    plt.imshow(img)
    axis = plt.gca()
    if boxes is not None:
        for box in boxes:
            rect = patches.Rectangle((int(box[0]), int(box[1])), int(box[2]), int(box[3]), linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rect)
    if save_name is not None:
        os.makedirs("./predict", exist_ok=True)
        plt.savefig("./predict/" + save_name + ".jpg")
    plt.show()
    return None


class FastRCNN(nn.Module):
    def __init__(self, num_classes, in_size, pool_size, spatial_scale, drop=0.1, device=None):
        """
        初始化
        :param num_classes: 类别数 N(物体类别) + 1(背景)
        :param in_size: 模型输入大小
        :param pool_size: roi池化目标大小
        :param spatial_scale: 模型空间缩放比例
        :param drop: drop rate
        :param device: CPU/GPU
        """
        super(FastRCNN, self).__init__()
        self.in_size = in_size
        self.pool_size = pool_size
        self.spatial_scale = spatial_scale
        self.device = device

        # 自适应最大值池化将推荐区域特征池化到固定大小
        self.pool = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        # 采用vgg16_bn作为backbone
        self.features = models.vgg16_bn(pretrained=True).features

        # 分类器, 输入为vgg16的512通道特征经过roi_pool的结果
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * self.pool_size * self.pool_size, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=32, out_features=num_classes)
        )

        # 回归器, 输入为vgg16的512通道特征经过roi_pool的结果
        self.regressor = nn.Sequential(
            nn.Linear(in_features=512 * self.pool_size * self.pool_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=64, out_features=4)
        )

    def roi_pool(self, im_features: Tensor, ss_boxes: list):
        """
        提取推荐区域特征图并缩放到固定大小
        :param im_features: backbone输出的图像特征->[batch, channel, rows, cols]
        :param ss_boxes: 推荐区域边界框信息->[batch, num, 4]
        :return: 推荐区域特征
        """
        roi_features = []
        for im_idx, im_feature in enumerate(im_features):
            im_boxes = ss_boxes[im_idx]
            for box in im_boxes:
                # 输入全图经过backbone后空间位置需进行缩放, 利用空间缩放比例将box位置对应到feature上
                fx, fy, fw, fh = [int(p / self.spatial_scale) for p in box]
                # 缩放后维度不足1个pixel, 是由于int取整导致, 仍取1个pixel防止维度为0
                if fw == 0:
                    fw = 1
                if fh == 0:
                    fh = 1
                # 在特征图上提取候选区域对应的区域特征
                roi_feature = im_feature[:, fy: fy + fh, fx: fx + fw]
                # 将区域特征池化到固定大小
                roi_feature = self.pool(roi_feature)
                # 将池化后特征展开方便后续送入分类器和回归器
                roi_feature = roi_feature.view(-1)
                roi_features.append(roi_feature)

        # 转换成tensor
        roi_features = torch.stack(roi_features)
        return roi_features

    def forward(self, x):
        """
        前向计算
        :param x: 输入数据图片和对应的边界框列表
        :return: 分类器和回归器结果
        """
        img, ss_boxes = x
        # backbone提取特征
        img_features = self.features(img)
        # 提取推荐区域特征并空间池化
        roi_features = self.roi_pool(im_features=img_features, ss_boxes=ss_boxes)
        # 将roi特征送入分类模型进行分类, 送入回归模型计算边界框偏移值
        classifier_result = self.classifier(roi_features)
        regressor_result = self.regressor(roi_features)
        return classifier_result, regressor_result


class GenDataSet:
    def __init__(self, root, im_width, im_height, batch_size, shuffle=False, prob_vertical_flip=0.5, prob_horizontal_flip=0.5):
        """
        初始化GenDataSet
        :param root: 数据路径
        :param im_width: 目标图片宽度
        :param im_height: 目标图片高度
        :param batch_size: 批数据大小
        :param shuffle: 是否随机打乱批数据
        :param prob_vertical_flip: 随机垂直翻转概率
        :param prob_horizontal_flip: 随机水平翻转概率
        """
        self.root = root
        self.im_width, self.im_height = (im_width, im_height)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flist = self.get_flist()
        self.num_batch = self.calc_num_batch()
        self.prob_vertical_flip = prob_vertical_flip
        self.prob_horizontal_flip = prob_horizontal_flip

    def get_flist(self) -> list:
        """
        获取原始图像和推荐区域边界框
        :return: [图像, 边界框]列表
        """
        flist = []
        for roots, dirs, files in os.walk(self.root):
            for file in files:
                if not file.endswith(".jpg"):
                    continue
                img_path = os.path.join(roots, file)
                doc_path = os.path.join(roots, "ss_loc.csv")
                if not os.path.exists(doc_path):
                    continue
                flist.append((img_path, doc_path))
        return flist

    def calc_num_batch(self) -> int:
        """
        计算batch数量
        :return: 批数据数量
        """
        total = len(self.flist)
        if total % self.batch_size == 0:
            num_batch = total // self.batch_size
        else:
            num_batch = total // self.batch_size + 1
        return num_batch

    @staticmethod
    def normalize(im: np.ndarray) -> np.ndarray:
        """
        将图像数据归一化
        :param im: 输入图像->uint8
        :return: 归一化图像->float32
        """
        if im.dtype != np.uint8:
            raise TypeError("uint8 img is required.")
        else:
            im = im / 255.0
        im = im.astype("float32")
        return im

    @staticmethod
    def calc_offsets(ss_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """
        计算候选区域与真值间的位置偏移
        :param ss_boxes: 候选边界框
        :param gt_boxes: 真值
        :return: 边界框偏移值
        """
        offsets = np.zeros_like(ss_boxes, dtype="float32")
        # 基于比例计算偏移值可以不受位置大小的影响
        offsets[:, 0] = (gt_boxes[:, 0] - ss_boxes[:, 0]) / ss_boxes[:, 2]
        offsets[:, 1] = (gt_boxes[:, 1] - ss_boxes[:, 1]) / ss_boxes[:, 3]
        # 使用log计算w/h的偏移值, 避免值过大
        offsets[:, 2] = np.log(gt_boxes[:, 2] / ss_boxes[:, 2])
        offsets[:, 3] = np.log(gt_boxes[:, 3] / ss_boxes[:, 3])
        return offsets

    @staticmethod
    def resize(im: np.ndarray, im_width: int, im_height: int, ss_boxes: np.ndarray, gt_boxes: Union[np.ndarray, None]):
        """
        对图像进行缩放
        :param im: 输入图像
        :param im_width: 目标图像宽度
        :param im_height: 目标图像高度
        :param ss_boxes: 推荐区域边界框->[n, 4]
        :param gt_boxes: 真实边界框->[n, 4]
        :return: 图像和两种边界框经过缩放后的结果
        """
        rows, cols = im.shape[:2]
        # 图像缩放
        im = cv.resize(src=im, dsize=(im_width, im_height), interpolation=cv.INTER_CUBIC)
        # 计算缩放过程中(x, y, w, h)尺度缩放比例
        scale_ratio = np.array([im_width / cols, im_height / rows, im_width / cols, im_height / rows])

        # 边界框也等比例缩放
        ss_boxes = (ss_boxes * scale_ratio).astype("int")
        if gt_boxes is None:
            return im, ss_boxes
        gt_boxes = (gt_boxes * scale_ratio).astype("int")
        return im, ss_boxes, gt_boxes

    def random_horizontal_flip(self, im: np.ndarray, ss_boxes: np.ndarray, gt_boxes: np.ndarray):
        """
        随机水平翻转图像
        :param im: 输入图像
        :param ss_boxes: 推荐区域边界框
        :param gt_boxes: 边界框真值
        :return: 翻转后图像和边界框结果
        """
        if random.uniform(0, 1) < self.prob_horizontal_flip:
            rows, cols = im.shape[:2]
            # 左右翻转图像
            im = np.fliplr(im)
            # 边界框位置重新计算
            ss_boxes[:, 0] = cols - 1 - ss_boxes[:, 0] - ss_boxes[:, 2]
            gt_boxes[:, 0] = cols - 1 - gt_boxes[:, 0] - gt_boxes[:, 2]
        else:
            pass
        return im, ss_boxes, gt_boxes

    def random_vertical_flip(self, im: np.ndarray, ss_boxes: np.ndarray, gt_boxes: np.ndarray):
        """
        随机垂直翻转图像
        :param im: 输入图像
        :param ss_boxes: 推荐区域边界框
        :param gt_boxes: 边界框真值
        :return: 翻转后图像和边界框结果
        """
        if random.uniform(0, 1) < self.prob_vertical_flip:
            rows, cols = im.shape[:2]
            # 上下翻转图像
            im = np.flipud(im)
            # 重新计算边界框位置
            ss_boxes[:, 1] = rows - 1 - ss_boxes[:, 1] - ss_boxes[:, 3]
            gt_boxes[:, 1] = rows - 1 - gt_boxes[:, 1] - gt_boxes[:, 3]
        else:
            pass
        return im, ss_boxes, gt_boxes

    def get_fdata(self):
        """
        数据集准备
        :return: 数据列表
        """
        fdata = []
        if self.shuffle:
            random.shuffle(self.flist)

        for num in range(self.num_batch):
            # 按照batch大小读取数据
            cur_flist = self.flist[num * self.batch_size: (num + 1) * self.batch_size]
            # 记录当前batch的图像/推荐区域标签/边界框/位置偏移
            cur_ims, cur_labels, cur_ss_boxes, cur_offsets = [], [], [], []
            for img_path, doc_path in cur_flist:
                # 读取图像
                img = io.imread(img_path)
                # 读取边界框并堆积打乱框顺序
                ss_info = pd.read_csv(doc_path, header=0, index_col=None)
                ss_info = ss_info.sample(frac=1).reset_index(drop=True)
                labels = ss_info.label.to_list()
                ss_boxes = ss_info.iloc[:, 1: 5].values
                gt_boxes = ss_info.iloc[:, 5: 9].values

                # 数据归一化
                img = self.normalize(im=img)
                # 随机翻转数据增强
                img, ss_boxes, gt_boxes = self.random_horizontal_flip(im=img, ss_boxes=ss_boxes, gt_boxes=gt_boxes)
                img, ss_boxes, gt_boxes = self.random_vertical_flip(im=img, ss_boxes=ss_boxes, gt_boxes=gt_boxes)
                # 将图像缩放到统一大小
                img, ss_boxes, gt_boxes = self.resize(im=img, im_width=self.im_width, im_height=self.im_height, ss_boxes=ss_boxes, gt_boxes=gt_boxes)

                # 计算最终坐标偏移值
                offsets = self.calc_offsets(ss_boxes=ss_boxes, gt_boxes=gt_boxes)

                # 转换为tensor
                im_tensor = torch.tensor(np.transpose(img, (2, 0, 1)))
                ss_boxes_tensor = torch.tensor(data=ss_boxes)

                cur_ims.append(im_tensor)
                cur_labels.extend(labels)
                cur_ss_boxes.append(ss_boxes_tensor)
                cur_offsets.extend(offsets)

            # 每个batch数据放一起方便后续训练调用
            cur_ims = torch.stack(cur_ims)
            cur_labels = torch.tensor(cur_labels)
            cur_offsets = torch.tensor(np.array(cur_offsets))
            fdata.append([cur_ims, cur_labels, cur_ss_boxes, cur_offsets])
        return fdata

    def __len__(self):
        # 以batch数量定义数据集大小
        return self.num_batch

    def __iter__(self):
        self.fdata = self.get_fdata()
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.num_batch:
            raise StopIteration
        # 生成当前batch数据
        value = self.fdata[self.index]
        self.index += 1
        return value

