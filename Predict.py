import os
import torch
import numpy as np
import skimage.io as io
from Utils import GenDataSet, draw_box
from torch.nn.functional import softmax
from SelectiveSearch import SelectiveSearch as ss
from Config import *


def rectify_bbox(ss_boxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    修正边界框位置
    :param ss_boxes: 边界框
    :param offsets: 边界框偏移值
    :return: 位置修正后的边界框
    """
    # 和Utils.GenDataSet.calc_offsets过程相反
    ss_boxes = ss_boxes.astype("float32")
    ss_boxes[:, 0] = ss_boxes[:, 2] * offsets[:, 0] + ss_boxes[:, 0]
    ss_boxes[:, 1] = ss_boxes[:, 3] * offsets[:, 1] + ss_boxes[:, 1]
    ss_boxes[:, 2] = np.exp(offsets[:, 2]) * ss_boxes[:, 2]
    ss_boxes[:, 3] = np.exp(offsets[:, 3]) * ss_boxes[:, 3]
    boxes = ss_boxes.astype("int")
    return boxes


def map_bbox_to_img(boxes: np.ndarray, src_img: np.ndarray, im_width: int, im_height: int):
    """
    根据缩放比例将边界框映射回原图
    :param boxes: 缩放后图像上的边界框
    :param src_img: 原始图像
    :param im_width: 缩放后图像宽度
    :param im_height: 缩放后图像高度
    :return: boxes->映射到原图像上的边界框
    """
    rows, cols = src_img.shape[:2]
    scale_ratio = np.array([cols / im_width, rows / im_height, cols / im_width, rows / im_height])
    boxes = (boxes * scale_ratio).astype("int")
    return boxes


def nms(bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    非极大值抑制去除冗余边界框
    :param bboxes: 目标边界框
    :param scores: 目标得分
    :param threshold: 阈值
    :return: keep->保留下的有效边界框
    """
    # 获取边界框和分数
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2] - 1
    y2 = bboxes[:, 1] + bboxes[:, 3] - 1
    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 逆序排序
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        # 取分数最高的一个
        i = order[0]
        keep.append(bboxes[i])

        if order.size == 1:
            break
        # 计算相交区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算IoU
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的bbox
        idx = np.where(iou <= threshold)[0]
        order = order[idx + 1]
    keep = np.array(keep)
    return keep


def predict(network, im, im_width, im_height, device, nms_thresh=None, save_name=None):
    """
    模型预测
    :param network: 模型结构
    :param im: 输入图像
    :param im_width: 模型输入图像宽度
    :param im_height: 模型输入图像长度
    :param device: CPU/GPU
    :param nms_thresh: 非极大值抑制阈值
    :param save_name: 保存文件名
    :return: None
    """
    network.eval()
    # 生成推荐区域
    ss_boxes_src = ss.cal_proposals(img=im)
    # 数据归一化
    im_norm = GenDataSet.normalize(im=im)
    # 将图像缩放固定大小, 并将边界框映射到缩放后图像上
    im_rsz, ss_boxes_rsz = GenDataSet.resize(im=im_norm, im_width=im_width, im_height=im_height, ss_boxes=ss_boxes_src, gt_boxes=None)
    im_tensor = torch.tensor(np.transpose(im_rsz, (2, 0, 1))).unsqueeze(0).to(device)
    ss_boxes_tensor = torch.tensor(ss_boxes_rsz).to(device)

    # 模型输入为[img: Tensor, ss_boxes: list(Tensor)]
    inputs = [im_tensor, [ss_boxes_tensor]]

    with torch.no_grad():
        outputs = network(inputs)

    # 计算各个类别的分类得分
    scores = softmax(input=outputs[0], dim=1)
    scores = scores.cpu().numpy()

    # 获取位置偏移
    offsets = outputs[1]
    offsets = offsets.cpu().numpy()
    # 根据模型计算出的offsets对推荐区域边界框位置进行修正
    out_boxes = rectify_bbox(ss_boxes=ss_boxes_rsz, offsets=offsets)
    # 将边界框位置映射回原始图像
    out_boxes = map_bbox_to_img(boxes=out_boxes, src_img=im, im_width=im_width, im_height=im_height)

    # 边界框筛选
    predicted_boxes = []
    for i in range(1, CLASSES):
        # 获取当前类别目标得分
        cur_obj_scores = scores[:, i]
        # 只选取置信度满足阈值要求的预测框
        idx = cur_obj_scores >= CONFIDENCE_THRESH
        valid_scores = cur_obj_scores[idx]
        valid_out_boxes = out_boxes[idx]

        # 遍历物体类别, 对每个类别的边界框预测结果进行非极大值抑制
        if nms_thresh is not None:
            used_boxes = nms(bboxes=valid_out_boxes, scores=valid_scores, threshold=nms_thresh)
        else:
            used_boxes = valid_out_boxes
        # 可能存在值不符合要求的情况, 需要剔除
        for j in range(used_boxes.shape[0]):
            if used_boxes[j, 0] < 0 or used_boxes[j, 1] < 0 or used_boxes[j, 2] <= 0 or used_boxes[j, 3] <= 0:
                continue
            predicted_boxes.append(used_boxes[j])

    draw_box(img=im, boxes=predicted_boxes, save_name=save_name)
    return None


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "./model/model.pth"
    model = torch.load(model_path, map_location=device)

    test_root = "./data/"
    for roots, dirs, files in os.walk(test_root):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            save_name = file.split(".")[0]
            im_path = os.path.join(roots, file)
            im = io.imread(im_path)
            predict(network=model, im=im, im_width=IM_SIZE, im_height=IM_SIZE, device=device, nms_thresh=NMS_THRESH, save_name=save_name)

