import os
import numpy as np
import pandas as pd
import cv2 as cv
import shutil
from Utils import cal_IoU
from skimage import io
from multiprocessing import Process
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Config import *


class SelectiveSearch:
    def __init__(self, root, max_pos_regions: int = None, max_neg_regions: int = None, threshold=0.5):
        """
        采用ss方法生成候选区域文件
        :param root: 训练/验证数据集所在路径
        :param max_pos_regions: 每张图片最多产生的正样本候选区域个数, None表示不进行限制
        :param max_neg_regions: 每张图片最多产生的负样本候选区域个数, None表示不进行限制
        :param threshold: IoU进行正负样本区分时的阈值
        """
        self.source_root = os.path.join(root, 'source')
        self.ss_root = os.path.join(root, 'ss')
        self.csv_path = os.path.join(self.source_root, "gt_loc.csv")
        self.max_pos_regions = max_pos_regions
        self.max_neg_regions = max_neg_regions
        self.threshold = threshold
        self.info = None

    @staticmethod
    def cal_proposals(img) -> np.ndarray:
        """
        计算后续区域坐标
        :param img: 原始输入图像
        :return: candidates, 候选区域坐标矩阵n*4维, 每列分别对应[x, y, w, h]
        """
        # 生成候选区域
        ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        proposals = ss.process()
        candidates = set()
        # 对区域进行限制
        for region in proposals:
            rect = tuple(region)
            if rect in candidates:
                continue
            candidates.add(rect)
        candidates = np.array(list(candidates))
        return candidates

    def save(self, num_workers=1, method="thread"):
        """
        生成目标区域并保存
        :param num_workers: 进程或线程数
        :param method: 多进程-process或者多线程-thread
        :return: None
        """
        self.info = pd.read_csv(self.csv_path, header=0, index_col=None)
        index = self.info.index.to_list()
        span = len(index) // num_workers

        # 多进程生成图像
        if "process" in method.lower():
            print("=" * 8 + "开始多进程生成候选区域图像" + "=" * 8)
            processes = []
            for i in range(num_workers):
                if i != num_workers - 1:
                    p = Process(target=self.save_proposals, kwargs={'index': index[i * span: (i + 1) * span]})
                else:
                    p = Process(target=self.save_proposals, kwargs={'index': index[i * span:]})
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        # 多线程生成图像
        elif "thread" in method.lower():
            print("=" * 8 + "开始多线程生成候选区域图像" + "=" * 8)
            threads = []
            for i in range(num_workers):
                if i != num_workers - 1:
                    thread = threading.Thread(target=self.save_proposals, kwargs={'index': index[i * span: (i + 1) * span]})
                else:
                    thread = threading.Thread(target=self.save_proposals, kwargs={'index': index[i * span: (i + 1) * span]})
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        else:
            print("=" * 8 + "开始生成候选区域图像" + "=" * 8)
            self.save_proposals(index=index)
        return None

    def save_proposals(self, index, show_fig=False):
        """
        生成候选区域图片并保存相关信息
        :param index: 文件index
        :param show_fig: 是否展示后续区域划分结果
        :return: None
        """
        for row in index:
            name = self.info.iloc[row, 0]
            label = self.info.iloc[row, 1]
            # gt值为[x, y, w, h]
            gt_box = self.info.iloc[row, 2:].values
            im_path = os.path.join(self.source_root, name)
            img = io.imread(im_path)

            # 计算推荐区域坐标矩阵[x, y, w, h]
            proposals = self.cal_proposals(img=img)
            # 计算proposals与gt的IoU结果
            IoU = cal_IoU(proposals, gt_box)
            # 根据IoU阈值将proposals图像划分到正负样本集
            p_boxes = proposals[np.where(IoU >= self.threshold)]
            n_boxes = proposals[np.where(IoU < self.threshold)]

            # 展示proposals结果
            if show_fig:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
                ax.imshow(img)
                for (x, y, w, h) in p_boxes:
                    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)
                for (x, y, w, h) in n_boxes:
                    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
                    ax.add_patch(rect)
                plt.show()

            # 根据图像名称创建文件夹, 保存原始图片/真实边界框/推荐区域边界框/推荐区域标签信息
            folder = name.split("/")[-1].split(".")[0]
            save_root = os.path.join(self.ss_root, folder)
            os.makedirs(save_root, exist_ok=True)
            # 保存原始图像
            im_save_path = os.path.join(save_root, folder + ".jpg")
            io.imsave(fname=im_save_path, arr=img, check_contrast=False)

            # loc.csv用于存储边界框信息
            loc_path = os.path.join(save_root, "ss_loc.csv")

            # 记录正负样本信息
            locations = []
            header = ["label", "px", "py", "pw", "ph", "gx", "gy", "gw", "gh"]
            num_p = num_n = 0
            for p_box in p_boxes:
                num_p += 1
                locations.append([label, *p_box, *gt_box])
                if self.max_pos_regions is None:
                    continue
                if num_p >= self.max_pos_regions:
                    break

            # 记录负样本信息, 负样本为背景, label置为0
            for n_box in n_boxes:
                num_n += 1
                locations.append([0, *n_box, *gt_box])
                if self.max_neg_regions is None:
                    continue
                if num_n >= self.max_neg_regions:
                    break
            print("{name}: {num_p}个正样本, {num_n}个负样本".format(name=name, num_p=num_p, num_n=num_n))
            pf = pd.DataFrame(locations)
            pf.to_csv(loc_path, header=header, index=False)


if __name__ == '__main__':
    data_root = "./data"
    ss_root = os.path.join(data_root, "ss")
    if os.path.exists(ss_root):
        print("正在删除{}目录下原有数据".format(ss_root))
        shutil.rmtree(ss_root)
    print("正在利用选择性搜索方法创建数据集: {}".format(ss_root))
    select = SelectiveSearch(root=data_root, max_pos_regions=MAX_POSITIVE, max_neg_regions=MAX_NEGATIVE, threshold=IOU_THRESH)
    select.save(num_workers=os.cpu_count(), method="thread")
