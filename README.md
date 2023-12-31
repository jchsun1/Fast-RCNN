# Fast RCNN
Object detection using Fast RCNN on the 17flowers dataset.

# 博客
算法原理及思路请参考本人博客：https://www.cnblogs.com/Haitangr/p/17709548.html

# 数据集
17flowers据集, 官网下载地址：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# 文件说明
data						数据目录，data/source用于存放原始数据，data/ss用于存放选择性搜索生成数据\
doc							文档目录，用于存放参考论文\
Predict.py					预测代码，用于最终的目标检测\
requirements.txt			环境依赖\
SelectiveSearch.py			项目的选择性搜索实现\
SelectiveSearchCode.py		选择性搜索源码\
Train.py					各阶段模型训练代码\
Utils.py					用于IoU计算、数据集加载、模型定义、图像绘制等功能的实现

# 程序运行流程
(1) 运行SelectiveSearch.py, 根据2flowers数据和标注在data/ss目录下生成推荐区域信息\
(2) 运行Train.py，程序会在ss数据集上进行分类和回归多任务模型训练，并生成./model/路径保存模型\
(3) 运行Predict.py，对数据集进行预测，生成./predict/目录保存预测结果
