# 模型输入图像大小
IM_SIZE = 512

# roi-pool大小
POOL_SIZE = 7

# 卷积网络尺度缩放大小
# VGG16共5个2*2池化, 缩放32倍
SCALE = 32

# 训练过程参数
CLASSES = 3
BATCH_SIZE = 4
EPOCHS = 200
LR = 0.001
STEP = 10
GAMMA = 0.6

# ss和nms阈值
MAX_POSITIVE = None
MAX_NEGATIVE = 40
IOU_THRESH = 0.5
NMS_THRESH = 0.05
CONFIDENCE_THRESH = 0.8

