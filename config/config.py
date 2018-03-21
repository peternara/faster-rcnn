# coding=utf-8
# rpn anchor
base_anchor_size = 256
anchor_stride = 16
scales = [0.5, 1, 2]
aspect_ratios = [0.5, 1, 2]

# rpn sample
iou_threshold = 0.7
max_output_size = 2000

# resize
resize_shorter_edge_size = 600

extractor = "vgg16"

rpn_max_region_proposals = 2000
rpn_nms_iou_threshold = 0.7

# 第二阶段
box_classifier_use_dropout = True
dropout_keep_prob = 0.5

# 类别
num_classes = 10

# anchor正负样本的阈值
positive_thresold = 0.7
negative_thresold = 0.3
# anchor 正负样本采样
positive_fraction = 0.5
batch_size = 1000
