# coding=utf-8
from __future__ import division
import tensorflow as tf

from anchor import anchor_clip, sample
from anchor import code, box_utils
from classifier import box_classifier_extrator, box_predict
from config import config
from nets import feature_extractor
from region_proposals import rpn
from resizer import img_resizer
from roi_pooling import roi_pooling


def model(images, ground_truth_boxes, is_training=True, scope='faster-rcnn'):
    with tf.name_scope(scope, 'faster-rcnn'):
        # 第一阶段
        # 1.缩放图片
        resized_imgs = img_resizer.resizer(images, config.resize_shorter_edge_size, scope='resize_image')
        image_shape = tf.shape(resized_imgs)
        image_height, image_width = image_shape[1], image_shape[2]

        # 2.抽取特征
        feature_maps = feature_extractor.extract(resized_imgs, config.extractor, scope='feature_extract')
        feature_shape = tf.shape(feature_maps)

        # 3.rpn网络预测分数，边框坐标，以及生成anchor
        rpn_scores, rpn_anchor_encodes, rpn_anchors = rpn.region_proposals_network(feature_maps, scope='region')

        # 4.剪裁anchor，在训练的时候去掉超出图像大小的anchor，预测的时候剪裁超出图像部分的anchor
        clip_window = tf.to_float(tf.stack([0, 0, image_height - 1, image_width - 1]))
        if is_training:
            # 去掉超出图像的框
            keep = anchor_clip.prune_outside_window(rpn_anchors, clip_window)
            rpn_scores = tf.gather(rpn_scores, keep)
            rpn_anchors = tf.gather(rpn_anchors, keep)
            rpn_anchor_encodes = tf.gather(rpn_anchor_encodes, keep)
        else:
            rpn_anchors, keep = anchor_clip.anchor_clip(rpn_anchors, clip_window)
            rpn_scores = tf.gather(rpn_scores, keep)
            rpn_anchor_encodes = tf.gather(rpn_anchor_encodes, keep)

        # 第二阶段
        # 5.anchor解码
        proposal_boxes = code.anchor_decode(rpn_anchors, rpn_anchor_encodes)

        # 6.剪切 proposal_boxes
        proposal_boxes, keep = anchor_clip.anchor_clip(proposal_boxes, clip_window)
        proposal_boxe_scores = tf.gather(rpn_scores, keep)

        # 7去重复框 非极大抑制,对object进行抑制
        keep = tf.image.non_max_suppression(proposal_boxes, proposal_boxe_scores[:, 1], config.rpn_max_region_proposals,
                                            config.rpn_nms_iou_threshold)
        proposal_boxes = tf.gather(proposal_boxes, keep)
        proposal_boxe_scores = tf.gather(proposal_boxe_scores, keep)

        # 8.在训练阶段，需要proposal_boxes采样，生成一批样本，用于分类网络的训练
        if is_training:
            proposal_boxes, proposal_scores = sample.sample_box_classifier_batch(proposal_boxes, proposal_boxe_scores,
                                                                                 ground_truth_boxes)

        # 9.对推荐框的坐标进行规范化
        normalized_proposal_boxes = box_utils.normalized_coordinates(proposal_boxes, image_height, image_width)

        # 10.roi_pooling 层，获取推荐框中的特征
        roi_feature_maps = roi_pooling.pooling(feature_maps, normalized_proposal_boxes)

        # 11.抽取分类特征,并进行分类
        box_classifier_features = box_classifier_extrator.extract(roi_feature_maps)
        refined_box_encodings, classes_predictions = box_predict.predict(box_classifier_features, is_training)

        return {
            'rpn_box_encoding': rpn_anchor_encodes,
            'rpn_scores': rpn_scores,
            'anchors': rpn_anchors,
            'image_shape': (image_height, image_width),
            'proposal_boxes': proposal_boxes,
            'proposal_boxes_normalized': normalized_proposal_boxes,
            'refined_box_encodings': refined_box_encodings,
            'obeject_classes': classes_predictions
        }
