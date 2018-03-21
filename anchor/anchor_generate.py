# coding=utf-8
from __future__ import division
import tensorflow as tf


def anchor_generate(height, width, scales, aspect_ratios, base_anchor_size, anchor_stride):
    scales = tf.convert_to_tensor(scales, dtype=tf.float32)
    aspect_ratios = tf.convert_to_tensor(aspect_ratios, tf.float32)
    scales_grid, aspect_ratios_grid = tf.meshgrid(scales, aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = tile_anchors(height,
                           width,
                           scales_grid,
                           aspect_ratios_grid,
                           base_anchor_size,
                           anchor_stride)
    # 将anchor移到目标中心
    anchors = anchors + anchor_stride / 2
    return anchors


def tile_anchors(height, width, scales, aspect_ratios, base_anchor_size, anchor_stride):
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size
    widths = scales * ratio_sqrts * base_anchor_size

    # Get a grid of box centers
    y_centers = tf.to_float(tf.range(height))
    y_centers = y_centers * anchor_stride
    x_centers = tf.to_float(tf.range(width))
    x_centers = x_centers * anchor_stride
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return bbox_corners


def _center_size_bbox_to_corners_bbox(centers, sizes):
    return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
