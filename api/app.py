from __future__ import print_function
#api/app.py
import os
import requests
from flask import Flask, current_app, Response, json, jsonify, request
from flask_cors import CORS
import pandas as pd
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import cv2
from PIL import Image
import imageio

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from LIP_model import *

N_CLASSES = 20
INPUT_SIZE = (384, 384)
DATA_DIRECTORY = './datasets/examples'
DATA_LIST_PATH = './datasets/examples/list/val.txt'
RESTORE_FROM = './checkpoint/JPPNet-s2'
OUTPUT_DIR = './output/dataset'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    os.makedirs('{}/images'.format(OUTPUT_DIR))
    os.makedirs('{}/labels'.format(OUTPUT_DIR))

def create_app():
  """
  Create app
  """
  app = Flask(__name__)

  CORS(app, supports_credentials=True)

  def custom_response(res, status_code):
    """
    Custom Response Function
    """
    return Response(
      mimetype="application/json",
      response=json.dumps(res),
      status=status_code
    )

  def convert_mask_lip(mask):
    LIP_to_FP_dict = {
      0: 0,
      1: 1,
      2: 2,
      3: 0,
      4: 3,
      5: 4,
      6: 7,
      7: 4,
      8: 0,
      9: 6,
      10: 7,
      11: 17,
      12: 5,
      13: 11,
      14: 14,
      15: 15,
      16: 12,
      17: 13,
      18: 9,
      19: 10
    }

    LIP_rgb_to_code_dict = {
      '0_0_0': 0,
      '128_0_0': 1,
      '255_0_0': 2,
      '0_85_0': 3,
      '170_0_51': 4,
      '255_85_0': 5,
      '0_0_85': 6,
      '0_119_221': 7,
      '85_85_0': 8,
      '0_85_85': 9,
      '85_51_0': 10,
      '52_86_128': 11,
      '0_128_0': 12,
      '0_0_255': 13,
      '51_170_221': 14,
      '0_255_255': 15,
      '85_255_170': 16,
      '170_255_85': 17,
      '255_255_0': 18,
      '255_170_0': 19
    }
    image_bounds_dict = {}
    new_matrix = []
    for i, row in enumerate(mask):
      new_row = []
      for j, elem in enumerate(row):
        new_col = []
        color_str = str(elem[0]) + '_' + str(elem[1]) + '_' + str(elem[2])

        LIP_code = LIP_rgb_to_code_dict[color_str]
        FP_code = LIP_to_FP_dict[LIP_code]
        FP_code = [FP_code]*3
        new_row.append(FP_code)
      new_matrix.append(new_row) 
    new_matrix = np.array(new_matrix).astype(np.uint8)
    return new_matrix
        

  def getBoundingBoxes(mask):
    image_bounds_dict = {}
    for i, row in enumerate(mask[0]):
      for j, elem in enumerate(row):
        color_str = str(elem[0]) + '_' + str(elem[1]) + '_' + str(elem[2])

        if color_str not in image_bounds_dict:
          image_bounds_dict[color_str] = {
              'left': j, 'top': i, 'right': j, 'bottom': i}
        else:
          previous_left = image_bounds_dict[color_str]['left']
          previous_right = image_bounds_dict[color_str]['right']
          previous_top = image_bounds_dict[color_str]['top']
          previous_bottom = image_bounds_dict[color_str]['bottom']

          image_bounds_dict[color_str]['left'] = min(j, previous_left)
          image_bounds_dict[color_str]['top'] = min(i, previous_top)
          image_bounds_dict[color_str]['right'] = max(j, previous_right)
          image_bounds_dict[color_str]['bottom'] = max(i, previous_bottom)

    data = []
    for key, item in image_bounds_dict.items():
      data.append({
        'id': key,
        'bounds': item
      })
    return data

  @app.route('/', methods=['GET'])
  def index():
    return 'alive'
  
  
  @app.route('/getSegmentation', methods=['POST'])
  def get_segmentation():
    if 'file' not in request.files:
      return custom_response({ 'error': 'No file provided' }, 400)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
      return custom_response({ 'error': 'File without name forbidden' }, 400)
   
    img_contents = file.read()

    with open('{}/images/{}.jpg'.format(OUTPUT_DIR, file.filename.split('.')[0]), "wb") as f:
      f.write(img_contents)

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False, coord)
        image = reader.read_images_from_binary(img_contents)
        image_rev = tf.reverse(image, tf.stack([1]))

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)

    
    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']

    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
        parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
        parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
        parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')


    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.
    
    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


    # Iterate over training steps.
    parsing_ = sess.run(pred_all)
    img_id = file.filename

    msk = decode_labels(parsing_, num_classes=N_CLASSES)
    parsing_im = convert_mask_lip(msk[0])
    imageio.imwrite('{}/labels/{}.png'.format(OUTPUT_DIR, img_id.split('.')[0]), parsing_im)

    coord.request_stop()

    bbox = getBoundingBoxes(msk)

    return custom_response(bbox, 200)

  

  return app