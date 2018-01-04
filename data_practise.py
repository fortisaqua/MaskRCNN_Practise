from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import glob
import libs.datasets.dataset_factory as datasets
import libs.configs.config_v1 as cfg
import libs.nets.resnet_v1 as resnet_v1
import libs.datasets.coco as coco
import libs.preprocessings.coco_v1 as preprocess_coco
import numpy as np

FLAGS = tf.app.flags.FLAGS

_FILE_PATTERN = 'coco_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train2014': 82783, 'val2014': 40504}

_NUM_CLASSES = 81

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'An annotation image of varying size. (pixel-level masks)',
    'gt_masks': 'masks of instances in this image. (instance-level masks), of shape (N, image_height, image_width)',
    'gt_boxes': 'bounding boxes and classes of instances in this image, of shape (N, 5), each entry is (x1, y1, x2, y2)',
}

resnet50 = resnet_v1.resnet_v1_50

def get_records(dataset_name, split_name, dataset_dir,
        im_batch=1, is_training=False, file_pattern=None, reader=None):
    """"""
    if file_pattern is None:
        file_pattern = dataset_name + '_' + split_name + '*.tfrecord'

    pattern = '/home/wanghx/deepleraning/MaskRCNN_Practise/' +dataset_dir + 'records/'+file_pattern

    tfrecords = glob.glob(pattern)
    return tfrecords

with tf.Graph().as_default():

    records = get_records(FLAGS.dataset_name,
                             FLAGS.dataset_split_name,
                             FLAGS.dataset_dir,
                             FLAGS.im_batch,
                             is_training=False)

    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        coco.read(records)

    image, gt_boxes, gt_masks = \
        preprocess_coco.preprocess_image(image, gt_boxes, gt_masks)

    # using queue to input
    queue = tf.RandomShuffleQueue(capacity=12,min_after_dequeue=6,
                                  dtypes=(image.dtype, ih.dtype, iw.dtype,
                                            gt_boxes.dtype, gt_masks.dtype,
                                            num_instances.dtype, img_id.dtype))
    enqueue_op = queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    (image_, ih_, iw_, gt_boxes_, gt_masks_, num_instances_, img_id_) = queue.dequeue()
    qr = tf.train.QueueRunner(queue,[enqueue_op]*4)


    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # init_op = tf.initialize_all_variables()

    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess,coord=coord,start=True)

    boxes = [[100, 100, 200, 200],
             [50, 50, 100, 100],
             [100, 100, 750, 750],
             [50, 50, 60, 60]]
    # boxes = np.zeros((0, 4))
    boxes = tf.constant(boxes, tf.float32)
    # feat = ROIAlign(image, boxes, False, 16, 7, 7)
    sess.run(init_op)

    tf.train.start_queue_runners(sess=sess)
    # with sess.as_default():
    try:
        for i in range(20000):
            # image_np, ih_np, iw_np, gt_boxes_np, gt_masks_np, num_instances_np, img_id_np, \
            # feat_np = \
            #     sess.run([image, ih, iw, gt_boxes, gt_masks, num_instances, img_id,
            #         feat])
            image_np, ih_np, iw_np, gt_boxes_np, gt_masks_np, num_instances_np, img_id_np = \
                sess.run([image_, ih_, iw_, gt_boxes_, gt_masks_, num_instances_, img_id_])
            # print (image_np.shape, gt_boxes_np.shape, gt_masks_np.shape)

            if i % 100 == 0:
                print('%d, image_id: %s, instances: %d' % (i, str(img_id_np), num_instances_np))
                image_np = 256 * (image_np * 0.5 + 0.5)
                image_np = image_np.astype(np.uint8)
                image_np = np.squeeze(image_np)
                print(image_np.shape, ih_np, iw_np)
                # print (feat_np.shape)
                # im = Image.fromarray(image_np)
                # imd = ImageDraw.Draw(im)
                # for i in range(gt_boxes_np.shape[0]):
                #     imd.rectangle(gt_boxes_np[i, :])
                # im.save(str(img_id_np) + '.png')
                print (gt_boxes_np)
    except Exception,e:
        coord.request_stop(e)
    coord.request_stop()
    coord.join(enqueue_threads)
sess.close()