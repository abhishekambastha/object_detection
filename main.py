import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys, os
from bbox_transform import clip_boxes, bbox_transform_inv
import cv2


def main():

    prototxt = './faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = './faster_rcnn_models/VGG16/VGG16_faster_rcnn_final.caffemodel'

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    im = cv2.imread('./000456.jpg')



def _get_image_blobs(im):
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_orig = im.astype(np.float32, copy=True)
    im_orig -= PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    #Resize Image
    target_size = 600
    MAX_SIZE = 1000

    im_scale = float(target_size)/float(im_size_min)
    if(np.round(im_size_max * im_scale) > MAX_SIZE):
        im_scale = float(target_size)/float(im_size_max)

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    blob = np.zeros((1, im_shape[0], im_shape[1], 3), dtype=np.float32)
    blob[0, ...] = im

    channel_swap = (0, 3, 1, 2)
    blob.transpose(channel_swap)

    return blob, im_scale

def im_detect(net, im):
    blobs = {'data': None, 'rois':None}
    blobs['data'], im_scale = _get_image_blobs(im)

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scale[0]]],
                                dtype =np.float32)

    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    rois = net.blobs['rois'].data.copy()

    boxes = rois[:, 1:5] / im_scale[0]
    scores = blobs_out['cls_prob']

    ##Bounding Box Regression
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes



