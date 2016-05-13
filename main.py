import numpy as np
import matplotlib.pyplot as plt
import sys, os
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import cv2

from logging import FileHandler
import logging




sys.path.insert(0, './caffe-fast-rcnn/python/')
import caffe

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def main():

    prototxt = './caffe_models/faster_rcnn_test.pt'
    caffemodel = './caffe_models/VGG16_faster_rcnn_final.caffemodel'

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    im = cv2.imread('./000456.jpg')

    boxes, scores = im_detect(net, im)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


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
        im_scale = float(MAX_SIZE)/float(im_size_max)

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    im_shape = im.shape
    ims = [im]
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    blob = np.zeros((1, max_shape[0], max_shape[1], 3), dtype=np.float32)
    blob[0, 0:im_shape[0], 0:im_shape[1], :] = im

    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    return blob, np.array([im_scale])

def im_detect(net, im):
    blobs = {'data': None, 'rois':None}
    blobs['data'], im_scale = _get_image_blobs(im)

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scale[0]]], dtype =np.float32)

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


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()



if __name__=='__main__':
    main()
