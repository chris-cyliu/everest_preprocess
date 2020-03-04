import argparse
import cv2
import itertools
import os
import pickle
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread
from sklearn.isotonic import IsotonicRegression

from darknet.darknet import performDetect


def load_image(path_queue, image_queue, resize_height, resize_width):
    while not path_queue.empty():
        idx, image_path = path_queue.get()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (resize_height, resize_width):
            image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        image_queue.put((idx, image))


def decode_boxes(pred_boxes):
    # [[xc, yc, w, h], ...] -> [[x1, y1, x2, y2], ...]
    boxes = np.zeros_like(pred_boxes)
    boxes[..., 0] = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    boxes[..., 1] = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    boxes[..., 2] = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    boxes[..., 3] = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    return boxes


def get_prediction(image_path_list,
                   config_path,
                   weight_path,
                   meta_path,
                   resize_height,
                   resize_width,
                   num_threads):
    num_images = len(image_path_list)

    path_queue = Queue()
    image_queue = Queue(maxsize=100)
    for idx, image_path in enumerate(image_path_list):
        path_queue.put((idx, image_path))
    workers = [Thread(target=load_image, args=(path_queue, image_queue, resize_height, resize_width)) for _ in range(num_threads)]
    for w in workers:
        w.start()

    prob_list = [None] * num_images
    boxes_list = [None] * num_images

    for _ in tqdm(range(num_images)):
        idx, image = image_queue.get()
        detections = performDetect(
            imagePath=image,
            configPath=config_path,
            weightPath=weight_path,
            metaPath=meta_path,
            thresh=0.1,
            showImage=False
        )
        # sort highest score first in preparation of gt matching
        detections.sort(key=lambda x: -x[1])
        prob = map(lambda x: x[1], detections)
        boxes = map(lambda x: x[2], detections)
        prob = np.array(list(prob)).reshape(-1)
        boxes = np.array(list(boxes)).reshape(-1, 4)
        # normalize as the same format of gt
        boxes[..., 0::2] /= resize_width
        boxes[..., 1::2] /= resize_height

        boxes = decode_boxes(boxes)

        prob_list[idx] = prob
        boxes_list[idx] = boxes

    return prob_list, boxes_list


def get_gt(image_path_list):
    label_list = []
    boxes_list = []

    for image_path in tqdm(image_path_list):
        gt_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        with open(gt_path, 'r') as sf:
            gts = sf.readlines()
            gts = list(map(lambda x: x.strip().split(' '), gts))
            label = map(lambda x: int(x[0]), gts)
            boxes = map(lambda x: [float(x[1]), float(x[2]), float(x[3]), float(x[4])], gts)
            label = np.array(list(label)).reshape(-1)
            boxes = np.array(list(boxes)).reshape(-1, 4)

            boxes = decode_boxes(boxes)

            label_list.append(label)
            boxes_list.append(boxes)

    return label_list, boxes_list


def compute_iou(a, b):
    # [[x1, y1, x2, y2], ...]
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    area = max(inter_w, 0) * max(inter_h, 0)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou = area / (area_a + area_b - area)

    return iou


def match_boxes(pred_boxes, gt_boxes, gt_label, iou_thr):
    # gt_label is unused here, support multiclass in the future
    num_images = len(pred_boxes)
    assert num_images == len(gt_boxes)

    pred_label = []
    for i in range(num_images):
        ious = []
        image_pred = pred_boxes[i]
        image_gt = gt_boxes[i]

        # compute iou between prediciton and gt
        num_image_pred = len(image_pred)
        num_image_gt = len(image_gt)
        ious = np.zeros((num_image_pred, num_image_gt))
        for m in range(num_image_pred):
            for n in range(num_image_gt):
                ious[m, n] = compute_iou(image_pred[m], image_gt[n])

        # find matched gt
        dtm = np.zeros((num_image_pred,))
        gtm = np.zeros((num_image_gt,))
        for m in range(num_image_pred):
            iou = iou_thr
            gind = -1
            for n in range(num_image_gt):
                # skip if gt already matched
                if gtm[n] > 0:
                    continue

                # continue to next gt unless better match made
                if ious[m, n] < iou:
                    continue

                # store if match successful and best so far
                iou = ious[m, n]
                gind = n

            # no matched gt
            if gind == -1:
                continue
            gtm[gind] = 1
            dtm[m] = 1
            # dtm[m] = gind

        pred_label += dtm.tolist()

    return pred_label


def calibrate(prob, label):
    # prob is a list of ndarray
    prob = list(itertools.chain(*prob))

    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    ir.fit(prob, label)

    return ir


def compute_error(ir, prob, label):
    prob = list(itertools.chain(*prob))
    prob_calibrate = ir.predict(prob)
    return np.mean(np.square(prob_calibrate - label))


def save_calibration_model(ir, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(ir, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_list', help='Path to image path', required=True)
    parser.add_argument('--save_path', help='Path to save calibrated result', required=True)
    # config of cheap model
    parser.add_argument('--cm_config_path', help='Path to cheap model config', required=True)
    parser.add_argument('--cm_weight_path', help='Path to cheap model weight', required=True)
    parser.add_argument('--cm_meta_path', help='Path to cheap model meta', required=True)
    parser.add_argument('--resize_width', type=int, help='Width to resizse', required=True)
    parser.add_argument('--resize_height', type=int, help='Height to resize', required=True)
    parser.add_argument('--iou_thr', type=float, help='IoU threshold for gt matching', required=True)
    parser.add_argument('--num_threads', type=int, help='Number of treads to launch', required=True)
    args = parser.parse_args()

    os.chdir('./darknet')

    with open(args.image_path_list, 'r') as f:
        image_path_list = f.readlines()
        image_path_list = list(map(lambda x: x.strip(), image_path_list))

        pred_prob, pred_boxes = get_prediction(
            image_path_list,
            args.cm_config_path,
            args.cm_weight_path,
            args.cm_meta_path,
            args.resize_height,
            args.resize_width,
            args.num_threads
        )
        print('Get prediction done')
        gt_label, gt_boxes = get_gt(image_path_list)
        print('Get gt done')

    pred_label = match_boxes(pred_boxes, gt_boxes, gt_label, args.iou_thr)

    ir = calibrate(pred_prob, pred_label)
    save_calibration_model(ir, args.save_path)
    #with open('vdata/taipei-bus/gt0.5/calibrator.pkl', 'rb') as f:
    #    ir = pickle.load(f)
    print('Save calibrator to {}'.format(args.save_path))
    print('MSE Error: {}'.format(compute_error(ir, pred_prob, pred_label)))
    print(ir.predict(np.linspace(0, 1, 11)))
