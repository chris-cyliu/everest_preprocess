import argparse
import os
import pickle
import numpy as np

from inference.poibin import PoiBin
from models.calibration import get_prediction, get_gt


def calibrate(calibrator_path, prob):
    with open(calibrator_path, 'rb') as f:
        ir = pickle.load(f)

    prob = list(map(lambda x: ir.predict(x), prob))

    return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_list', help='Path to image path', required=True)
    parser.add_argument('--cm_config_path', help='Path to cheap model config', required=True)
    parser.add_argument('--cm_weight_path', help='Path to cheap model weight', required=True)
    parser.add_argument('--cm_meta_path', help='Path to cheap model meta', required=True)
    parser.add_argument('--calibrator_path', help='Path to calibrator', required=True)
    parser.add_argument('--resize_width', help='Width to resize', type=int, required=True)
    parser.add_argument('--resize_height', help='Height to resize', type=int, required=True)
    parser.add_argument('--num_threads', help='Number of threads to launch', type=int, required=True)
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

        gt_label, gt_boxes = get_gt(image_path_list)

    pred_prob = calibrate(args.calibrator_path, pred_prob)

    pred_count = list(map(lambda x: len(x), pred_prob))
    gt_count = list(map(lambda x: len(x), gt_label))

    pred_count_prob = list(map(lambda x: PoiBin(x).pmf(range(len(x)+1)), pred_prob))
    pred_count_num = list(map(lambda x: np.arange(x+1), pred_count))

    pred_mean = list(map(lambda x, y: np.dot(x, y), pred_count_prob, pred_count_num))

    num_frames = len(gt_count)
    max_count = np.max(gt_count)
    mean_list = [[] for _ in range(max_count + 1)]
    idx_list = [[] for _ in range(max_count + 1)]
    get_timestamp = lambda x: int(x.split('/')[-1].split('.')[0])
    for i in range(num_frames):
        mean_list[gt_count[i]].append(pred_mean[i])
        idx_list[gt_count[i]].append(get_timestamp(image_path_list[i]))

    mean_list = list(map(lambda x: np.mean(x) if len(x) > 0 else 0, mean_list))

    for i in range(1, len(mean_list)):
        print('{}: {}, {}'.format(i, round(mean_list[i], 2), idx_list[i][:10]))
