import cv2
import os
import json
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread

from darknet.darknet import performDetect
# from models.darknet_utils import darknet_resize


def load_image(path_queue, image_queue, resize_height, resize_width):
    while not path_queue.empty():
        idx, image_path = path_queue.get()
        image_path = image_path.strip()
        timestamp = int(image_path.split('/')[-1].split('.')[0])
        # (height, width, channel)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (args.resize_height, args.resize_width):
            # ratio_height = image.shape[0] / args.resize_height
            # ratio_width = image.shape[1] / args.resize_width
            image = cv2.resize(image, (args.resize_width, args.resize_height), interpolation=cv2.INTER_LINEAR)
            # image = darknet_resize(image, (args.resize_height, args.resize_width, 3))
        image_queue.put((timestamp, image, image_path))


def call_model(image,
               config_path,
               weight_path,
               meta_path,
               class_name,
               ret_type,
               image_height,
               image_width):
    detections = performDetect(
        imagePath=image,
        configPath=config_path,
        weightPath=weight_path,
        metaPath=meta_path,
        thresh=0.5,
        showImage=False
    )
    # filter non-target objects
    detections = [obj for obj in detections if obj[0] == class_name]
    if ret_type == 'count':
        ret = len(detections)
    else:
        ret = map(lambda x: x[2], detections)
        ret = list(ret)
        ret = np.array(ret)
        # normalize as required format of darknet
        ret[..., 0::2] /= image_width
        ret[..., 1::2] /= image_height
    return ret


def generate_label(args):
    start = time.time()

    with open(args.image_path_list, 'r') as f:
        image_path_list = f.readlines()
        random.shuffle(image_path_list)
        num_images = len(image_path_list)

        path_queue = Queue()
        image_queue = Queue(maxsize=100)
        for idx, image_path in enumerate(image_path_list):
            path_queue.put((idx, image_path))
        workers = [Thread(target=load_image, args=(path_queue, image_queue, args.resize_height, args.resize_width)) for _ in range(args.num_threads)]
        for w in workers:
            w.daemon = True
            w.start()

        num_valid = 0
        valid_path_list = []
        ret_list = []

        for _ in tqdm(range(num_images)):
            timestamp, image, image_path = image_queue.get()
            ret = call_model(
                image,
                args.config_path,
                args.weight_path,
                args.meta_path,
                args.class_name,
                args.type,
                args.resize_height,
                args.resize_width
            )

            if args.type == 'count':
                ret_list.append([timestamp, ret, image_path])
            else:
                save_path_image = os.path.join(args.save_path, '{}.txt'.format(timestamp))

                num_ret = len(ret)
                if num_ret == 0:
                    continue
                with open(save_path_image, 'w') as sf:
                    for i in range(num_ret):
                        sf.write('{} {} {} {} {}\n'.format(0, ret[i, 0], ret[i, 1], ret[i, 2], ret[i, 3]))

                num_valid += 1
                valid_path_list.append(image_path)
                if num_valid == args.sample_size:
                    break

    if args.type == 'count':
        key = ['timestamp', 'count', 'image_path']
        label_list = map(lambda x: dict(zip(key, x)), ret_list)
        label_list = list(label_list)

        with open(args.save_path, 'w') as f:
            json.dump(label_list, f, indent=4)
    else:
        valid_path_list.sort(key=lambda x: (len(x), x))
        with open(args.image_path_list, 'w') as f:
            for path in valid_path_list:
                f.write('{}\n'.format(path))
        print('Rewrite {}'.format(args.image_path_list))
        print('Sample size: {}'.format(num_valid))

    end = time.time()
    elapsed = end - start
    print('Time: {}'.format(elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config of test image
    parser.add_argument('--image_path_list', required=True)
    # config of model
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--weight_path', required=True)
    parser.add_argument('--meta_path', required=True)
    # config of query
    parser.add_argument('--class_name', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--resize_height', type=int, required=True)
    parser.add_argument('--resize_width', type=int, required=True)
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--num_threads', type=int, required=True)
    parser.add_argument('--type', required=True, choices=['count', 'bbox'])

    args = parser.parse_args()

    os.chdir('./darknet')

    generate_label(args)
