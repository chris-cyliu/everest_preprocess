import argparse
import json
import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread

from darknet.darknet import performDetect
# from models.darknet_utils import darknet_resize
from topk.topk import PQueue


def load_label(label_path):
    label = None
    with open(label_path, 'r') as f:
        label = json.load(f)
    return label


def save_label(label, label_path):
    with open(label_path, 'w') as f:
        json.dump(label, f, indent=4)


def load_image(path_queue, image_queue):
    while not path_queue.empty():
        idx, image_path = path_queue.get()
        image_path = image_path.strip()
        timestamp = int(image_path.split('/')[-1].split('.')[0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (416, 416):
            print('Resize to (416, 416)')
            image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
            # image = darknet_resize(image, (416, 416, 3))
        image_queue.put((timestamp, image, image_path))


def run_model(args):
    label = []
    start = time.time()

    with open(args.image_path_list, 'r') as f:
        image_path_list = f.readlines()
        num_images = len(image_path_list)

        path_queue = Queue()
        image_queue = Queue(maxsize=100)
        for idx, image_path in enumerate(image_path_list):
            path_queue.put((idx, image_path))
        workers = [Thread(target=load_image, args=(path_queue, image_queue)) for _ in range(args.num_threads)]
        for w in workers:
            w.start()

        for _ in tqdm(range(num_images)):
            timestamp, image, image_path = image_queue.get()
            detections = performDetect(
                imagePath=image,
                configPath=args.config_path,
                weightPath=args.weight_path,
                metaPath=args.meta_path,
                showImage=False
            )
            # filter non-target objects
            detections = [obj for obj in detections if obj[0] == args.class_name]
            count = len(detections)
            label.append({
                'timestamp': timestamp,
                'count': count,
                'image_path': image_path
            })

    end = time.time()
    elapsed = end - start

    return elapsed, label


def get_topk(label):
    start = time.time()

    num_images = len(label)
    topk_queue = PQueue(args.k)

    for i in tqdm(range(num_images)):
        topk_queue.push((label[i]['count'], label[i]['timestamp']))

    topk = sorted(topk_queue.data, reverse=True)
    topk_value = [x[0] for x in topk]
    topk_indices = [x[1] for x in topk]

    end = time.time()
    elapsed = end - start

    return elapsed, topk_value, topk_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config of test image
    parser.add_argument('--image_path_list', required=True, help='Path to image list')
    # config of full model
    parser.add_argument('--config_path', required=True, help='Path to model config')
    parser.add_argument('--weight_path', required=True, help='Path to model weight')
    parser.add_argument('--meta_path', required=True, help='Path to model meta')
    # config of query
    parser.add_argument('--class_name', required=True, help='Class name for query')
    parser.add_argument('--label_path', required=True, help='Path to save and load label')
    parser.add_argument('--k', type=int, required=True, help='K of top-k')
    parser.add_argument('--num_threads', type=int, required=True, help='Number of threads')
    parser.add_argument('--read_label', action='store_true', help='Read label from file')

    args = parser.parse_args()

    os.chdir('./darknet')

    elapsed = []
    if args.read_label:
        label = load_label(args.label_path)
    else:
        t1, label = run_model(args)
        save_label(label, args.label_path)
        elapsed.append(t1)
    t2, topk_value, topk_indices = get_topk(label)
    elapsed.append(t2)

    print('Time: {}, {}'.format(elapsed, np.sum(elapsed)))
    print('Top-{} value: {}'.format(args.k, topk_value))
    print('Top-{} indices: {}'.format(args.k, topk_indices))
