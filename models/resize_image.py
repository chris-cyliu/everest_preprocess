import argparse
import cv2
import os
from tqdm import tqdm
from queue import Queue
from threading import Thread

# from models.darknet_utils import darknet_resize


def replace_path(path):
    return path.replace('images', 'images_resize')


def resize_image(path_queue, image_queue, resize_width, resize_height):
    while not path_queue.empty():
        image_path = path_queue.get()
        image_path = image_path.strip()

        image = cv2.imread(image_path)
        image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        # image = darknet_resize(image, (resize_height, resize_width, 3))

        image_path = replace_path(image_path)

        image_queue.put((image, image_path))


def main(args):
    with open(args.image_path_list, 'r') as f:
        image_path_list = f.readlines()
        num_images = len(image_path_list)

        path_queue = Queue()
        image_queue = Queue(maxsize=100)
        for image_path in image_path_list:
            path_queue.put(image_path)
        workers = [Thread(target=resize_image, args=(path_queue, image_queue, args.resize_width, args.resize_height)) for _ in range(args.num_threads)]
        for w in workers:
            w.start()

        dirname = os.path.dirname(image_path_list[0])
        dirname = replace_path(dirname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        for _ in tqdm(range(num_images)):
            image, image_path = image_queue.get()

            cv2.imwrite(image_path, image)

    print('Rewrite {}'.format(args.image_path_list))
    with open(args.image_path_list, 'w') as f:
        image_path_list = map(lambda x: x.strip(), image_path_list)
        image_path_list = map(replace_path, image_path_list)
        for path in image_path_list:
            f.write('{}\n'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_list', required=True)
    parser.add_argument('--resize_width', type=int, required=True)
    parser.add_argument('--resize_height', type=int, required=True)
    parser.add_argument('--num_threads', type=int, required=True)
    args = parser.parse_args()

    main(args)
