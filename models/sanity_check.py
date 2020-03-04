import os
import argparse


def is_image(filename):
    possible_ext = {'jpg', 'jpeg'}
    ext = os.path.splitext(filename)[1][1:].strip().lower()

    return (ext in possible_ext)


def get_index(image_path):
    image_path = os.path.basename(image_path)
    idx = os.path.splitext(image_path)[0]
    idx = int(idx)
    return idx


def check_image_continuous(image_dir):
    image_path_list = os.listdir(image_dir)
    image_path_list = list(filter(lambda x: is_image(x), image_path_list))
    image_path_list.sort(key=lambda x: (len(x), x))

    num_images = len(image_path_list)
    miss_idx_list = []
    num_miss = 0

    first_idx = get_index(image_path_list[0])
    last_idx = get_index(image_path_list[-1])
    num_miss = (last_idx-first_idx+1) - num_images
    if num_miss:
        image_path_set_gt = set(range(first_idx, last_idx+1))
        image_path_set = set(map(lambda x: get_index(x), image_path_list))
        miss_idx_list = list(image_path_set_gt - image_path_set)

    print('Frames missing list: {}'.format(miss_idx_list))
    print('Total {} frames missing'.format(num_miss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='Directory to images', required=True)
    args = parser.parse_args()

    check_image_continuous(args.image_dir)
