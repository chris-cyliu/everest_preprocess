import os
import cv2
import argparse
import random

from sanity_check import is_image


def save_dataset(save_path, image_dir, path_list):
    with open(save_path, 'w') as f:
        for path in path_list:
            f.write('{}\n'.format(os.path.join(image_dir, path)))


def split_dataset(video_path, image_dir, save_dir):
    image_path_list = os.listdir(image_dir)
    image_path_list = list(filter(lambda x: is_image(x), image_path_list))
    image_path_list.sort(key=lambda x: (len(x), x))

    num_images = len(image_path_list)

    caps = cv2.VideoCapture(video_path)
    fps = round(caps.get(cv2.CAP_PROP_FPS))
    caps.release()

    hour_frames = fps * 3600
    day_frames = hour_frames * 24
    hours = num_images / hour_frames
    days = num_images / day_frames

    split_type = None

    if days >= 3:
        split_type = 0
        # train 1 day, val 1 day, test 1 day
        train = image_path_list[:day_frames]
        val = image_path_list[day_frames:day_frames*2]
        test = image_path_list[day_frames*2:day_frames*3]

        # train = random.sample(train, 5000)
        # val = random.sample(val, 2000)
    elif days >= 2:
        split_type = 1
        # train 1 day, val left hour, test 1 day
        train = image_path_list[:day_frames]
        val = image_path_list[day_frames:num_images-day_frames]
        test = image_path_list[num_images-day_frames:]

        # train = random.sample(train, 5000)
        # val = random.sample(val, 2000)
    elif hours >= 3:
        split_type = 2
        # train 1 hour, val 1 hour, test left day
        train = image_path_list[:hour_frames]
        val = image_path_list[hour_frames:hour_frames*2]
        test = image_path_list[hour_frames*2:]

        # train = random.sample(train, 5000)
        # val = random.sample(val, 2000)
    else:
        split_type = 3
        # 1/3 for train, val and test
        num_split = hours / 3
        train = image_path_list[:num_split]
        val = image_path_list[num_split:num_split*2]
        test = image_path_list[num_split*2:]

        # train = train[::fps]
        # val = val[::fps]

    # train.sort(key=lambda x: (len(x), x))
    # val.sort(key=lambda x: (len(x), x))
    # test.sort(key=lambda x: (len(x), x))

    print('Split type: {}'.format(split_type))
    print('Train: {}'.format(len(train)))
    print('Val: {}'.format(len(val)))
    print('Test: {}'.format(len(test)))

    save_dataset(os.path.join(save_dir, 'train.txt'), image_dir, train)
    save_dataset(os.path.join(save_dir, 'val.txt'), image_dir, val)
    save_dataset(os.path.join(save_dir, 'test.txt'), image_dir, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='Path to video', required=True)
    parser.add_argument('--image_dir', help='Directory to images', required=True)
    parser.add_argument('--save_dir', help='Directory to save dataset path', required=True)
    args = parser.parse_args()

    split_dataset(args.video_path, args.image_dir, args.save_dir)
