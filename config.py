# dataset
video_name = 'taipei-bus'
class_name = 'car'
train_data_path = 'data/{}/train.txt'.format(video_name)
val_data_path = 'data/{}/val.txt'.format(video_name)
test_data_path = 'data/{}/test.txt'.format(video_name)

# yolov3
config_path = 'yolov3/config/yolov3.cfg'
weight_path = 'yolov3/weights/yolov3.weights'
class_path = 'yolov3/data/coco.names'

# mdn
lr = 0.00001
epoch = 20
batch_size = 64
seed = 1
mean = [0.4366, 0.4232, 0.4504]
std = [0.1763, 0.1690, 0.1709]
checkpoint_prefix = 'mdn_model'
checkpoint_dir = 'data/{}/checkpoint'.format(video_name)
checkpoint_path = '{}/{}_epoch{}.pth'.format(checkpoint_dir, checkpoint_prefix, epoch)
train_log_path = 'data/{}/logs/{}.log'.format(video_name, checkpoint_prefix)

# difference detector
threshold = 0

# query
k = 100
confidence = 0.9
gap = 0
num_threads = 10
table_path = 'data/{}/cache/cptable_mdn_1k.json'.format(video_name)
label_path = 'data/{}/cache/labels_1k.json'.format(video_name)
log_path = 'data/{}/logs/k{}_conf{}_gap{}_mdn_1k.log'.format(video_name, k, confidence, gap)


# ----------end of config---------- #
import sys
sys.path.append('yolov3')

__config_key__ = dir()


def merge_config(args):
    args_dict = vars(args)
    for key in __config_key__:
        if key.startswith('_'):
            continue
        if key not in args_dict or args_dict[key] is None:
            args_dict[key] = eval(key)


def show_config(args, keys=['k', 'confidence', 'gap']):
    args_dict = vars(args)
    for key in keys:
        print('{}: {}'.format(key, args_dict[key]))


def set_logger(log_path):
    import os
    import sys
    import logging
    FORMAT = '%(asctime)-15s %(message)s'
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=log_path, mode='w'),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
