import multiprocessing
import os
import argparse
import time

import torch
import torchvision

import config as cfg
from torch.utils.data import DataLoader
from models.models import YOLOv3
from yolov3.utils.utils import load_classes
from yolov3.utils.datasets import ImageFolder
import datetime

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import lmdb
from tqdm import tqdm
import logging

batch_size = 5
sequence_length = 10
initial_prefetch_size = 1
resize_x = 416
resize_y = 416
buffer_size_image = 5

num_inference_process = 4


# Dali VideoPipe as reader
class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id,
                                        seed=0)
        self.input = ops.VideoReader(device="gpu", filenames=data,
                                     sequence_length=sequence_length,
                                     shard_id=0, num_shards=1,
                                     initial_fill=initial_prefetch_size,
                                     skip_vfr_check=True, dtype=types.FLOAT)
    def define_graph(self):
        output = self.input(name="Reader")
        return output


class WrapLMDB():

    def __init__(self, lmdb_path):
        self.lmdb_ctx = lmdb.open(lmdb_path, map_size=1099511627776)

    def write_lmdb(self, frame_id, label):
        with self.lmdb_ctx.begin(write=True) as txn:
            txn.put(str(frame_id).encode(), label.encode())


class ImageQueue():
    def __init__(self):
        self.queue = multiprocessing.Queue(buffer_size_image)

    def add_image(self, frame_idx, tensor):

        self.queue.put((frame_idx, tensor))

    def pop_image(self):
        return self.queue.get()

    def finish(self):
        self.queue.close()
        self.queue.join_thread()


def model_output_to_str(outputs):
    ret = ""
    lines = []
    for out in outputs:
        if out is not None:
            for o in out:
                local_o = o
                local_o = [str(float(x)) for x in local_o]
                one_line = ','.join(local_o)
                lines.append(one_line)
            ret = '\n'.join(lines)
    return ret

def process_batch_images(dali_iter, image_queue):
    frame_counter = 0
    with tqdm(total=target_num_frame) as pbar:
        for i, data in enumerate(dali_iter):
            for batch_idx in range(batch_size):
                for seq_idx in range(sequence_length):
                    frame_counter += 1
                    pbar.update(1)
                    image_queue.add_image(frame_counter, data[0]['frame'][batch_idx][seq_idx])

                    if frame_counter >= target_num_frame and target_num_frame != 0:
                        return

def run_video_reader(image_queue, video_file, target_num_frame):
    logging.debug("Spawn process, video reader")
    videoPipe = VideoPipe(batch_size=batch_size, num_threads=1, device_id=0,
                          data=video_file)
    videoPipe.build()

    dali_iter = DALIGenericIterator([videoPipe], ['frame'],
                                    videoPipe.epoch_size("Reader"),
                                    fill_last_batch=False)

    process_batch_images(dali_iter, image_queue)
    # special object to end inference processes
    for x in range(num_inference_process):
        image_queue.add_image(-1, None)
    return


def run_inference_yolo(image_queue, config_path, weight_path, lmdb_path):
    logging.debug("Spawn process, inference yolo")
    model = YOLOv3(config_path, weight_path)
    wrap_lmdb = WrapLMDB(lmdb_path)

    while(True):
        frame_id, frame_tensor = image_queue.pop_image()
        if frame_id == -1:
            break

        # resize here
        frame_tensor = torch.unsqueeze(frame_tensor, 0)
        frame_tensor = frame_tensor.permute(0,3,1,2)
        frame_tensor = torch.nn.functional.interpolate(frame_tensor,
                                                       size=(resize_y,resize_x),
                                                       mode='bilinear')
        label_raw_output = model.predict_tensor(frame_tensor)
        str_label_output = model_output_to_str(label_raw_output)
        wrap_lmdb.write_lmdb(frame_id, str_label_output)
        logging.debug("Inference image id: %d".format(frame_id))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', required=True)
    parser.add_argument('--label_file', required=True)
    parser.add_argument('--time_file', required=True)
    parser.add_argument('--target_num_frame', required=True, type=int)
    args = parser.parse_args()
    cfg.merge_config(args)
    #    cfg.show_config(args)


    video_file = args.video_file
    lmdb_file = args.label_file
    target_num_frame = args.target_num_frame
    time_file = args.time_file     # The time file

    start_time = datetime.datetime.now()
    image_queue = ImageQueue()

    process_video_loader = multiprocessing.Process(target=run_video_reader,
                                                   args=(image_queue,
                                                         video_file,
                                                        target_num_frame))
    process_video_loader.start()
    process_inference = []
    for x in range(num_inference_process):
        p = multiprocessing.Process(target=run_inference_yolo, args=(image_queue,
                                                                     args.config_path,
                                                                     args.weight_path,
                                                                     lmdb_file))
        p.start()
        process_inference.append(p)

    process_video_loader.join()
    for p in process_inference:
        p.join()

    image_queue.finish()

    end_time = datetime.datetime.now()

    delta = (end_time - start_time)
    days = delta.days
    seconds = delta.seconds
    microseconds = delta.microseconds

    time_f = open(time_file, 'w')
    time_f.write('days:' + str(days) + '\n')
    time_f.write('seconds:' + str(seconds) + '\n')
    time_f.write('microseconds:' + str(microseconds) + '\n')
    time_f.close()
    
