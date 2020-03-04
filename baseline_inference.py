import os
import argparse
import config as cfg
from torch.utils.data import DataLoader
from models.models import YOLOv3
from yolov3.utils.utils import load_classes
from yolov3.utils.datasets import ImageFolder
import datetime 

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--label_folder', required=True)
    parser.add_argument('--time_file', required=True)
    args = parser.parse_args()
    cfg.merge_config(args)
#    cfg.show_config(args)

    model = YOLOv3(args.config_path, args.weight_path)

    image_folder = args.image_folder
    test_image_path = os.listdir(image_folder)
    
    # The label folder
    label_folder = args.label_folder

    # The time file
    time_file = args.time_file

    classes = load_classes(args.class_path)

        
    start_time = datetime.datetime.now()
    for image_path in tqdm(test_image_path):
        outputs = model.predict(os.path.join(image_folder, image_path))
        
        prefix = image_path.split('.')[0]
        label_path = prefix + '.txt'
        
        with open(os.path.join(label_folder, label_path), 'w') as f:
            lines = []
            for out in outputs:
                if out is not None:
                    for o in out:
                        local_o = o
                        local_o = [str(float(x)) for x in local_o]
                        one_line = ','.join(local_o)
                        lines.append(one_line)
                    string = '\n'.join(lines) 
                    f.write(string)
    
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
    
