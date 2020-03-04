
import numpy as np
import os 
import argparse 

from tqdm import tqdm

from yolov3.utils.utils import load_classes
from shutil import copyfile


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--label_folder', required=True)
    parser.add_argument('--class_path', required=True)
    parser.add_argument('--clss', required=True)
    
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    labels_path = os.listdir(args.label_folder)
    
    # Load the classes
    classes = load_classes(args.class_path)
    
    clss = args.clss.strip().lower() 
    
    try:
        clss_ind = classes.index(clss)
    
    except ValueError:
        print('{} is not in predefined classes'.format(clss))
        exit(1)
    
    num_images = len(labels_path)
    object_nums = []
    
    for i in tqdm(range(len(labels_path))):
        label_path = str(i+1) + '.txt'
        
        data = open(os.path.join(args.label_folder, label_path)).read() 
        
        lines = data.split('\n')
        
        num_object = 0
        for line in lines:
            line = line.strip() 
            
            if line == '':
                continue 
            
            clss_num = line.split(',')[-1] 
            clss_num = int(float(clss_num))
            
            if clss_ind == clss_num:
                num_object += 1
        
        object_nums.append(num_object)
    
    object_nums = np.array(object_nums)

    top100_ind = (-object_nums).argsort()[:100]
    
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)


    for index in top100_ind:
        image_path = os.path.join(args.image_folder, str(index+1) + '.jpg')
        
        dst = os.path.join(args.output_folder, str(index+1) + '.jpg')
        
        try:
            copyfile(image_path, dst)
        except Exception:
            print('{} is not copied successfully'.format(image_path))
     
