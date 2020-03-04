
import argparse
import cv2
import os 


def read_bboxes(args):
    
    bboxes = []
    with open(args.bboxes_path, 'r') as f:
        lines = f.readlines()
        for line in lines: 
            values = line.strip().split(',')
            box = [float(value) for value in values]
            bboxes.append(box)
            
    return bboxes


def plot_bboxes(image, bboxes, args):
    
    
    for box in bboxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    filename = os.path.basename(args.image_path)
    cv2.imwrite(filename, image)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True) 
    parser.add_argument('--bboxes_path', required=True)

    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    
    bboxes = read_bboxes(args)
    
    plot_bboxes(image, bboxes, args)
    
