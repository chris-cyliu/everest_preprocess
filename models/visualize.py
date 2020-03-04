import argparse
import cv2
import os
import pickle
import numpy as np
from darknet.darknet import performDetect


def calibrate(calibrator_path, detections):
    with open(calibrator_path, 'rb') as f:
        ir = pickle.load(f)

    for i in range(len(detections)):
        label, prob, bbox = detections[i]
        detections[i] = (label, ir.predict([prob])[0], bbox)

    return detections


def call_model(image, args):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = image.shape[1]
    image_height = image.shape[0]
    width_ratio = image_width / args.resize_width
    height_ratio = image_height / args.resize_height

    if image.shape[:2] != (args.resize_height, args.resize_width):
        image = cv2.resize(
            image,
            (args.resize_width, args.resize_height),
            interpolation=cv2.INTER_LINEAR
        )
    detections = performDetect(
        imagePath=image,
        configPath=args.config_path,
        weightPath=args.weight_path,
        metaPath=args.meta_path,
        thresh=args.threshold,
        showImage=False
    )

    for i in range(len(detections)):
        label, prob, bbox = detections[i]
        bbox = (
            bbox[0] * width_ratio,
            bbox[1] * height_ratio,
            bbox[2] * width_ratio,
            bbox[3] * height_ratio
        )
        detections[i] = (label, prob, bbox)

    return detections


def plot_bbox(image, detections, args):
    if args.class_name:
        detections = filter(lambda x: x[0] == args.class_name, detections)
        detections = list(detections)

    print('{} objects found.'.format(len(detections)))
    for det in detections:
        label = det[0]
        confidence = det[1]
        pstring = label + ': ' + str(np.rint(100 * confidence)) + '%'
        print(pstring)

    font_scale = 0.3
    font = cv2.FONT_HERSHEY_SIMPLEX
    prob_sum = 0
    for det in detections:
        prob = det[1]
        bbox = det[2]
        x1 = round(bbox[0] - bbox[2] / 2)
        x2 = round(bbox[0] + bbox[2] / 2)
        y1 = round(bbox[1] - bbox[3] / 2)
        y2 = round(bbox[1] + bbox[3] / 2)
        text = str(round(prob, 3))
        text_width, text_height = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        cv2.rectangle(image, (x1, y1-6), (x1+text_width+2, y1-6-text_height-2), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            text,
            (x1, y1-6),
            font,
            font_scale,
            (0, 0, 255)
        )
        prob_sum += prob

    filename = os.path.basename(args.image_path)
    filename = os.path.splitext(filename)[0]
    filename = '{}_{}.jpg'.format(filename, args.class_name)
    print('Expect: {}'.format(prob_sum))
    print('Save to {}'.format(filename))
    cv2.imwrite(filename, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--weight_path', required=True)
    parser.add_argument('--meta_path', required=True)
    parser.add_argument('--calibrator_path', required=True)
    parser.add_argument('--class_name', required=True)
    parser.add_argument('--resize_width', type=int, required=True)
    parser.add_argument('--resize_height', type=int, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--calibrate', action='store_true')

    args = parser.parse_args()

    wd = os.getcwd()
    os.chdir('./darknet')

    image = cv2.imread(args.image_path)
    detections = call_model(image, args)

    if args.calibrate:
        calibrate(args.calibrator_path, detections)

    os.chdir(wd)

    plot_bbox(image, detections, args)
