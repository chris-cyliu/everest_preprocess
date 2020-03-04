import os
import argparse
import config as cfg
from torch.utils.data import DataLoader
from models.models import YOLOv3
from yolov3.utils.utils import load_classes
from yolov3.utils.datasets import ImageFolder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_table', action='store_true')
    args = parser.parse_args()
    cfg.merge_config(args)
    cfg.show_config(args)

    model = YOLOv3(args.config_path, args.weight_path)

    image_folder = 'yolov3/data/samples'
    test_image_path = os.listdir(image_folder)

    classes = load_classes(args.class_path)

    for image_path in test_image_path:
        output = model.predict(os.path.join(image_folder, image_path))
        for out in output:
            for o in out:
                cls = int(o[-1])
                prob = o[-2].item()
                print(classes[cls], prob)

"""
    dataloader = DataLoader(
        ImageFolder(image_folder),
        batch_size=9,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        for path, image in dataloader:
            output = model.forward(image)
            for out in output:
                for o in out:
                    cls = int(o[-1].item())
                    prob = o[-2].item()
                    print(classes[cls], prob)
"""
