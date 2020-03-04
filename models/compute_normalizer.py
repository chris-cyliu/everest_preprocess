import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from models.train import ObjectCountDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean and std of dataset')
    parser.add_argument('--train_data_path', help='path to train data')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    train_data = ObjectCountDataset(
        txt_file=args.train_data_path,
        transform=transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    mean = 0
    std = 0
    count = 0
    for data, label in tqdm(train_loader):
        batch_size = data.shape[0]
        channels = data.shape[1]
        data = data.view(batch_size, channels, -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        count += batch_size

    mean /= count
    std /= count

    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))
