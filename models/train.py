import argparse
import torch
import os
import numpy as np
import config as cfg
from PIL import Image
from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import show_config, set_logger
from models.models import ResNetMDN
from models import mdn


class ObjectCountDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        with open(txt_file, 'r') as f:
            image_path_list = f.readlines()
            self.image_path_list = list(map(lambda x: x.strip(), image_path_list))
            # https://github.com/pytorch/pytorch/issues/13246
            self.image_path_list = np.array(self.image_path_list)
        self.transform = transform
        self.num_images = len(self.image_path_list)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # TODO: support tensor
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        image_path = self.image_path_list[idx]
        label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        density_path = image_path.replace('images', 'density').replace('jpg', 'npy')
        image = Image.open(image_path).convert('RGB')
        with open(label_path, 'r') as f:
            label = len(f.readlines())
        density = np.load(density_path)
        density = np.expand_dims(density, axis=0)

        if self.transform:
            image = self.transform(image)
        label = torch.Tensor([label])
        density = torch.from_numpy(density)

        return image, label, density


def get_weight(label, weight_dict):
    weight = torch.Tensor([weight_dict[lab.item()] for lab in label])
    return weight


def parse_args():
    parser = argparse.ArgumentParser(description='Mixed Density Network training')
    parser.add_argument('--test_only', action='store_true', help='run test only')

    args = parser.parse_args()
    cfg.merge_config(args)
    show_config(args, ['lr', 'epoch', 'batch_size', 'checkpoint_path'])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    args_dict = vars(args)
    args_dict['use_cuda'] = use_cuda
    args_dict['device'] = device

    return args


if __name__ == '__main__':
    args = parse_args()
    set_logger(args.train_log_path)

    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    train_data = ObjectCountDataset(
        txt_file=args.train_data_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
    )
    test_data = ObjectCountDataset(
        txt_file=args.val_data_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_threads,
        pin_memory=True
    )

    # compute weight for samples
    # count_dict = dict()
    # for data, label in train_loader:
    #     label = label.view(-1)
    #     for lab in label:
    #         lab = lab.item()
    #         if lab not in count_dict:
    #             count_dict[lab] = 1
    #         else:
    #             count_dict[lab] += 1
    # keys = [int(k) for k in count_dict.keys()]
    # keys.sort()
    # for k in keys:
    #     print('{}: {}'.format(k, count_dict[k]))
    # weight_dict = dict([(k, 1/v) for k, v in count_dict.items()])
    # weight_sum = sum(weight_dict.values())
    # weight_dict = dict([(k, 100*v/weight_sum) for k, v in weight_dict.items()])

    # model
    model = ResNetMDN().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    checkpoint_path = args.checkpoint_path
    optim_path = os.path.join(checkpoint_path, '{}_optimizer.pth'.format(args.checkpoint_prefix))
    loss_path = os.path.join(checkpoint_path, '{}_train_losses.pth'.format(args.checkpoint_prefix))

    if args.test_only:
        model_path = os.path.join(checkpoint_path, '{}_model_epoch{}.pth'.format(args.checkpoint_prefix, args.epoch))
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optim_path))

    # train
    if not args.test_only:
        model.train()
        train_losses = []

        train_size = len(train_loader)
        for epoch in range(1, args.epoch+1):
            for i, (data, label, density) in enumerate(train_loader):
                data = data.to(device=args.device, non_blocking=True)
                label = label.to(device=args.device, non_blocking=True)
                density = density.to(device=args.device, non_blocking=True)
                # weight = get_weight(label, weight_dict).to(device=args.device, non_blocking=True)
                optimizer.zero_grad()
                out, (pi, sigma, mu) = model(data)
                mdn_loss = mdn.mdn_loss(pi, sigma, mu, label)
                mse_loss = model.mse_loss(out, density * 255 * 100)
                loss = mdn_loss + mse_loss
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                if i % 10 == 0:
                    mu = mu.squeeze(-1)
                    label = label.squeeze(-1)
                    mean = (mu * pi).sum(1)
                    acc = (torch.round(mean) == label).sum() / float(len(pi))
                    print('Epoch: [{}/{}] Iter: [{}/{}] MDN-Loss: {:.2f} MSE-Loss: {:.2f} Loss: {:.2f} Acc: {:.2f}'.format(epoch, args.epoch, i, train_size, mdn_loss.item(), mse_loss.item(), loss.item(), acc.item()))

            # scheduler.step()

            if epoch % 10 == 0 or epoch == args.epoch:
                model_path = os.path.join(checkpoint_path, '{}_model_epoch{}.pth'.format(args.checkpoint_prefix, epoch))
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
                torch.save(train_losses, loss_path)

    # test
    model.eval()
    tp = 0
    acc = 0

    mean_dict = dict()
    with torch.no_grad():
        for data, label, _ in tqdm(test_loader):
            data = data.to(device=args.device, non_blocking=True)
            label = label.to(device=args.device, non_blocking=True)
            # density = density.to(device=args.device, non_blocking=True)
            out, (pi, sigma, mu) = model(data)

            # from torch.distributions import normal
            # mean = mu[0][i].item()
            # var = sigma[0][i].item()
            # normals = [normal.Normal(loc=mean, scale=var) for i in range(pi.shape[0])]
            mu = mu.squeeze(-1)
            label = label.squeeze(-1)
            mean = (mu * pi).sum(1)
            tp += (torch.round(mean) == label).sum().item()
            acc += len(pi)

            mean = mean.view(-1)
            label = label.view(-1)
            for i in range(len(mean)):
                val = mean[i].item()
                lab = label[i].item()
                if lab not in mean_dict:
                    mean_dict[lab] = [val]
                else:
                    mean_dict[lab] += [val]

    acc = tp / acc
    print('Accuracy: {:.2f}'.format(acc))

    print('K N Mean MSE')
    se_dict = dict()
    count_dict = dict()
    keys = [int(k) for k in mean_dict.keys()]
    keys.sort()
    for k in keys:
        samples = len(mean_dict[k])
        se = ((np.array(mean_dict[k]) - k) ** 2).sum()
        mean = np.mean(mean_dict[k])
        se_dict[k] = se
        count_dict[k] = samples
        mean_dict[k] = np.mean(mean_dict[k])
        print('{}: {} {:.2f} {:.2f}'.format(k, samples, mean, se / samples))
    se = np.array(list(se_dict.values()))
    count = np.array(list(count_dict.values()))
    print('MSE: {:.2f}'.format(se.sum() / count.sum()))
