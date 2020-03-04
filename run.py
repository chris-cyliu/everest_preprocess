import argparse
import json
import logging
import pickle
import time
import torch
import math
import numpy as np
import config as cfg
from tqdm import tqdm
from torch.distributions import normal
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import show_config, set_logger
from inference.inference import build_uncertain_table_fast
from topk.topk import TopKOp, TopKGAPOp, GSList, VT
from models.models import difference_detector, ResNetMDN


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, help='K of topk-k')
    parser.add_argument('--confidence', type=float, help='Confidence for result')
    parser.add_argument('--gap', type=int, help='Gap for result')
    parser.add_argument('--threshold', type=float, help='Threshold for difference detector')
    parser.add_argument('--read_table', action='store_true', help='Read table from file')

    args = parser.parse_args()
    cfg.merge_config(args)
    show_config(args)
    return args


def save_uncertain_table(timestamp_list, uncertain_scores, image_path_list, save_path):
    logging.info('save uncertain table to {}'.format(save_path))
    with open(save_path, 'w') as f:
        uncertain_table = []

        table_length = len(timestamp_list)
        assert table_length == len(uncertain_scores)

        for i in range(table_length):
            uncertain_table.append({
                'timestamp': timestamp_list[i],
                'prob': uncertain_scores[i].tolist(),
                'image_path': image_path_list[i].strip()
            })

        json.dump(uncertain_table, f, indent=4)


class ObjectCountDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        with open(txt_file, 'r') as f:
            image_path_list = f.readlines()
            # self.image_path_list = list(map(lambda x: x.replace('vdata/taipei-bus', '/data/ssd/public/cxhan').strip(), image_path_list))
            self.image_path_list = list(map(lambda x: x.strip(), image_path_list))
            self.image_path_list.sort(key=lambda x: (len(x), x))
            self.image_path_list = self.image_path_list[:1000]
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
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def run_mdn_model(args):
    """phase 1: run mixed density model for the whole test set"""
    logging.info('phase 1 start.')

    start = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

    test_data = ObjectCountDataset(
        txt_file=args.test_data_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_threads,
        pin_memory=True
    )

    with open(args.test_data_path, 'r') as f:
        image_path_list = f.readlines()
        image_path_list.sort(key=lambda x: (len(x), x))
        image_path_list = image_path_list[:1000]

        timestamp_list = [int(x.split('/')[-1].split('.')[0]) for x in image_path_list]
        mdn_list = []

        # initiate template
        template = 0
        mdn_cache = None

        model = ResNetMDN(training=False).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path))
        model.eval()

        num_skip = 0
        max_skip = 0
        with torch.no_grad():
            for data in tqdm(test_loader):
                diff = difference_detector(template, data.numpy(), args.threshold)

                if diff:
                    image = data.to(device=device, non_blocking=True)
                    (pi, sigma, mu) = model(image)

                    mdn = (pi, sigma, mu)
                    mdn_cache = mdn
                    max_skip = max(max_skip, num_skip)
                    num_skip = 1
                else:
                    mdn = mdn_cache
                    num_skip += 1

                mdn_list.append(mdn)

    # pi: (N, G), sigma: (N, G, 1), mu: (N, G, 1)
    error = 1e-5
    error = torch.Tensor([1-error]).cuda()
    pi = torch.cat([x[0] for x in mdn_list], dim=0)
    sigma = torch.cat([x[1] for x in mdn_list], dim=0)
    mu = torch.cat([x[2] for x in mdn_list], dim=0)
    normals = normal.Normal(loc=mu, scale=sigma)
    max_count = normals.icdf(error).max().item()
    max_count = math.ceil(max_count)

    pi = pi.unsqueeze(-1)
    scale = torch.arange(max_count+1).cuda()
    uncertain_scores = (normals.cdf(scale) * pi).sum(1).cpu().numpy()

    save_uncertain_table(timestamp_list, uncertain_scores, image_path_list, args.table_path)

    end = time.time()
    elapsed = end - start

    logging.info('save table to {}'.format(args.table_path))

    return elapsed


def calibrate_probability(args, scores):
    """phase 2: calibrate probability"""
    logging.info('phase 2 start.')
    if args.calibrator_path is None:
        logging.info('skip phase 2.')
        elapsed = 0
        return elapsed, scores

    start = time.time()

    shape = scores.shape

    logging.info('loading calibrator {}'.format(args.calibrator_path))
    with open(args.calibrator_path, 'rb') as f:
        ir = pickle.load(f)

    scores = ir.predict(scores.flatten())
    scores = scores.reshape(shape)

    end = time.time()
    elapsed = end - start

    return elapsed, scores


def build_uncertain_table(args, scores, timestamp_list, image_path_list):
    """phase 3: build table from detection prediction"""
    logging.info('phase 3 start.')
    start = time.time()

    uncertain_scores = build_uncertain_table_fast(scores)
    save_uncertain_table(timestamp_list, uncertain_scores, image_path_list, args.table_path)

    end = time.time()
    elapsed = end - start

    return elapsed


def get_topk(args):
    """phase 4: get topk result from table"""
    logging.info('phase 4 start.')
    start = time.time()

    out = []
    op_args = dict(
        config_path=args.config_path,
        weight_path=args.weight_path,
        class_path=args.class_path,
        class_name=args.class_name,
        table_path=args.table_path,
        k=args.k,
        confidence=args.confidence,
        batch_size=args.batch_size
    )
    if args.gap:
        op_type = TopKGAPOp
        op_args.update(gap=args.gap)
    else:
        op_type = TopKOp
    op = op_type(**op_args)
    op.forward(None, out)

    end = time.time()
    elapsed = end - start

    return elapsed, out[0], out[1], out[2], out[3]


def precision(approx, exact):
    k = len(exact)

    tp = 0
    exact_dict = dict()
    for v in exact:
        if v in exact_dict:
            exact_dict[v] += 1
        else:
            exact_dict[v] = 1

    for v in approx:
        if v in exact_dict and exact_dict[v] > 0:
            tp += 1
            exact_dict[v] -= 1

    return tp / k


def recall(approx, exact):
    k = len(exact)

    tp = 0
    approx_dict = dict()
    for v in approx:
        if v in approx_dict:
            approx_dict[v] += 1
        else:
            approx_dict[v] = 1

    for v in exact:
        if v in approx_dict and approx_dict[v] > 0:
            tp += 1
            approx_dict[v] -= 1

    return tp / k


def rank_distance(approx, exact):
    k = len(exact)

    rank = np.zeros((k,))
    # assume rank k+1 if missing element
    rank[:] = k
    j = 0
    for i in range(k):
        if j > k-1:
            break
        while approx[i] != exact[j] and j < k-1:
            j += 1
        if approx[i] == exact[j]:
            rank[i] = j
            j += 1

    dis = np.sum(np.abs(np.arange(k) - rank))

    return dis / k


def score_error(approx, exact):
    k = len(exact)

    approx_ = np.array(approx)
    exact_ = np.array(exact)

    err = np.sum(np.abs(approx_ - exact_))

    return err / k


def evaluate(args, topk_value, topk_indices, elapsed):
    """phase 5: evaluate"""
    with open(args.label_path, 'r') as f:
        reader = json.load(f)
        reader.sort(key=lambda x: x['timestamp'])
        reader = reader[:1000]
    labels = sorted(reader, key=lambda x: x['count'], reverse=True)
    if args.gap == 0:
        topk_labels = labels[:args.k]
        gt_value = [label['count'] for label in topk_labels]
        gt_indices = [label['timestamp'] for label in topk_labels]
    else:
        gslist = GSList(args.gap, args.k)
        for i in range(len(labels)):
            v = labels[i]['count']
            t = labels[i]['timestamp']
            gslist.push(VT(v=v, t=t))
            if len(gslist.top()) == args.k:
                break
        top = gslist.top()
        gt_value = [x.v for x in top]
        gt_indices = [x.t for x in top]

    elapsed = [round(x, 3) for x in elapsed]

    prec = precision(topk_value, gt_value)
    rec = recall(topk_value, gt_value)
    rd = rank_distance(topk_value, gt_value)
    se = score_error(topk_value, gt_value)

    logging.info('----------------------------------------')
    logging.info('Time: {}, {:.3f}'.format(elapsed, np.sum(elapsed)))
    logging.info('Top-{} value: {}'.format(args.k, topk_value))
    logging.info('Top-{} gt value: {}'.format(args.k, gt_value))
    logging.info('Top-{} indices: {}'.format(args.k, topk_indices))
    logging.info('Top-{} gt indices: {}'.format(args.k, gt_indices))
    logging.info('Precision: {}'.format(prec))
    logging.info('Recall: {}'.format(rec))
    logging.info('Rank Distance: {}'.format(rd))
    logging.info('Score Error: {}'.format(se))

    return prec, rec, rd, se


def run(args):
    elapsed = []

    if not args.read_table:
        t1 = run_mdn_model(args)
        elapsed.append(t1)
    t2, topk_value, topk_indices, niter, niter_select = get_topk(args)
    elapsed.append(t2)
    prec, rec, rd, se = evaluate(args, topk_value, topk_indices, elapsed)

    return elapsed, niter, niter_select, [prec, rec, rd, se]


if __name__ == '__main__':
    args = parse_config()
    set_logger(args.log_path)

    run(args)
