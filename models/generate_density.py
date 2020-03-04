import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import filters
# from sklearn.neighbors import NearestNeighbors


def create_density(gts, d_map_h, d_map_w):
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)
    for k in range(len(gts)):
        gt = gts[k]
        if bool_res[k]:
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    # num_gt = len(gts)
    # neighbors = NearestNeighbors(n_neighbors=min(4, num_gt-1), algorithm='kd_tree', leaf_size=100)
    # neighbors.fit(pts.copy())
    # distances, _ = neighbors.kneighbors()
    map_shape = [d_map_h, d_map_w]
    density = np.zeros((d_map_h, d_map_w), dtype=np.float32)
    # sigmas = distances.sum(axis=1) * 0.015
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        density += filters.gaussian_filter(pt2d, 15, mode='constant')
    return density


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--resize_width', type=int, required=True)
    parser.add_argument('--resize_height', type=int, required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    with open(args.image_path, 'r') as f:
        image_path_list = f.readlines()
        image_path_list = list(map(lambda x: x.strip(), image_path_list))

    for image_path in tqdm(image_path_list):
        label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(label_path, 'r') as f:
            label = f.readlines()
        gts = []
        for line in label:
            line = line.strip().split(' ')
            gts.append([float(line[1]), float(line[2])])
        gts = np.array(gts)

        d_map_h = args.resize_height
        d_map_w = args.resize_width
        h, w = image.shape[:2]
        ratio_h = h / d_map_h
        ratio_w = w / d_map_w
        gts[:, 0] *= w / ratio_w
        gts[:, 1] *= h / ratio_h
        den_map = create_density(gts, d_map_h, d_map_w)

        # save density map
        den_path = image_path.replace('images', 'density').replace('jpg', 'npy')
        np.save(den_path, den_map)

        # visualize
        if args.visualize:
            image_resize = cv2.resize(image, (d_map_w, d_map_h), interpolation=cv2.INTER_LINEAR)
            den_map = den_map
            den_map_min = np.min(den_map)
            den_map_max = np.max(den_map)
            den_map_mean = (den_map_min + den_map_max) / 2
            den_map -= den_map_mean
            den_map /= den_map_max - den_map_min
            den_map += 0.5
            den_map *= 255
            den_map = np.stack([den_map, np.full(den_map.shape, 0), np.full(den_map.shape, 0)], axis=-1)
            den_map = den_map * 0.5 + image_resize * 0.5
            den_map = den_map.astype(np.uint8)

            fig = plt.figure()
            plt.imshow(den_map)
            plt.show()
