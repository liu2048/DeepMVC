import os
import numpy as np
import torch as th
from sklearn import preprocessing, decomposition
from skimage.io import imread
from skimage.transform import resize
from scipy.io import loadmat
import cv2
import sys
from tabulate import tabulate
from tqdm.auto import tqdm
import faiss
from argparse import ArgumentParser

import config
from data import torchvision_datasets


RAW_DIR = config.DATA_DIR / "raw"
PROCESSED_DIR = config.DATA_DIR / "processed"
EXPORTERS = {}


def _load_npz(name, split):
    file_path = config.DATA_DIR / "processed" / f"{name}_{split}.npz"
    if not os.path.exists(file_path):
        print(f"Could not find dataset '{name} ({split})' at {file_path}.")
        return None
    return np.load(file_path)


def _fix_labels(l):
    uniq = np.unique(l)[None, :]
    new = (l[:, None] == uniq).argmax(axis=1)
    return new


def _normalize(views, mode):
    if mode == "l2":
        views = [preprocessing.normalize(v, norm="l2") for v in views]
    elif mode == "minmax":
        views = [preprocessing.minmax_scale(v, feature_range=(0, 1)) for v in views]
    else:
        raise RuntimeError(f"Invalid normalization mode: {mode}")
    return views


def _pca(X, out_dim):
    if out_dim is None:
        return X
    return decomposition.PCA(n_components=out_dim).fit_transform(X)


def _flatten_list(lst):
    out = []
    for elem in lst:
        if isinstance(elem, list):
            out += elem
        else:
            out.append(elem)
    return out


def load_dataset(name, split="train", random_seed=None, n_samples=None,
                 select_views=None, select_labels=None, label_counts=None, noise_sd=None, noise_views=None,
                 to_dataset=True, normalization=None, pca_dims=None, include_index=False):

    npz = _load_npz(name, split)
    if npz is None:
        return

    labels = npz["labels"]
    views = [npz[f"view_{i}"] for i in range(npz["n_views"])]

    if random_seed is not None:
        prev_state = np.random.get_state()
        np.random.seed(random_seed)

    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        views = [v[mask] for v in views]
        labels = _fix_labels(labels)

    if label_counts is not None:
        idx = []
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(label_counts)
        for l, n in zip(unique_labels, label_counts):
            _idx = np.random.choice(np.where(labels == l)[0], size=n, replace=False)
            idx.append(_idx)

        idx = np.concatenate(idx, axis=0)
        labels = labels[idx]
        views = [v[idx] for v in views]

    if n_samples is not None:
        idx = np.random.choice(labels.shape[0], size=min(labels.shape[0], int(n_samples)), replace=False)
        labels = labels[idx]
        views = [v[idx] for v in views]

    if select_views is not None:
        if not isinstance(select_views, (list, tuple)):
            select_views = [select_views]
        views = [views[i] for i in select_views]

    if noise_sd is not None:
        assert noise_views is not None, "'noise_views' has to be specified when 'noise_sd' is not None."
        for v in noise_views:
            views[v] += np.random.normal(loc=0, scale=noise_sd, size=views[v].shape)

    if normalization is not None:
        views = _normalize(views, normalization)

    if pca_dims is not None:
        assert len(pca_dims) == len(views)
        views = [_pca(v, d) for v, d in zip(views, pca_dims)]

    views = [v.astype(np.float32) for v in views]

    dataset = [views, labels]

    if include_index:
        dataset.insert(2, np.arange(labels.shape[0]))

    if to_dataset:
        tensors = [th.from_numpy(arr) for arr in _flatten_list(dataset)]
        dataset = th.utils.data.TensorDataset(*tensors)

    if random_seed is not None:
        np.random.set_state(prev_state)

    return dataset


def print_summary(name, views, labels, file_path, suffix):
    uniq, count = np.unique(labels, return_counts=True)
    rows = [
        ["name", name],
        ["suffix", suffix],
        ["file_path", file_path],
        *[[f"view_{i}.shape", v.shape] for i, v in enumerate(views)],
        *[[f"view_{i}.(min,max)", (round(v.min(), 3), round(v.max(), 3))] for i, v in enumerate(views)],
        ["labels.shape", labels.shape],
        ["n_clusters", len(uniq)],
        ["unique labels", " ".join([str(u) for u in uniq])],
        ["label counts", " ".join([str(c) for c in count])]
    ]
    print(tabulate(rows), end="\n\n")


def export_dataset(name, views, labels, suffix="train"):
    processed_dir = config.DATA_DIR / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    file_path = processed_dir / f"{name}_{suffix}.npz"
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)
    print_summary(name, views, labels, file_path, suffix)


def exporter(name=None, split="train", **kwargs):
    def decorator(load_func):
        _name = name or load_func.__name__

        def wrapper():
            views, labels = load_func(**kwargs)
            export_dataset(_name, views, labels, suffix=split)

        EXPORTERS[_name] = wrapper
        return wrapper

    return decorator


def max_min_normalize(x, low=0, high=1):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = (x * (high - low)) + low
    return x


def _noisy_mnist(view_1, labels):
    rng = np.random.default_rng(7)
    view_2 = np.empty_like(view_1)
    for lab in np.unique(labels):
        idx = np.nonzero(labels == lab)[0]
        perm_idx = rng.permutation(idx)
        view_2[idx] = view_1[perm_idx]

    view_2 += rng.normal(0, 0.2, size=view_2.shape)

    view_1 = max_min_normalize(view_1, low=0, high=1)
    view_2 = max_min_normalize(view_2, low=0, high=1)
    return [view_1, view_2]


def _concat_edge_image(img):
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return np.stack((img, edge), axis=-1)


@exporter()
def noisymnist():
    data, labels = torchvision_datasets.mnist()
    views = _noisy_mnist(data, labels)
    return views, labels


@exporter()
def noisyfashionmnist():
    data, labels = torchvision_datasets.fashion_mnist()
    views = _noisy_mnist(data, labels)
    return views, labels


@exporter()
def edgemnist():
    data, labels = torchvision_datasets.mnist(custom_transforms=(_concat_edge_image,))
    views = np.split(data, data.shape[1], axis=1)
    views = [max_min_normalize(v, low=0, high=1) for v in views]
    return views, labels


@exporter()
def edgefashionmnist():
    data, labels = torchvision_datasets.fashion_mnist(custom_transforms=(_concat_edge_image,))
    views = np.split(data, data.shape[1], axis=1)
    views = [max_min_normalize(v, low=0, high=1) for v in views]
    return views, labels


@exporter()
def patchedmnist():
    data, labels = torchvision_datasets.mnist()

    select_labels = (0, 1, 2)
    mask = np.isin(labels, select_labels)
    data, labels = data[mask], labels[mask]

    h, w = data.shape[2], data.shape[3]
    new_hw = (4 * h, 4 * w)
    resized = np.empty((data.shape[0], 1, *new_hw))

    print("Resizing images...")
    for i, img in enumerate(data):
        resized_img = resize(image=img[0], output_shape=new_hw)
        resized[i, 0] = resized_img

    print("Creating patches...")
    views = []
    for i in range(4):
        for j in range(4):
            if (i in {0, 3}) and (j in {0, 3}) {
                # Skip corner patches
                continue
            }

            h_min, h_max = i * h, (i + 1) * h
            w_min, w_max = j * w, (j + 1) * w
            patches = resized[:, :, h_min: h_max, w_min: w_max]
            views.append(max_min_normalize(patches, low=0, high=1))

    return views, labels


def _caltech(select_categories=None):
    mat = loadmat(str(RAW_DIR / "Caltech101-20.mat"), simplify_cells=True)
    views = list(mat["X"])
    views = [max_min_normalize(x) for x in views]
    labels = (mat["Y"] - 1).astype(int)

    if select_categories is not None:
        cats = mat["categories"]
        select_labels = np.isin(cats, select_categories).nonzero()[0]
        mask = np.isin(labels, select_labels)
        views = [v[mask] for v in views]
        labels = labels[mask]
        labels = _fix_labels(labels)

    views = [max_min_normalize(v) for v in views]
    return views, labels


@exporter()
def caltech20():
    return _caltech()


@exporter()
def caltech7():
    select_categories = ["Faces", "Motorbikes", "dollar_bill", "garfield", "snoopy", "stop_sign", "windsor_chair"]
    return _caltech(select_categories=select_categories)


@exporter()
def coil20():
    rng = np.random.default_rng(7)
    data_dir = config.DATA_DIR / "raw" / "coil20"
    img_size = (1, 64, 64)
    n_objs = 20
    n_imgs = 72
    n_views = 3
    assert n_imgs % n_views == 0

    n = (n_objs * n_imgs) // n_views
    imgs = np.empty((n_views, n, *img_size))
    labels = []
    img_idx = np.arange(n_imgs)
    for obj in range(n_objs):
        obj_img_idx = rng.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]
        for view, indices in enumerate(obj_img_idx):
            for i, idx in enumerate(indices):
                file_name = data_dir / f"obj{obj + 1}__{idx}.png"
                img = imread(file_name)
                img = resize(img, output_shape=img_size[1:], anti_aliasing=True)
                img = max_min_normalize(img)
                imgs[view, ((obj * (n_imgs // n_views)) + i)] = img[None, ...]

    assert not np.isnan(imgs).any()
    views = [imgs[v] for v in range(n_views)]
    labels = np.array(labels)
    return views, labels


@exporter()
def blobs_dep():
    centers = np.array([
        [0, 0, 0, 0],
        [0, 5, 5, 0],
        [5, 0, 0, 5]
    ])
    cov = np.array([
        [.5, 0, .5, 0],
        [0, .5, 0, .5],
        [.5, 0, .5, 0],
        [0, .5, 0, .5],
    ])

    data = np.concatenate([np.random.multivariate_normal(mean=c, cov=cov, size=1000) for c in centers], axis=0)
    views = [data[:, :2], data[:, 2:]]
    labels = np.concatenate([np.full(1000, i) for i in range(3)], axis=0)
    return views, labels


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, dest="dataset", type=str)
    parser.add_argument("-k", dest="k", default=3, type=int)
    return parser.parse_args()


def make_index(x, gpu=False):
    index = faiss.IndexFlatL2(x.shape[1])
    if gpu:
        index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            index,
        )
    index.add(x)
    return index


def generate_pairs(x, k):
    x = x.astype(np.float32)
    x_flat = np.ascontiguousarray(x.reshape((x.shape[0], -1)))

    index = make_index(x_flat)
    ngh_idx = index.search(x_flat, k + 1)[0][:, 1:].astype(int)
    rng = np.random.default_rng(7)
    n_samples = x.shape[0]
    rang = np.arange(n_samples)

    pairs, labels = [], []

    for i in tqdm(range(n_samples)):
        pos = [np.stack((x[i], x[j]), axis=0) for j in ngh_idx[i]]
        pairs += pos

        neg_weights = np.ones(n_samples)
        neg_weights[ngh_idx[i]] = 0
        neg_weights[i] = 0
        neg_weights /= neg_weights.sum()
        neg_idx = rng.choice(rang, size=k, replace=False, p=neg_weights)

        neg = [np.stack((x[i], x[j]), axis=0) for j in neg_idx]
        pairs += neg

        labels += (k * [1]) + (k * [0])

    pairs = np.stack(pairs, axis=0)
    labels = np.array(labels)

    print("Paired shape:", pairs.shape)
    print("Pair labels shape:", labels.shape)
    assert labels.shape[0] == pairs.shape[0]
    return pairs, labels


def main():
    args = parse_args()
    data = np.load(str(PROCESSED_DIR / f"{args.dataset}_train.npz"))
    n_views = data["n_views"]
    views = [data[f"view_{v}"] for v in range(n_views)]

    new_labels = np.stack(2 * args.k * [data["labels"]], axis=1).ravel()
    pair_data = {"n_views": 2 * n_views, "labels": new_labels}
    for i, v in enumerate(views):
        pairs, pair_lab = generate_pairs(x=v, k=args.k)
        pair_data[f"view_{i}"] = pairs
        pair_data[f"view_{n_views + i}"] = pair_lab

    out_file = str(PROCESSED_DIR / f"{args.dataset}_paired_train")
    np.savez(out_file, **pair_data)
    print(f"Successfully saved paired data to '{out_file}'")


if __name__ == '__main__':
    main()
