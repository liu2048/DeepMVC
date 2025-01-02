import torch as th
import torch.nn as nn
import numpy as np
from sklearn import preprocessing, decomposition
from scipy.optimize import linear_sum_assignment
from scipy.special import comb
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
import wandb
import argparse
import plotly.figure_factory as ff
from contextlib import contextmanager
from copy import deepcopy
from tabulate import tabulate
from typing import Union, Dict, Optional, Any

import helpers
import config
from lib.encoder import Encoder
from lib.metrics import cmat_from_dict
from lib.wandb_utils import WANDB_PROJECT, get_default_run_info


# Utility functions
def kernel_from_distance_matrix(dist, sigma):
    dist = nn.functional.relu(dist)
    k = th.exp(- dist / (2 * sigma**2))
    return k


def vector_kernel(x, sigma):
    return kernel_from_distance_matrix(cdist(x, x), sigma=sigma)


def cdist(X, Y):
    xyT = X @ th.t(Y)
    x2 = th.sum(X**2, dim=1, keepdim=True)
    y2 = th.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + th.t(y2)
    return d


def ordered_cmat(labels, pred):
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered


def cmat_to_dict(cmat, prefix=""):
    return {prefix + f"{i}_{j}": cmat[i, j] for i in range(cmat.shape[0]) for j in range(cmat.shape[1])}


def cmat_from_dict(dct, prefix="", del_elements=False):
    n_clusters = 0
    while prefix + f"{n_clusters}_{0}" in dct.keys():
        n_clusters += 1

    out = np.empty((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            key = prefix + f"{i}_{j}"
            out[i, j] = int(dct[key])

            if del_elements:
                del dct[key]

    return out.astype(int)


def calc_metrics(labels, pred, flatten_cmat=True):
    acc, cmat = ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "nmi": normalized_mutual_info_score(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
    }
    if flatten_cmat:
        metrics.update(cmat_to_dict(cmat, prefix="cmat/"))
    else:
        metrics["cmat"] = cmat
    return metrics


class L2Norm(nn.Module):
    def __init__(self, dim=-1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return nn.functional.normalize(inputs, dim=self.dim)


def get_normalizer(name, dim=-1):
    if name is None:
        norm = nn.Identity()
    elif name == "l2":
        norm = L2Norm(dim=dim)
    elif name == "softmax":
        norm = nn.Softmax(dim=dim)
    else:
        raise RuntimeError(f"Invalid normalizer: {name}")
    return norm


class IdentityProjector(nn.Module):
    @staticmethod
    def forward(inputs):
        return inputs


class Projector(nn.Module):
    def __init__(self, cfg, input_sizes):
        super(Projector, self).__init__()

        self.output_sizes = deepcopy(input_sizes)

        if cfg is None:
            self.op = IdentityProjector()
            self._forward = self.op

        elif cfg.layout == "separate":
            encoder_config = cfg.encoder_config
            if not isinstance(encoder_config, (list, tuple)):
                encoder_config = len(input_sizes) * [encoder_config]

            self.op = EncoderList(encoder_config, input_sizes=input_sizes)
            self._forward = self._list_forward
            self.output_sizes = self.op.output_sizes

        elif cfg.layout == "shared":
            assert all([input_sizes[0] == s for s in input_sizes]), "Shared projection head assumes that all encoder " \
                                                                    "output sizes are equal"
            self.op = Encoder(cfg.encoder_config, input_size=input_sizes[0])
            self._forward = self._concat_forward
            self.output_sizes = len(input_sizes) * [self.op.output_size]

        else:
            raise ValueError(f"Invalid projector layout: {cfg.layout}")

    def _list_forward(self, views):
        return self.op(views)

    def _concat_forward(self, views):
        v, n = len(views), views[0].size(0)
        projections = self.op(th.cat(views, dim=0))
        projections = [projections[idx] for idx in th.arange(n * v).view(v, n)]
        return projections

    def forward(self, views):
        return self._forward(views)
