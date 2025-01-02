import argparse
import numpy as np
from copy import deepcopy
from tabulate import tabulate
from typing import Union, Dict, Optional, Any

import helpers
import config
from lib.metrics import cmat_from_dict


def fix_cmat(metrics, set_type, cmat_hook=lambda x: x):
    if f"{set_type}_metrics/cmat/0_0" not in metrics:
        return

    cmat = cmat_from_dict(metrics, prefix=f"{set_type}_metrics/cmat/", del_elements=True)
    metrics[f"{set_type}_metrics/cmat"] = cmat_hook(cmat)


class ConsoleLogger:
    def __init__(self, ename, ignore_keys=tuple(), print_cmat=True):
        self.ignore_keys = list(ignore_keys)
        self.print_cmat = print_cmat
        self.epoch_offset = 0
        self._ename = ename
        self._version = "0"

        if not self.print_cmat:
            self.ignore_keys.append("val_metrics/cmat")
            self.ignore_keys.append("test_metrics/cmat")

    def log_metrics(self, metrics, step=None):
        print_logs = deepcopy(metrics)
        for key in metrics.keys():
            if any([key.startswith(ik) for ik in self.ignore_keys]):
                del print_logs[key]

        if self.print_cmat:
            fix_cmat(print_logs, "val")
            fix_cmat(print_logs, "test")

        if "epoch" in print_logs:
            print_logs["epoch"] += self.epoch_offset

        headers = list(print_logs.keys())
        values = list(print_logs.values())

        if "epoch" in headers:
            helpers.move_elem_to_idx(headers, elem="epoch", idx=0, twins=(values,))
        if "time_delta" in headers:
            helpers.move_elem_to_idx(headers, elem="time_delta", idx=1, twins=(values,))

        print(tabulate([values], headers=headers), "\n")
