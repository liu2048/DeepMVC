import os
import warnings
import torch as th
from copy import deepcopy

import config
from lib.logging import fix_cmat


def evaluate(net, ckpt_path, loader, logger):
    if not ckpt_path:
        warnings.warn(f"No 'best' checkpoint found at: '{ckpt_path}'.")
        ckpt_path = None

    net.load_state_dict(th.load(ckpt_path)["state_dict"])
    net.eval()

    results = []
    with th.no_grad():
        for batch in loader:
            output = net(batch)
            results.append(output)

    return results


def log_best_run(val_logs_list, test_logs_list, cfg, experiment_name, tag):
    best_run = None
    best_loss = float("inf")
    for run, logs in enumerate(val_logs_list):
        tot_loss = logs[f"val_loss/{cfg.best_loss_term}"]
        if tot_loss < best_loss:
            best_run = run
            best_loss = tot_loss

    def _log_best(set_type, best_logs):
        best_logs = deepcopy(best_logs)
        best_logs["is_best"] = True
        best_logs["best_run"] = best_run

        if f"{set_type}_metrics/cmat" not in best_logs:
            fix_cmat(best_logs, set_type=set_type)

    best_val_logs = val_logs_list[best_run]
    best_test_logs = test_logs_list[best_run]

    _log_best("val", best_val_logs)
    _log_best("test", best_test_logs)

    return best_val_logs, best_test_logs
