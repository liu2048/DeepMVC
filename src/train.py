import os
import wandb
import torch as th
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import config
import helpers
from data.data_module import DataModule
from models.build_model import build_model
from lib.logging import ConsoleLogger, WeightsAndBiasesLogger
from lib.evaluation import evaluate, log_best_run
from lib import wandb_utils


def pre_train(cfg, net, data_module, save_dir, wandb_logger, console_logger):
    print(f"{80 * '='}\nPre-training started\n{80 * '='}")
    best_model_path = os.path.join(save_dir, "pre_train_best.pth")
    checkpoint_path = os.path.join(save_dir, "pre_train_checkpoint.pth")

    optimizer = Adam(net.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    for epoch in range(cfg.n_pre_train_epochs):
        net.train()
        for batch in data_module.train_dataloader():
            optimizer.zero_grad()
            loss = net.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % cfg.eval_interval == 0:
            val_loss = evaluate(net, data_module.val_dataloader())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                th.save(net.state_dict(), best_model_path)

        if epoch % cfg.checkpoint_interval == 0:
            th.save(net.state_dict(), checkpoint_path)

    print(f"{80 * '='}\nPre-training finished\n{80 * '='}")


def train(cfg, net, data_module, save_dir, wandb_logger, console_logger, initial_epoch=0):
    best_model_path = os.path.join(save_dir, "best.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    optimizer = Adam(net.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    best_val_loss = float('inf')

    for epoch in range(initial_epoch, cfg.n_epochs + initial_epoch):
        net.train()
        for batch in data_module.train_dataloader():
            optimizer.zero_grad()
            loss = net.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % cfg.eval_interval == 0:
            val_loss = evaluate(net, data_module.val_dataloader())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                th.save(net.state_dict(), best_model_path)

        if epoch % cfg.checkpoint_interval == 0:
            th.save(net.state_dict(), checkpoint_path)

    # ==== Evaluate ====
    # Validation set
    net.test_prefix = "val"
    val_results = evaluate(net, best_model_path, data_module.val_dataloader(), console_logger)
    # Test set
    net.test_prefix = "test"
    test_results = evaluate(net, best_model_path, data_module.test_dataloader(), console_logger)
    # Log evaluation results
    wandb_logger.log_summary(val_results, test_results)
    wandb.join()

    return val_results, test_results


def set_seeds(seed=None, workers=False, offset=0, deterministic_algorithms=True):
    if seed is not None:
        th.manual_seed(seed + offset)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed + offset)
    # th.use_deterministic_algorithms(deterministic_algorithms)


def main(ename, cfg, tag):
    set_seeds(cfg.everything_seed)
    data_module = DataModule(cfg.dataset_config)

    val_logs, test_logs = [], []
    for run in range(cfg.n_runs):
        wandb_utils.clear_wandb_env()
        set_seeds(seed=cfg.everything_seed, offset=run)

        net = build_model(cfg.model_config, run=run)
        print(net)
        net.attach_data_module(data_module)

        save_dir = helpers.get_save_dir(ename, tag, run)
        os.makedirs(save_dir, exist_ok=True)
        cfg.to_pickle(save_dir / "config.pkl")

        wandb_logger = WeightsAndBiasesLogger(ename, tag, run, cfg, net)
        console_logger = ConsoleLogger(ename, print_cmat=(cfg.n_clusters <= 10))

        initial_epoch = 0

        if net.requires_pre_train:
            net.init_pre_train()
            pre_train(
                cfg=cfg,
                net=net,
                data_module=data_module,
                save_dir=save_dir,
                wandb_logger=wandb_logger,
                console_logger=console_logger,
            )
            net.init_fine_tune()
            console_logger.epoch_offset = cfg.n_pre_train_epochs
            wandb_logger.epoch_offset = cfg.n_pre_train_epochs

        val, test = train(
            cfg=cfg,
            net=net,
            data_module=data_module,
            save_dir=save_dir,
            wandb_logger=wandb_logger,
            console_logger=console_logger,
            initial_epoch=initial_epoch,
        )
        val_logs.append(val)
        test_logs.append(test)

    best_val_logs, best_test_logs = log_best_run(val_logs, test_logs, cfg, ename, tag)
    return val_logs, test_logs, best_val_logs, best_test_logs


if __name__ == '__main__':
    print("Torch version:", th.__version__)

    ename, cfg = config.get_experiment_config()
    wandb_env_vars = wandb_utils.clear_wandb_env()

    tag = wandb_utils.get_experiment_tag()

    all_logs = main(ename, cfg, tag)

    if cfg.is_sweep:
        # Log to the original sweep-run if this experiment is part of a sweep
        sweep_run = wandb_utils.init_sweep_run(ename, tag, cfg, wandb_env_vars)
        wandb_utils.finalize_sweep_run(sweep_run, all_logs)
