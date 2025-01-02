import os
import wandb
import torch as th
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl

import config
import helpers
from data.data_module import DataModule
from models.build_model import build_model
from lib.loggers import ConsoleLogger, WeightsAndBiasesLogger
from lib.evaluate import evaluate, log_best_run
from lib import wandb_utils


def pre_train(cfg, net, data_module, save_dir, wandb_logger, console_logger):
    print(f"{80 * '='}\nPre-training started\n{80 * '='}")
    best_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename="pre_train_best", verbose=True,
                                                 monitor="val_loss/tot", mode="min", every_n_epochs=cfg.eval_interval,
                                                 save_top_k=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename="pre_train_checkpoint_{epoch:04d}",
                                                       verbose=True, save_top_k=-1,
                                                       every_n_epochs=cfg.checkpoint_interval,
                                                       save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[best_callback, checkpoint_callback],
        logger=[wandb_logger, console_logger],
        log_every_n_steps=data_module.n_batches,
        check_val_every_n_epoch=cfg.eval_interval,
        enable_progress_bar=False,
        max_epochs=cfg.n_pre_train_epochs,
        gpus=cfg.gpus,
        deterministic=cfg.trainer_deterministic,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        detect_anomaly=cfg.detect_anomaly,
    )
    trainer.fit(net, datamodule=data_module)
    print(f"{80 * '='}\nPre-training finished\n{80 * '='}")


def train(cfg, net, data_module, save_dir, console_logger):
    print(f"{80 * '='}\nTraining started\n{80 * '='}")

    # 设置优化器和损失函数
    optimizer = optim.Adam(net.parameters(), lr=cfg.model_config.optimizer_config.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(cfg.n_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(data_module.train_dataloader(), 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % cfg.log_interval == cfg.log_interval - 1:
                console_logger.log(f"[{epoch + 1}, {i + 1}] loss: {running_loss / cfg.log_interval:.3f}")
                running_loss = 0.0

        # 验证
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in data_module.val_dataloader():
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        console_logger.log(f"Validation loss after epoch {epoch + 1}: {val_loss / len(data_module.val_dataloader()):.3f}")

        # 保存模型
        torch.save(net.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    print("Finished Training")

    # ==== Evaluate ====
    # Validation set
    net.test_prefix = "val"
    val_results = evaluate(net, os.path.join(save_dir, f"checkpoint_epoch_{cfg.n_epochs}.pth"), data_module.val_dataloader(), console_logger)
    # Test set
    net.test_prefix = "test"
    test_results = evaluate(net, os.path.join(save_dir, f"checkpoint_epoch_{cfg.n_epochs}.pth"), data_module.test_dataloader(), console_logger)
    # Log evaluation results
    wandb.join()

    return val_results, test_results


def set_seeds(seed=None, workers=False, offset=0, deterministic_algorithms=True):
    if seed is not None:
        pl.seed_everything(seed + offset, workers=workers)
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
            console_logger=console_logger,
        )
        val_logs.append(val)
        test_logs.append(test)

    best_val_logs, best_test_logs = log_best_run(val_logs, test_logs, cfg, ename, tag)
    return val_logs, test_logs, best_val_logs, best_test_logs


if __name__ == '__main__':
    print("Torch version:", th.__version__)
    print("Lightning version:", pl.__version__)

    ename, cfg = config.get_experiment_config()
    wandb_env_vars = wandb_utils.clear_wandb_env()

    tag = wandb_utils.get_experiment_tag()

    all_logs = main(ename, cfg, tag)

    if cfg.is_sweep:
        # Log to the original sweep-run if this experiment is part of a sweep
        sweep_run = wandb_utils.init_sweep_run(ename, tag, cfg, wandb_env_vars)
        wandb_utils.finalize_sweep_run(sweep_run, all_logs)
