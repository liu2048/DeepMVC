from typing import Tuple, List, Optional
from config import Config, constants
from config.templates import models, augmenter

class Dataset(Config):
    # Name of the dataset. Must correspond to a filename in data/processed/
    name: str

    # Seed for random data loading ops.
    random_seed: int = 7

    # Include indices of batch elements in the dataset
    include_index: bool = False

    # Number of samples to load. Set to None to load all samples
    n_train_samples: Optional[int] = None
    n_val_samples: Optional[int] = None
    n_test_samples: Optional[int] = None

    # Subset of views to load. Set to None to load all views
    select_views: Optional[List[int]] = None

    # Subset of labels (classes) to load. Set to None to load all classes
    select_labels: Optional[List[int]] = None

    # Number of samples to load for each class. Set to None to load all samples
    train_label_counts: Optional[List[int]] = None
    val_label_counts: Optional[List[int]] = None
    test_label_counts: Optional[List[int]] = None

    # Drop last batch (if not a complete batch), when dataset is batched.
    drop_last: bool = True

    # Whether to shuffle the validation and test data
    train_shuffle: bool = True
    val_shuffle: bool = False
    test_shuffle: bool = False

    # Number of DataLoader workers
    n_train_workers: int = 8
    n_val_workers: int = 8
    n_test_workers: int = 8

    # Prefetch factor for train dataloader (only used when n_train_workers > 0).
    prefetch_factor: int = 1

    # Config for data augmentation. Set to None to disable augmentation.
    augmenter_configs: Optional[List[augmenter.Augmenter]] = None

    # Pre-train-specific parameters. Set to None to use same values as in fine-tune (specified above)
    pre_train_batch_size: Optional[int] = None
    pre_train_train_shuffle: Optional[bool] = None
    pre_train_val_shuffle: Optional[bool] = None
    pre_train_test_shuffle: Optional[bool] = None

    # Batch size (This is a placeholder. Set the batch size in Experiment).
    batch_size: Optional[int] = None