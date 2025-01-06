# src/config Directory

This directory contains configuration files and templates for experiments, datasets, models, and other components of the project.

## Files

- `__init__.py`: Initializes the `config` module and provides utility functions for parsing and updating configurations.
- `config.py`: Defines the base `Config` class used for creating hierarchical configuration objects.
- `constants.py`: Contains constant values used throughout the project, such as directory paths and device settings.
- `experiments`: Directory containing experiment configurations and templates.
  - `__init__.py`: Initializes the `experiments` module and imports experiment configurations.
  - `ablation`: Directory containing ablation study configurations.
    - `__init__.py`: Initializes the `ablation` module and imports ablation study configurations.
    - `fusion.py`: Defines ablation study configurations for fusion methods.
    - `mv_ssl.py`: Defines ablation study configurations for multi-view self-supervised learning.
    - `sv_ssl.py`: Defines ablation study configurations for single-view self-supervised learning.
  - `base_experiments.py`: Defines base experiment configurations for various datasets.
  - `benchmark`: Directory containing benchmark experiment configurations.
    - `__init__.py`: Initializes the `benchmark` module and imports benchmark experiment configurations.
    - `comvc.py`: Defines benchmark experiment configurations for the CoMVC model.
    - `contrastive_ae.py`: Defines benchmark experiment configurations for the Contrastive Autoencoder model.
    - `dmsc.py`: Defines benchmark experiment configurations for the DMSC model.
    - `eamc.py`: Defines benchmark experiment configurations for the EAMC model.
    - `mimvc.py`: Defines benchmark experiment configurations for the MIMVC model.
    - `mvae.py`: Defines benchmark experiment configurations for the MVAE model.
    - `mviic.py`: Defines benchmark experiment configurations for the MvIIC model.
    - `mvscn.py`: Defines benchmark experiment configurations for the MvSCN model.
    - `simvc.py`: Defines benchmark experiment configurations for the SiMVC model.
  - `increasing_n_views`: Directory containing experiment configurations for increasing the number of views.
    - `__init__.py`: Initializes the `increasing_n_views` module and imports experiment configurations.
    - `caltech7.py`: Defines experiment configurations for increasing the number of views on the Caltech7 dataset.
    - `patchedmnist.py`: Defines experiment configurations for increasing the number of views on the PatchedMNIST dataset.
- `templates`: Directory containing configuration templates for various components.
  - `__init__.py`: Initializes the `templates` module and imports configuration templates.
  - `augmenter.py`: Defines configuration templates for data augmentation.
  - `clustering_module.py`: Defines configuration templates for clustering modules.
  - `dataset.py`: Defines configuration templates for datasets.
  - `encoder.py`: Defines configuration templates for encoders.
  - `experiment.py`: Defines the base `Experiment` class used for creating experiment configurations.
  - `fusion.py`: Defines configuration templates for fusion methods.
  - `kernel_width.py`: Defines configuration templates for kernel width estimation methods.
  - `layers.py`: Defines configuration templates for neural network layers.
  - `models`: Directory containing configuration templates for models.
    - `__init__.py`: Initializes the `models` module and imports model configuration templates.
    - `custom.py`: Defines custom model configuration templates.
    - `dmsc.py`: Defines configuration templates for the DMSC model.
    - `eamc.py`: Defines configuration templates for the EAMC model.
    - `mimvc.py`: Defines configuration templates for the MIMVC model.
    - `mvae.py`: Defines configuration templates for the MVAE model.
    - `mviic.py`: Defines configuration templates for the MvIIC model.
    - `mvscn.py`: Defines configuration templates for the MvSCN model.
    - `simvc_comvc.py`: Defines configuration templates for the SiMVC and CoMVC models.
  - `optimizer.py`: Defines configuration templates for optimizers.
