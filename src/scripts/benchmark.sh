#!/bin/bash

# This script runs benchmark experiments for the project.
# Usage: ./benchmark.sh

# Navigate to the src directory
cd "$(dirname "$0")/.."

# Run benchmark experiments for CoMVC model across different datasets
python train.py -c blobs_comvc
python train.py -c noisymnist_comvc
python train.py -c edgemnist_comvc
python train.py -c caltech20_comvc
python train.py -c caltech7_comvc
python train.py -c noisyfashionmnist_comvc
python train.py -c edgefashionmnist_comvc
python train.py -c coil20_comvc
python train.py -c patchedmnist_comvc

# Run benchmark experiments for CAE model across different datasets
python train.py -c noisymnist_cae
python train.py -c edgemnist_cae
python train.py -c caltech20_cae
python train.py -c caltech7_cae
python train.py -c noisyfashionmnist_cae
python train.py -c edgefashionmnist_cae
python train.py -c coil20_cae
python train.py -c patchedmnist_cae

# Run benchmark experiments for CAEKM model across different datasets
python train.py -c noisymnist_caekm
python train.py -c edgemnist_caekm
python train.py -c caltech20_caekm
python train.py -c caltech7_caekm
python train.py -c noisyfashionmnist_caekm
python train.py -c edgefashionmnist_caekm
python train.py -c coil20_caekm
python train.py -c patchedmnist_caekm

# Run benchmark experiments for SAE model across different datasets
python train.py -c noisymnist_sae
python train.py -c edgemnist_sae
python train.py -c caltech20_sae
python train.py -c caltech7_sae
python train.py -c noisyfashionmnist_sae
python train.py -c edgefashionmnist_sae
python train.py -c coil20_sae
python train.py -c patchedmnist_sae

# Run benchmark experiments for SAEKM model across different datasets
python train.py -c noisymnist_saekm
python train.py -c edgemnist_saekm
python train.py -c caltech20_saekm
python train.py -c caltech7_saekm
python train.py -c noisyfashionmnist_saekm
python train.py -c edgefashionmnist_saekm
python train.py -c coil20_saekm
python train.py -c patchedmnist_saekm

# Run benchmark experiments for DMSC model across different datasets
python train.py -c blobs_dmsc
python train.py -c noisymnist_dmsc
python train.py -c edgemnist_dmsc
python train.py -c noisyfashionmnist_dmsc
python train.py -c edgefashionmnist_dmsc
python train.py -c caltech20_dmsc
python train.py -c caltech7_dmsc
python train.py -c coil20_dmsc
python train.py -c patchedmnist_dmsc

# Run benchmark experiments for EAMC model across different datasets
python train.py -c blobs_eamc
python train.py -c noisymnist_eamc
python train.py -c edgemnist_eamc
python train.py -c caltech20_eamc
python train.py -c caltech7_eamc
python train.py -c noisyfashionmnist_eamc
python train.py -c edgefashionmnist_eamc
python train.py -c coil20_eamc
python train.py -c patchedmnist_eamc

# Run benchmark experiments for MIMVC model across different datasets
python train.py -c blobs_mimvc
python train.py -c noisymnist_mimvc
python train.py -c edgemnist_mimvc
python train.py -c caltech20_mimvc
python train.py -c caltech7_mimvc
python train.py -c noisyfashionmnist_mimvc
python train.py -c edgefashionmnist_mimvc
python train.py -c coil20_mimvc
python train.py -c patchedmnist_mimvc

# Run benchmark experiments for MultiVAE model across different datasets
python train.py -c blobs_mvae
python train.py -c noisymnist_mvae
python train.py -c edgemnist_mvae
python train.py -c caltech20_mvae
python train.py -c caltech7_mvae
python train.py -c noisyfashionmnist_mvae
python train.py -c edgefashionmnist_mvae
python train.py -c coil20_mvae
python train.py -c patchedmnist_mvae

# Run benchmark experiments for MvIIC model across different datasets
python train.py -c blobs_mviic
python train.py -c noisymnist_mviic
python train.py -c edgemnist_mviic
python train.py -c caltech20_mviic
python train.py -c caltech7_mviic
python train.py -c noisyfashionmnist_mviic
python train.py -c edgefashionmnist_mviic
python train.py -c coil20_mviic
python train.py -c patchedmnist_mviic

# Run benchmark experiments for MVSCN model across different datasets
python train.py -c blobs_mvscn
python train.py -c noisymnist_mvscn
python train.py -c noisyfashionmnist_mvscn
python train.py -c edgemnist_mvscn
python train.py -c edgefashionmnist_mvscn
python train.py -c coil20_mvscn
python train.py -c caltech20_mvscn
python train.py -c caltech7_mvscn
python train.py -c patchedmnist_mvscn

# Run benchmark experiments for SiMVC model across different datasets
python train.py -c blobs_simvc
python train.py -c noisymnist_simvc
python train.py -c noisyfashionmnist_simvc
python train.py -c edgemnist_simvc
python train.py -c edgefashionmnist_simvc
python train.py -c coil20_simvc
python train.py -c caltech20_simvc
python train.py -c caltech7_simvc
python train.py -c patchedmnist_simvc
