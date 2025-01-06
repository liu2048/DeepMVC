#!/bin/bash

# This script runs ablation study experiments for the project.
# Usage: ./ablation.sh

# Navigate to the src directory
cd "$(dirname "$0")/.."

# Run ablation study experiments for fusion methods on NoisyMNIST dataset
python train.py -c ablfusion_noisymnist_eamc
python train.py -c ablfusion_noisymnist_simvc
python train.py -c ablfusion_noisymnist_comvc
python train.py -c ablfusion_noisymnist_sae
python train.py -c ablfusion_noisymnist_cae

# Run ablation study experiments for fusion methods on Caltech7 dataset
python train.py -c ablfusion_caltech7_eamc
python train.py -c ablfusion_caltech7_simvc
python train.py -c ablfusion_caltech7_comvc
python train.py -c ablfusion_caltech7_sae
python train.py -c ablfusion_caltech7_cae

# Run ablation study experiments for MV-SSL methods on NoisyMNIST dataset
python train.py -c ablmvssl_noisymnist_eamc
python train.py -c ablmvssl_caltech7_eamc
python train.py -c ablmvssl_noisymnist_mvscn
python train.py -c ablmvssl_caltech7_mvscn
python train.py -c ablmvssl_noisymnist_mvae
python train.py -c ablmvssl_caltech7_mvae
python train.py -c ablmvssl_noisymnist_mviic
python train.py -c ablmvssl_caltech7_mviic

# Run ablation study experiments for SV-SSL methods on NoisyMNIST dataset
python train.py -c ablsvssl_noisymnist_mvscn
python train.py -c ablsvssl_noisymnist_caekm
python train.py -c ablsvssl_noisymnist_saekm
python train.py -c ablsvssl_caltech7_mvscn
python train.py -c ablsvssl_caltech7_caekm
python train.py -c ablsvssl_caltech7_saekm
python train.py -c ablsvssl_noisymnist_dmsc
python train.py -c ablsvssl_caltech7_dmsc
