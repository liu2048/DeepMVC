#!/bin/bash

# This script runs experiments with increasing views for the project.
# Usage: ./increasing_views.sh

# Navigate to the src directory
cd "$(dirname "$0")/.."

# Run experiments with increasing views for Caltech7 dataset
# SiMVC model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_simvc
python train.py -c incviews3wb_caltech7_simvc
python train.py -c incviews4wb_caltech7_simvc
python train.py -c incviews5wb_caltech7_simvc
python train.py -c incviews6wb_caltech7_simvc

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_simvc
python train.py -c incviews3bw_caltech7_simvc
python train.py -c incviews4bw_caltech7_simvc
python train.py -c incviews5bw_caltech7_simvc
python train.py -c incviews6bw_caltech7_simvc

# CoMVC model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_comvc
python train.py -c incviews3wb_caltech7_comvc
python train.py -c incviews4wb_caltech7_comvc
python train.py -c incviews5wb_caltech7_comvc
python train.py -c incviews6wb_caltech7_comvc

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_comvc
python train.py -c incviews3bw_caltech7_comvc
python train.py -c incviews4bw_caltech7_comvc
python train.py -c incviews5bw_caltech7_comvc
python train.py -c incviews6bw_caltech7_comvc

# CoMVC without adaptive weight model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_comvcnoad
python train.py -c incviews3wb_caltech7_comvcnoad
python train.py -c incviews4wb_caltech7_comvcnoad
python train.py -c incviews5wb_caltech7_comvcnoad
python train.py -c incviews6wb_caltech7_comvcnoad

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_comvcnoad
python train.py -c incviews3bw_caltech7_comvcnoad
python train.py -c incviews4bw_caltech7_comvcnoad
python train.py -c incviews5bw_caltech7_comvcnoad
python train.py -c incviews6bw_caltech7_comvcnoad

# SAEKM model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_saekm
python train.py -c incviews3wb_caltech7_saekm
python train.py -c incviews4wb_caltech7_saekm
python train.py -c incviews5wb_caltech7_saekm
python train.py -c incviews6wb_caltech7_saekm

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_saekm
python train.py -c incviews3bw_caltech7_saekm
python train.py -c incviews4bw_caltech7_saekm
python train.py -c incviews5bw_caltech7_saekm
python train.py -c incviews6bw_caltech7_saekm

# CAEKM model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_caekm
python train.py -c incviews3wb_caltech7_caekm
python train.py -c incviews4wb_caltech7_caekm
python train.py -c incviews5wb_caltech7_caekm
python train.py -c incviews6wb_caltech7_caekm

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_caekm
python train.py -c incviews3bw_caltech7_caekm
python train.py -c incviews4bw_caltech7_caekm
python train.py -c incviews5bw_caltech7_caekm
python train.py -c incviews6bw_caltech7_caekm

# SAE model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_sae
python train.py -c incviews3wb_caltech7_sae
python train.py -c incviews4wb_caltech7_sae
python train.py -c incviews5wb_caltech7_sae
python train.py -c incviews6wb_caltech7_sae

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_sae
python train.py -c incviews3bw_caltech7_sae
python train.py -c incviews4bw_caltech7_sae
python train.py -c incviews5bw_caltech7_sae
python train.py -c incviews6bw_caltech7_sae

# CAE model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_caltech7_cae
python train.py -c incviews3wb_caltech7_cae
python train.py -c incviews4wb_caltech7_cae
python train.py -c incviews5wb_caltech7_cae
python train.py -c incviews6wb_caltech7_cae

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_caltech7_cae
python train.py -c incviews3bw_caltech7_cae
python train.py -c incviews4bw_caltech7_cae
python train.py -c incviews5bw_caltech7_cae
python train.py -c incviews6bw_caltech7_cae

# Run experiments with increasing views for PatchedMNIST dataset
# SiMVC model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_simvc
python train.py -c incviews3wb_patchedmnist_simvc
python train.py -c incviews4wb_patchedmnist_simvc
python train.py -c incviews5wb_patchedmnist_simvc
python train.py -c incviews6wb_patchedmnist_simvc
python train.py -c incviews7wb_patchedmnist_simvc
python train.py -c incviews8wb_patchedmnist_simvc
python train.py -c incviews9wb_patchedmnist_simvc
python train.py -c incviews10wb_patchedmnist_simvc
python train.py -c incviews11wb_patchedmnist_simvc
python train.py -c incviews12wb_patchedmnist_simvc

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_simvc
python train.py -c incviews3bw_patchedmnist_simvc
python train.py -c incviews4bw_patchedmnist_simvc
python train.py -c incviews5bw_patchedmnist_simvc
python train.py -c incviews6bw_patchedmnist_simvc
python train.py -c incviews7bw_patchedmnist_simvc
python train.py -c incviews8bw_patchedmnist_simvc
python train.py -c incviews9bw_patchedmnist_simvc
python train.py -c incviews10bw_patchedmnist_simvc
python train.py -c incviews11bw_patchedmnist_simvc
python train.py -c incviews12bw_patchedmnist_simvc

# CoMVC model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_comvc
python train.py -c incviews3wb_patchedmnist_comvc
python train.py -c incviews4wb_patchedmnist_comvc
python train.py -c incviews5wb_patchedmnist_comvc
python train.py -c incviews6wb_patchedmnist_comvc
python train.py -c incviews7wb_patchedmnist_comvc
python train.py -c incviews8wb_patchedmnist_comvc
python train.py -c incviews9wb_patchedmnist_comvc
python train.py -c incviews10wb_patchedmnist_comvc
python train.py -c incviews11wb_patchedmnist_comvc
python train.py -c incviews12wb_patchedmnist_comvc

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_comvc
python train.py -c incviews3bw_patchedmnist_comvc
python train.py -c incviews4bw_patchedmnist_comvc
python train.py -c incviews5bw_patchedmnist_comvc
python train.py -c incviews6bw_patchedmnist_comvc
python train.py -c incviews7bw_patchedmnist_comvc
python train.py -c incviews8bw_patchedmnist_comvc
python train.py -c incviews9bw_patchedmnist_comvc
python train.py -c incviews10bw_patchedmnist_comvc
python train.py -c incviews11bw_patchedmnist_comvc
python train.py -c incviews12bw_patchedmnist_comvc

# CoMVC without adaptive weight model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_comvcnoad
python train.py -c incviews3wb_patchedmnist_comvcnoad
python train.py -c incviews4wb_patchedmnist_comvcnoad
python train.py -c incviews5wb_patchedmnist_comvcnoad
python train.py -c incviews6wb_patchedmnist_comvcnoad
python train.py -c incviews7wb_patchedmnist_comvcnoad
python train.py -c incviews8wb_patchedmnist_comvcnoad
python train.py -c incviews9wb_patchedmnist_comvcnoad
python train.py -c incviews10wb_patchedmnist_comvcnoad
python train.py -c incviews11wb_patchedmnist_comvcnoad
python train.py -c incviews12wb_patchedmnist_comvcnoad

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_comvcnoad
python train.py -c incviews3bw_patchedmnist_comvcnoad
python train.py -c incviews4bw_patchedmnist_comvcnoad
python train.py -c incviews5bw_patchedmnist_comvcnoad
python train.py -c incviews6bw_patchedmnist_comvcnoad
python train.py -c incviews7bw_patchedmnist_comvcnoad
python train.py -c incviews8bw_patchedmnist_comvcnoad
python train.py -c incviews9bw_patchedmnist_comvcnoad
python train.py -c incviews10bw_patchedmnist_comvcnoad
python train.py -c incviews11bw_patchedmnist_comvcnoad
python train.py -c incviews12bw_patchedmnist_comvcnoad

# SAEKM model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_saekm
python train.py -c incviews3wb_patchedmnist_saekm
python train.py -c incviews4wb_patchedmnist_saekm
python train.py -c incviews5wb_patchedmnist_saekm
python train.py -c incviews6wb_patchedmnist_saekm
python train.py -c incviews7wb_patchedmnist_saekm
python train.py -c incviews8wb_patchedmnist_saekm
python train.py -c incviews9wb_patchedmnist_saekm
python train.py -c incviews10wb_patchedmnist_saekm
python train.py -c incviews11wb_patchedmnist_saekm
python train.py -c incviews12wb_patchedmnist_saekm

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_saekm
python train.py -c incviews3bw_patchedmnist_saekm
python train.py -c incviews4bw_patchedmnist_saekm
python train.py -c incviews5bw_patchedmnist_saekm
python train.py -c incviews6bw_patchedmnist_saekm
python train.py -c incviews7bw_patchedmnist_saekm
python train.py -c incviews8bw_patchedmnist_saekm
python train.py -c incviews9bw_patchedmnist_saekm
python train.py -c incviews10bw_patchedmnist_saekm
python train.py -c incviews11bw_patchedmnist_saekm
python train.py -c incviews12bw_patchedmnist_saekm

# CAEKM model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_caekm
python train.py -c incviews3wb_patchedmnist_caekm
python train.py -c incviews4wb_patchedmnist_caekm
python train.py -c incviews5wb_patchedmnist_caekm
python train.py -c incviews6wb_patchedmnist_caekm
python train.py -c incviews7wb_patchedmnist_caekm
python train.py -c incviews8wb_patchedmnist_caekm
python train.py -c incviews9wb_patchedmnist_caekm
python train.py -c incviews10wb_patchedmnist_caekm
python train.py -c incviews11wb_patchedmnist_caekm
python train.py -c incviews12wb_patchedmnist_caekm

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_caekm
python train.py -c incviews3bw_patchedmnist_caekm
python train.py -c incviews4bw_patchedmnist_caekm
python train.py -c incviews5bw_patchedmnist_caekm
python train.py -c incviews6bw_patchedmnist_caekm
python train.py -c incviews7bw_patchedmnist_caekm
python train.py -c incviews8bw_patchedmnist_caekm
python train.py -c incviews9bw_patchedmnist_caekm
python train.py -c incviews10bw_patchedmnist_caekm
python train.py -c incviews11bw_patchedmnist_caekm
python train.py -c incviews12bw_patchedmnist_caekm

# SAE model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_sae
python train.py -c incviews3wb_patchedmnist_sae
python train.py -c incviews4wb_patchedmnist_sae
python train.py -c incviews5wb_patchedmnist_sae
python train.py -c incviews6wb_patchedmnist_sae
python train.py -c incviews7wb_patchedmnist_sae
python train.py -c incviews8wb_patchedmnist_sae
python train.py -c incviews9wb_patchedmnist_sae
python train.py -c incviews10wb_patchedmnist_sae
python train.py -c incviews11wb_patchedmnist_sae
python train.py -c incviews12wb_patchedmnist_sae

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_sae
python train.py -c incviews3bw_patchedmnist_sae
python train.py -c incviews4bw_patchedmnist_sae
python train.py -c incviews5bw_patchedmnist_sae
python train.py -c incviews6bw_patchedmnist_sae
python train.py -c incviews7bw_patchedmnist_sae
python train.py -c incviews8bw_patchedmnist_sae
python train.py -c incviews9bw_patchedmnist_sae
python train.py -c incviews10bw_patchedmnist_sae
python train.py -c incviews11bw_patchedmnist_sae
python train.py -c incviews12bw_patchedmnist_sae

# CAE model
# incviews2wb: increasing views from worst to best
python train.py -c incviews2wb_patchedmnist_cae
python train.py -c incviews3wb_patchedmnist_cae
python train.py -c incviews4wb_patchedmnist_cae
python train.py -c incviews5wb_patchedmnist_cae
python train.py -c incviews6wb_patchedmnist_cae
python train.py -c incviews7wb_patchedmnist_cae
python train.py -c incviews8wb_patchedmnist_cae
python train.py -c incviews9wb_patchedmnist_cae
python train.py -c incviews10wb_patchedmnist_cae
python train.py -c incviews11wb_patchedmnist_cae
python train.py -c incviews12wb_patchedmnist_cae

# incviews2bw: increasing views from best to worst
python train.py -c incviews2bw_patchedmnist_cae
python train.py -c incviews3bw_patchedmnist_cae
python train.py -c incviews4bw_patchedmnist_cae
python train.py -c incviews5bw_patchedmnist_cae
python train.py -c incviews6bw_patchedmnist_cae
python train.py -c incviews7bw_patchedmnist_cae
python train.py -c incviews8bw_patchedmnist_cae
python train.py -c incviews9bw_patchedmnist_cae
python train.py -c incviews10bw_patchedmnist_cae
python train.py -c incviews11bw_patchedmnist_cae
python train.py -c incviews12bw_patchedmnist_cae
