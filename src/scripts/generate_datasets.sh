#!/bin/bash

# This script generates datasets for the project.
# Usage: ./generate_datasets.sh

# Navigate to the src directory
cd "$(dirname "$0")/.."

# Generate datasets
python -m data.make_dataset noisymnist noisyfashionmnist edgemnist edgefashionmnist patchedmnist
python -m data.make_dataset caltech20 caltech7 coil20
