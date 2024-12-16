#!/bin/bash

conda init
conda create -n gromo python=3.12
conda activate gromo

# install dependencies
pip install -e .