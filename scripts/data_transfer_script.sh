#!/bin/bash

# Rsync command with a smooth progress bar using 'pv'
rsync -azh --info=progress2 strivaud@titanic.saclay.inria.fr:/home/tau/strivaud/gromo/logs/mlp_run /Users/strivaud/PycharmProjects/gromo/logs