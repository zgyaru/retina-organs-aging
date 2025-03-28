#!/bin/sh
#PBS -N brain_models
#PBS -l walltime=1000:00:00

#PBS -j oe

source  ~/.bashrc
cd /share2/pub/zhangyr/zhangyr/myIdea/bioAge
conda activate web_jupyter
#python ./scripts/body_6models.py
#python ./scripts/organs_6models.py
python ./scripts/brain_6models.py
