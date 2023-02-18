#!/bin/bash

#$-l rt_G.small=1
#$ -l h_rt=72:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.7/3.7.13 cuda/11.6 cudnn/8.4

~/brats/bin/python3 train_anogan.py -c configs/config.json -s