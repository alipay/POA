# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
semi-supervised learning main file.
Most are copyed from BEiT library:
https://github.com/microsoft/unilm/tree/master/beit
"""

import argparse
import datetime
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json
import os
import math
import sys
from pathlib import Path
from dataset import ImageFolder 


root = '/gruntdata/rs_nas/workspace/qichu.zyy/Dataset/imagenet'
train_root = os.path.join(root, 'train')

target_list_path = 'poa/eval/semi_supervised_learning/subset/10percent.txt'
percent_name = Path(target_list_path).stem
print(f'create image list for train data: {percent_name}')
f = open(target_list_path)
lines = f.readlines()
target_list = []
for line2 in lines:
    target_list.append(line2[:-1])
dataset = ImageFolder(train_root, transform=None, class_num=1000, target_list=target_list, percent_name=percent_name)

target_list_path = 'poa/eval/semi_supervised_learning/subset/1percent.txt'
percent_name = Path(target_list_path).stem
print(f'create image list for train data: {percent_name}')
f = open(target_list_path)
lines = f.readlines()
target_list = []
for line2 in lines:
    target_list.append(line2[:-1])
dataset = ImageFolder(train_root, transform=None, class_num=1000, target_list=target_list, percent_name=percent_name)

print(f'create image list for valdata')
val_root = os.path.join(root, 'val')
target_list = []
dataset = ImageFolder(train_root, transform=None, class_num=1000, target_list=target_list, percent_name='')
