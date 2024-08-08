# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from torchvision.datasets.folder import default_loader, DatasetFolder, IMG_EXTENSIONS



class ImageFolder(DatasetFolder):

    def __init__(
        self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None,
    ):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        list_file = self.root + '.list'
        if os.path.exists(list_file):
            samples = torch.load(list_file)
        else:
            samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS, is_valid_file)
            torch.save(samples, list_file)
        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
