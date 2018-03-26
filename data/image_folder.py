###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

TXT_EXTENSIONS = ['.txt']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)


def make_dataset(dir):
    image_lists = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_lists.append(path)

    return image_lists


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 image lists in: " + root + "\n"
                               "Supported image list extensions are: " +
                               ",".join(TXT_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        cur_list = self.imgs[index]
        img_list = []
        for path in cur_list[1:]:
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            img_list.append(img)
        if self.return_paths:
            return img_list, cur_list
        else:
            return img_list

    def __len__(self):
        return len(self.imgs)
