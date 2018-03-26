import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

dir = "/home/zibo/Data/Database/CKPLUS/cropImg_128x128/"

paths = make_dataset(dir)

