import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import util.util as util


class LandmarkDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        print(AB_path)
        assert(os.path.exists(AB_path)), "[%s] does not exist." % AB_path
        with open(AB_path) as AB_path:
            AB_path = AB_path.readlines()
            AB_path = [x.strip().split() for x in AB_path]
        print('im here')
        img_list = []
        img_list.append(int(AB_path[0][0]))
        flip = (not self.opt.no_flip) and random.random() < 0.5
        for x in range(1, len(AB_path) - 2):
            A_path = AB_path[x][0];
            A_lm = [float(m) for m in AB_path[x][1:]]
            size = int(len(A_lm))
            half = int(size/2)
            #print("[aligned_dataset line 30] Read A image is <%s>" % A_path)
            A = Image.open(A_path).convert('RGB')
            w_orig, h_orig = A.size
            A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A = transforms.ToTensor()(A)

            B_path = AB_path[x + 1][0]
            B_lm = [float(m) for m in AB_path[x+1][1:]]
            #print("[aligned_dataset line 36] Read B image is <%s>" % B_path)
            B = Image.open(B_path).convert('RGB')
            B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = transforms.ToTensor()(B)

            w = A.size(2)
            h = A.size(1)
            A_lm = util.rescale_landmarks(w_orig, w, h_orig, h, A_lm)
            B_lm = util.rescale_landmarks(w_orig, w, h_orig, h, B_lm)
            if x == 1:
                w_offset = random.randint(0, max(0, w - self.opt.fineSize))
                h_offset = random.randint(0, max(0, h - self.opt.fineSize))

            A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
            B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
            for i in range(half,size):
                A_lm[i] = A_lm[i] - h_offset
                B_lm[i] = B_lm[i] - h_offset
            for i in range(0,half):
                A_lm[i] = A_lm[i] - w_offset
                B_lm[i] = B_lm[i] - w_offset

            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

            if self.opt.which_direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            if flip:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)
                for i in range(0,half):
                    A_lm[i] = w - A_lm[i] - 1
                    B_lm[i] = w - B_lm[i] - 1

            if input_nc == 1:  # RGB to gray
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)

            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)

            #print("A")
            #print(A_lm)
            for i in range(size):
                A_lm[i] /= self.opt.fineSize
                B_lm[i] /= self.opt.fineSize

            #print("After A")
            #print(A_lm)
            img_list.append({'A': A, 'B': B, 'A_lm': A_lm, 'B_lm': B_lm,
                'A_paths': A_path, 'B_paths': B_path})

        return img_list

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'LandmarkDataset'
