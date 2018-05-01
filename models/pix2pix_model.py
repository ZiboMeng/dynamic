import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
#from networks import LSTMBlock
import torch.nn as nn


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.exp_num = opt.exp_num
        self.label = self.exp_num * [0]
        self.isTrain = opt.isTrain
        self.cur = []
        self.next = []
        self.num_lm = opt.lm_num

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, self.exp_num, self.num_lm,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_init_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.len = len(input) - 1
        self.real_A = [0] * self.len
        self.real_B = [0] * self.len
        self.image_paths = [0] * self.len
        self.c, self.h, self.w = input[1]['A'].size(1), input[1]['A'].size(2), input[1]['A'].size(3)
        if self.num_lm:
            self.cur = [0] * self.len
            self.next = [0] * self.len
            self.pred_landmarks = [0] * self.len
        for i in range(1, self.len + 1):
            self.real_A[i - 1] = input[i]['A' if AtoB else 'B']
            self.real_B[i - 1] = input[i]['B' if AtoB else 'A']
            self.image_paths[i - 1] = input[i]['A_paths' if AtoB else 'B_paths']
            if self.num_lm:
                self.cur[i - 1] = input[i]['A_lm']
                self.next[i - 1] = input[i]['B_lm']
        #print(self.real_A)
        self.real_A = torch.cat(self.real_A).view(self.len, self.c, self.h, self.w)
        self.real_B = torch.cat(self.real_B).view(self.len, self.c, self.h, self.w)
        self.fake_B = [0] * (self.len + 1)
        self.fake_B[0] = Variable(self.real_A[0].view(1, self.c, self.h, self.w))
        self.index = 0

        if len(self.gpu_ids) > 0:
            self.real_A = self.real_A.cuda(self.gpu_ids[0], async=True)
            self.real_B = self.real_B.cuda(self.gpu_ids[0], async=True)
            self.fake_B[0] = self.fake_B[0].cuda(self.gpu_ids[0], async=True)
            self.label = self.label.cuda(self.gpu_ids[0], async=True)


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        #print(input_A)
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_seq(self, input):
        # set label vector
        self.len = len(input) - 1
        label_template = [0] * self.exp_num
        label_template[int(input[0])] = 1
        self.label = [Variable(torch.FloatTensor(label_template)) for _ in range(self.len)]
        self.label = torch.cat(self.label).view(self.len, self.exp_num, 1, 1)
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = [0] * (self.len)
        input_B = [0] * (self.len)
        self.image_paths = [0] * (self.len)
        c, h, w = input[1]['A'].size(1), input[1]['A'].size(2), input[1]['A'].size(3)
        for i in range(1, self.len + 1):
            input_A[i-1] = input[i]['A' if AtoB else 'B']
            input_B[i-1] = input[i]['B' if AtoB else 'A']
            self.image_paths = input[i]['A_paths' if AtoB else 'B_paths']
        #print("Length of A is {} v.s. length of data is {}".format(len(input_A), self.len))
        #print(input_A)
        #print("c h w are {} {} {}".format(c, h, w))
        input_A = torch.cat(input_A).view(len(input_A),c,h,w)
        input_B = torch.cat(input_B).view(len(input_B),c,h,w)
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            self.label = self.label.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B


    def set_test_input(self, input, frames):
        self.frames = frames
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True) 
        self.fake_B = [0] * self.frames
        self.fake_B[0] = Variable(input_A)
        self.image_paths = input['A_paths']

    def set_label(self, input):
        self.label = self.exp_num * [0]
        self.label[input] = 1
        self.label = Variable(torch.FloatTensor(self.label), requires_grad=False)#.cuda()
        self.label = self.label.view(1, self.exp_num, 1,1)
        if len(self.gpu_ids) > 0:
            self.label = self.label.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    def forward_single_with_landmarks(self):
        #self.real_A[self.index] = Variable(self.input_A)
        for i in range(self.len):
            # Set landmarks
            #print("set landmarks")
            print(self.cur)
            self.set_landmarks_lstm(self.cur[i], self.next[i])
            #print(self.fake_B[i])
            # We do not want the input to be treated as a response map
            # Thus detach at the input
            self.fake_B[i + 1], self.pred_landmarks[i] = self.netG(self.fake_B[i].detach())

        self.fake_B = torch.cat(self.fake_B[1:]).view(self.len, self.c, self.h, self.w)
        self.pred_landmarks = torch.cat(self.pred_landmarks).view(self.len, -1, 1, 1)
        self.real_A = Variable(self.real_A)
        self.real_B = Variable(self.real_B)


    def forward_single(self):
        #self.real_A[self.index] = Variable(self.input_A)
        for i in range(self.len):
            #print(self.fake_B[i])
            # We do not want the input to be treated as a response map
            # Thus detach at the input
            self.fake_B[i + 1] = self.netG(self.fake_B[i].detach())
            
        self.fake_B = torch.cat(self.fake_B[1:]).view(self.len, self.c, self.h, self.w)
        self.real_A = Variable(self.real_A)
        self.real_B = Variable(self.real_B)
        #print(self.fake_B)
        #print(self.real_A)
        #self.fake_input = self.fake_B
        #self.real_B[self.index] = Variable(self.input_B)
        #self.index += 1

    # no backprop gradients
    def test(self):
        for i in range(self.frames - 1):
            #self.real_A = Variable(self.real_A[-1])
            self.fake_B[i + 1] = self.netG(self.fake_B[i])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_single_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #print(self.real_A[self.index - 1])
        #self.real_A = torch.cat(real_A).view(self.len,self.c,self.h,self.w)
        #self.fake_B = torch.cat(fake_B[1:]).view(self.len, self.c, self.h, self.w)
        #print(torch.cat(self.real_A).view(self.len,self.c,self.h,self.w))
        #print(torch.cat(self.fake_B[1:]).view(self.len,self.c,self.h,self.w))
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #for i in range(1,3):
        #    fake_AB = torch.cat((fake_AB, torch.cat((self.real_A, torch.cat((self.fake_B[i:], self.fake_B[0:i]), 0)), 1)), 0)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        #self.real_B = torch.cat(real_B).view(self.len, self.c, self.h, self.w)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward(retain_graph=True)


    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward(retain_graph=True)


    def backward_single_with_landmarks_G(self):
        # First, G(A) should fake the discriminator
        #self.real_A = torch.cat(self.real_A).view(self.len,self.c,self.h,self.w)
        #self.fake_B = torch.cat(self.fake_B[1:]).view(self.len, self.c, self.h, self.w)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #print(fake_AB)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        #self.loss_G_MSE = self.criterionMSE(self.pred_landmarks, torch.cat(self.next).view(self.len, -1, 1, 1))

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #self.loss_G = self.loss_G_GAN

        self.loss_G.backward(retain_graph=True)


    def backward_single_G(self):
        # First, G(A) should fake the discriminator
        #self.real_A = torch.cat(self.real_A).view(self.len,self.c,self.h,self.w)
        #self.fake_B = torch.cat(self.fake_B[1:]).view(self.len, self.c, self.h, self.w)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #print(fake_AB)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #self.loss_G = self.loss_G_GAN

        self.loss_G.backward(retain_graph=True)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters_single(self):
        self.optimizer_D.zero_grad()
        self.backward_single_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_single_G()
        self.optimizer_G.step()

    def backward_D_seq(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # fake_AB = [0] * self.len
        #for i in range(self.len):
        #fake_AB[i] = self.fake_AB_pool.query(torch.cat((self.real_A[i], self.fake_B[i]), 1).data)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #print(fake_AB)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # real_AB = [0] * self.len
        # for i in range(self.len):
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward(retain_graph=True)

    def backward_G_seq(self):
        # First, G(A) should fake the discriminator
        # fake_AB = [0] * self.len
        # for i in range(self.len):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters_seq(self):
        self.optimizer_D.zero_grad()
        self.backward_D_seq()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G_seq()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_single_visuals(self):
        real_A = util.tensor2im(self.real_A[self.index - 1].data)
        fake_B = util.tensor2im(self.fake_B[self.index].data)
        real_B = util.tensor2im(self.real_B[self.index - 1].data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


    def get_current_visuals_seq(self):
        diction = [{}] * self.len
        c, h, w = self.real_A.size(1), self.real_A.size(2), self.real_A.size(3)
        #print(self.real_A[0])
        for i in range(self.len):
            real_A = util.tensor2im(self.real_A[i].view(1,c,h,w).data)
            #print(self.real_A[i])
            fake_B = util.tensor2im(self.fake_B[i].view(1,c,h,w).data)
            real_B = util.tensor2im(self.real_B[i].view(1,c,h,w).data)
            diction[i] = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        return diction

    def get_current_test_visuals(self):
        diction = [0] * self.frames
        for i, img in enumerate(self.fake_B):
            diction[i] = util.tensor2im(img.data)
        return diction


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def init_lstm(self):
        for layer in self.netG.modules():
            if isinstance(layer, networks.LSTMBlock):
                layer.init_grad()
                layer.init_hidden()
                layer.set_label(self.label)

    def set_landmarks_lstm(self, cur, next):
        for layer in self.netG.modules():
            if isinstance(layer, networks.LSTMBlock):
                layer.set_landmarks(cur, next)
