from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)
	self.frames = opt.frames

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
	self.fake_B = [0] * self.frames
        self.fake_B[0] = Variable(input_A)
        self.image_paths = input['A_paths']

    def test(self):
	for i in range(self.frames - 1):
        	#self.real_A = Variable(self.real_A[-1])
        	self.fake_B[i + 1] = self.netG(self.real_B[i])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
	OrderedDict = [0] * self.frames
	for i, img in enumerate(self.fake_B):
            OrderedDict[i] = util.tensor2im(img)
	return OrderedDict
