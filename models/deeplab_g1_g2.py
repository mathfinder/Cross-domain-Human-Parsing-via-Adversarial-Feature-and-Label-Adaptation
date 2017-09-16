import os
import torch
from collections import OrderedDict
from .base_model import BaseModel
import networks
import itertools
from torch.autograd import Variable

def get_parameters(model, parameter_name):
    for name, param in model.named_parameters():
        if name in [parameter_name]:
            return param

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def define_D(which_netD, input_nc):
    if which_netD == 'FFC':
        return networks.FFC()
    elif which_netD == 'NoBNMultPathdilationNet':
        return networks.NoBNMultPathdilationNet()
    elif which_netD == 'SinglePathdilationSingleOutputNet':
        return networks.SinglePathdilationSingleOutputNet()
    elif which_netD == 'SinglePathdilationMultOutputNet':
        return networks.SinglePathdilationMultOutputNet()
    elif which_netD == 'NoBNSinglePathdilationMultOutputNet':
        return networks.NoBNSinglePathdilationMultOutputNet(input_nc)
    elif which_netD == 'RandomMultPathdilationNet':
        return networks.RandomMultPathdilationNet()
    elif which_netD == 'lsgan_D':
        return networks.lsgan_D()
    elif which_netD == 'lsganMultOutput_D':
        return networks.lsganMultOutput_D(input_nc)

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def constant_learning_rate(optimizer, init_lr):
    """constant"""
    for param_group in optimizer.param_groups:
        print param_group['lr']
        #param_group['lr'] = init_lr

class deeplabG1G2(BaseModel):
    def name(self):
        return 'deeplabG1G2'

    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.if_adv_train = args['if_adv_train']
        self.Iter = 0
        self.interval_g2 = args['interval_g2']
        self.interval_d2 = args['interval_d2']
        self.nb = args['batch_size']
        sizeH, sizeW = args['fineSizeH'], args['fineSizeW']

        self.input_A = self.Tensor(self.nb, args['input_nc'], sizeH, sizeW)
        self.input_B = self.Tensor(self.nb, args['input_nc'], sizeH, sizeW)
        self.input_A_label = torch.cuda.LongTensor(self.nb, 1, sizeH, sizeW)
        self.input_label_onehot =  self.Tensor(self.nb, args['label_nums'], sizeH, sizeW)
        self.loss_G = Variable()
        self.loss_D = Variable()

        self.netG1 = networks.netG().cuda(device_id=args['device_ids'][0])
        self.netD1 = define_D(args['net_d1'],512).cuda(device_id=args['device_ids'][0])
        self.netD2 = define_D(args['net_d2'],args['label_nums']).cuda(device_id=args['device_ids'][0])

        self.deeplabPart1 = networks.DeeplabPool1().cuda(device_id=args['device_ids'][0])
        self.deeplabPart2 = networks.DeeplabPool12Pool5().cuda(device_id=args['device_ids'][0])
        self.deeplabPart3 = networks.DeeplabPool52Fc8_interp(output_nc=args['label_nums']).cuda(device_id=args['device_ids'][0])

        # define loss functions
        self.criterionCE = torch.nn.CrossEntropyLoss(size_average=False)
        self.criterionAdv = networks.Advloss(use_lsgan=args['use_lsgan'], tensor=self.Tensor)


        if not args['resume']:
            #initialize networks
            self.netG1.apply(weights_init)
            self.netD1.apply(weights_init)
            self.netD2.apply(weights_init)
            pretrained_dict = torch.load(args['weigths_pool'] + '/' + args['pretrain_model'])
            self.deeplabPart1.weights_init(pretrained_dict=pretrained_dict)
            self.deeplabPart2.weights_init(pretrained_dict=pretrained_dict)
            self.deeplabPart3.weights_init(pretrained_dict=pretrained_dict)

        # initialize optimizers
        self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                            lr=args['lr_g1'], betas=(args['beta1'], 0.999))
        self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(),
                                            lr=args['lr_g1'], betas=(args['beta1'], 0.999))

        self.optimizer_G2 = torch.optim.Adam([
            {'params': self.deeplabPart1.parameters()},
            {'params': self.deeplabPart2.parameters()},
            {'params': self.deeplabPart3.parameters()}],
                                            lr=args['lr_g2'], betas=(args['beta1'], 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                            lr=args['lr_g2'], betas=(args['beta1'], 0.999))

        ignored_params = list(map(id, self.deeplabPart3.fc8_1.parameters()))
        ignored_params.extend(list(map(id, self.deeplabPart3.fc8_2.parameters())))
        ignored_params.extend(list(map(id, self.deeplabPart3.fc8_3.parameters())))
        ignored_params.extend(list(map(id, self.deeplabPart3.fc8_4.parameters())))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.deeplabPart3.parameters())
        base_params = base_params +  filter(lambda p: True, self.deeplabPart1.parameters())
        base_params = base_params +  filter(lambda p: True, self.deeplabPart2.parameters())

        deeplab_params = [{'params': base_params},
            {'params': get_parameters(self.deeplabPart3.fc8_1, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_2, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_3, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_4, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_1, 'bias'), 'lr': args['l_rate'] * 20},
            {'params': get_parameters(self.deeplabPart3.fc8_2, 'bias'), 'lr': args['l_rate'] * 20},
            {'params': get_parameters(self.deeplabPart3.fc8_3, 'bias'), 'lr': args['l_rate'] * 20},
            {'params': get_parameters(self.deeplabPart3.fc8_4, 'bias'), 'lr': args['l_rate'] * 20},
        ]


        self.optimizer_P = torch.optim.SGD(deeplab_params, lr=args['l_rate'], momentum=0.9, weight_decay=5e-4)

        self.optimizer_R = torch.optim.SGD(deeplab_params, lr=args['l_rate'], momentum=0.9, weight_decay=5e-4)


        print('---------- Networks initialized -------------')
        networks.print_network(self.netG1)
        networks.print_network(self.netD1)
        networks.print_network(self.netD2)
        networks.print_network(self.deeplabPart1)
        networks.print_network(self.deeplabPart2)
        networks.print_network(self.deeplabPart3)
        print('-----------------------------------------------')


    def set_input(self, input):
        self.input = input
        input_A = input['A']
        input_A_label = input['A_label']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_label.resize_(input_A_label.size()).copy_(input_A_label)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        if input.has_key('label_onehot'):
            input_label_onehot = input['label_onehot']
            self.input_label_onehot.resize_(input_label_onehot.size()).copy_(input_label_onehot)

    def forward(self):
        self.A = Variable(self.input_A)
        self.A_label = Variable(self.input_A_label)
        self.label_onehot = Variable(self.input_label_onehot)
        self.B = Variable(self.input_B)

    # used in test time, no backprop
    def test(self, input):
        self.input_A.resize_(input.size()).copy_(input)
        self.A = Variable(self.input_A)
        self.output = self.deeplabPart3(self.deeplabPart2(self.deeplabPart1(self.A)))
        return self.output

    def get_image_paths(self):
        pass

    def backward_P(self):
        # Maintain pool5_B in this status
        self.pool5_B = self.deeplabPart2(self.deeplabPart1(self.B))
        self.pool5_B_for_d1 = Variable(self.pool5_B.data)

        self.pool1_A = self.deeplabPart1(self.A)
        self.pool5_A = self.deeplabPart2(self.pool1_A)
        self.predic_A = self.deeplabPart3(self.pool5_A)
        self.output = Variable(self.predic_A.data)

        self.loss_P = self.criterionCE(self.predic_A, self.A_label) / self.nb
        self.loss_P.backward()

        self.pool1_A = Variable(self.pool1_A.data)
        self.pool5_A = Variable(self.pool5_A.data)


    def backward_G1(self):
        self.pool5_A = self.pool5_A + self.netG1(self.pool1_A)
        pred_fake = self.netD1.forward(self.pool5_A)

        self.loss_G1 = self.criterionAdv(pred_fake, True)
        self.loss_G1.backward()

        self.pool5_A = Variable(self.pool5_A.data)

    def backward_D1(self):
        pred_real = self.netD1.forward(self.pool5_B_for_d1)
        loss_D1_real = self.criterionAdv(pred_real, True)

        pred_fake = self.netD1.forward(self.pool5_A)
        loss_D1_fake = self.criterionAdv(pred_fake, False)

        self.loss_D1 = (loss_D1_real + loss_D1_fake) * 0.5
        self.loss_D1.backward()

    def backward_G2(self):
        self.predic_B = self.deeplabPart3(self.pool5_B)
        pred_fake = self.netD2.forward(self.predic_B)

        self.loss_G2 = self.criterionAdv(pred_fake, True)
        self.loss_G2.backward()

    def backward_D2(self):
        #self.label_onehot = Variable(self.label_onehot.data)
        pred_real = self.netD2.forward(self.label_onehot)
        loss_D2_real = self.criterionAdv(pred_real, True)

        self.predic_B = Variable(self.predic_B.data)
        pred_fake = self.netD2.forward(self.predic_B)
        loss_D2_fake = self.criterionAdv(pred_fake, False)

        self.loss_D2 = (loss_D2_real + loss_D2_fake) * 0.5

        self.loss_D2.backward()

    def backward_R(self):
        pool1 = self.deeplabPart1(self.A)
        self.predic_A_R = self.deeplabPart3(self.deeplabPart2(pool1) + self.netG1(pool1))
        self.loss_R = self.criterionCE(self.predic_A_R, self.A_label) / self.nb

        self.loss_R.backward()

    def optimize_parameters(self):
        self.Iter += 1
        # forward
        self.forward()
        # deeplab
        self.optimizer_P.zero_grad()
        self.backward_P()
        self.optimizer_P.step()

        # G1
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()
        # D1
        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()
        if self.Iter % self.interval_g2 == 0 and self.if_adv_train:
            # G2
            self.optimizer_G2.zero_grad()
            self.backward_G2()
            self.optimizer_G2.step()
        if self.Iter % self.interval_d2 == 0 and self.if_adv_train:
            # D2
            self.optimizer_D2.zero_grad()
            self.backward_D2()
            self.optimizer_D2.step()

        # Refine
        self.optimizer_R.zero_grad()
        self.backward_R()
        self.optimizer_R.step()


    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}


    def save(self, model_name, Iter=None, epoch=None, acc=[]):
        save_filename = '%s_model.pth' % (model_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save({
            'name':self.name(),
            'Iter': Iter,
            'epoch': epoch,
            'acc':acc,
            'state_dict_netG1': self.netG1.state_dict(),
            'state_dict_netD1': self.netD1.state_dict(),
            'state_dict_netD2': self.netD2.state_dict(),
            'state_dict_deeplabPart1': self.deeplabPart1.state_dict(),
            'state_dict_deeplabPart2':self.deeplabPart2.state_dict(),
            'state_dict_deeplabPart3': self.deeplabPart3.state_dict(),
            'optimizer_P':self.optimizer_P.state_dict(),
            'optimizer_R': self.optimizer_R.state_dict(),
            'optimizer_G1': self.optimizer_G1.state_dict(),
            'optimizer_D1': self.optimizer_D1.state_dict(),
            'optimizer_G2': self.optimizer_G2.state_dict(),
            'optimizer_D2': self.optimizer_D2.state_dict(),
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.netG1.load_state_dict(checkpoint['state_dict_netG1'])
        self.netD1.load_state_dict(checkpoint['state_dict_netD1'])
        self.netD2.load_state_dict(checkpoint['state_dict_netD2'])
        self.deeplabPart1.load_state_dict(checkpoint['state_dict_deeplabPart1'])
        self.deeplabPart2.load_state_dict(checkpoint['state_dict_deeplabPart2'])
        self.deeplabPart3.load_state_dict(checkpoint['state_dict_deeplabPart3'])

        self.optimizer_P.load_state_dict(checkpoint['optimizer_P'])
        self.optimizer_G1.load_state_dict(checkpoint['optimizer_G1'])
        self.optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
        self.optimizer_G2.load_state_dict(checkpoint['optimizer_G2'])
        self.optimizer_D2.load_state_dict(checkpoint['optimizer_D2'])
        self.optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        for k,v in checkpoint['acc'].items():
            print('=================================================')
            if k == 'acc_Ori_on_B':
                best_f1 = v['avg_f1score']
            print('accuracy: {0:.4f}\t'
                  'fg_accuracy: {1:.4f}\t'
                  'avg_precision: {2:.4f}\t'
                  'avg_recall: {3:.4f}\t'
                  'avg_f1score: {4:.4f}\t'
                  .format(v['accuracy'],v['fg_accuracy'],v['avg_precision'], v['avg_recall'], v['avg_f1score']))
            print('=================================================')

        return checkpoint['Iter'], checkpoint['epoch'], best_f1

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for param_group in self.optimizer_D1.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        for param_group in self.optimizer_G1.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        for param_group in self.optimizer_G2.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    def train(self):
        self.deeplabPart1.train()
        self.deeplabPart2.train()
        self.deeplabPart3.train()
        self.netG1.train()
        self.netD1.train()
        self.netD2.train()

    def eval(self):
        self.deeplabPart1.eval()
        self.deeplabPart2.eval()
        self.deeplabPart3.eval()
        self.netG1.train()
        self.netD1.train()
        self.netD2.train()