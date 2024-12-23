import copy
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from ..losses.losses import *
from ..losses.lnconsistent_labels_loss import *
# from mixstyle import MixStyle, activate_mixstyle, deactivate_mixstyle
from models.unet_model.SAW import SAW
from models.unet_model.SAN import SAN

###############################################################################
# Helper Functions
###############################################################################

def define_loss(type='CrossEntropyLoss', alpha=0, class_num=2):
    if type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif type == 'BCW':
        criterion = BCWLoss(class_num=class_num+1)
    elif type == 'Focal':
        criterion = FocalLoss(alpha=alpha)
    elif type == 'Inconsistent_Labels_loss':
        marginal_loss = mar_loss()
        exclusion_loss = exc_loss()
        criterion = [marginal_loss,exclusion_loss]
    else:
        raise NotImplementedError('loss type [%s] is not recoginized' % type)
    return criterion


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], main_device=0):
    # print(gpu_ids)
    if len(gpu_ids) > 1:
        assert (torch.cuda.is_available()), 'no available gpu devices'
        gpu_ids.pop(main_device)
        try:
            net = torch.nn.DataParallel(net, device_ids=[main_device] + gpu_ids)
            net.to(main_device)
        except:
            net = torch.nn.DataParallel(net)
            net.cuda()
    if len(gpu_ids) != 0:
        net.cuda()
    if init_type is not None:
        init_weights(net, init_type, gain=init_gain)
    return net


def define_net(input_nc, output_nc, net='unet', init_type='xavier_uniform', init_gain=1.0, gpu_ids=[],
               main_device=0, SAN_SAW=False, san_lsit=[0,1],base_ch=32):
    if net != 'unet':
        init_type = None
    if net == 'unet' and SAN_SAW == True:
        net = Unet(input_nc=input_nc, output_nc=output_nc, base_ch=base_ch, san_lsit=san_lsit, selected_classes=list(range(1,output_nc+1)))
    elif net == 'unet':
        net = Unet(input_nc=input_nc, output_nc=output_nc, base_ch=32)
    else:
        net = Unet(input_nc=input_nc, output_nc=output_nc, base_ch=32)
        # raise NotImplementedError('model name [%s] is not recoginized'%net)
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=copy.deepcopy(gpu_ids),
                    main_device=main_device)


# class Unet(nn.Module):
#     def __init__(self, input_nc=1, output_nc=2):
#         super(Unet, self).__init__()
#         num_feat = [32, 64, 128, 256, 512]
#         # unet structure
#         # innermost
#         unet_block = UnetSkipConnectionBlock(num_feat[3], num_feat[4], innermost=True)
#
#         unet_block = UnetSkipConnectionBlock(num_feat[2], num_feat[3], submodule=unet_block)
#         unet_block = UnetSkipConnectionBlock(num_feat[1], num_feat[2], submodule=unet_block)
#         unet_block = UnetSkipConnectionBlock(num_feat[0], num_feat[1], submodule=unet_block)
#         # outermost
#         unet_block = UnetSkipConnectionBlock(output_nc, num_feat[0], input_nc=input_nc,
#                                              submodule=unet_block, outermost=True)
#         self.model = unet_block
#
#     def forward(self, input):
#         return self.model(input)
# class UnetSkipConnectionBlock(nn.Module):
#     # define the submodule with skip connection
#     # ----------------------------------------#
#     # /-downsampling-/submodule/-upsampling-/ #
#     # ----------------------------------------#
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False):
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if input_nc is None:
#             input_nc = outer_nc
#         down_maxpool = nn.MaxPool2d(kernel_size=2)
#         conv1 = self.conv_3x3(input_nc, inner_nc)
#         conv2 = self.conv_3x3(inner_nc * 2, inner_nc)
#         up_conv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                      kernel_size=2,
#                                      stride=2)
#         conv_out = nn.Conv2d(inner_nc, outer_nc,
#                              kernel_size=1)
#         if outermost:
#             down = [conv1]
#             up = [conv2, conv_out]
#         elif innermost:
#             down = [down_maxpool, conv1]
#             up = [up_conv]
#         else:
#             down = [down_maxpool, conv1]
#             up = [conv2, up_conv]
#
#         model = down + up if innermost else down + [submodule] + up
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         # print('input shape:',input.size())
#         output = self.model(input)
#         # print('output shape:',output.size())
#         if not self.outermost:
#             output = torch.cat([input, output], 1)
#         return output
#
#     def conv_3x3(self, input_nc, output_nc):
#         conv_block1 = nn.Sequential(nn.Conv2d(input_nc, output_nc,
#                                               kernel_size=3,
#                                               stride=1,
#                                               padding=1),
#                                     nn.BatchNorm2d(output_nc),
#                                     nn.ReLU(inplace=True))
#         conv_block2 = nn.Sequential(nn.Conv2d(output_nc, output_nc,
#                                               kernel_size=3,
#                                               stride=1,
#                                               padding=1),
#                                     nn.BatchNorm2d(output_nc),
#                                     nn.ReLU(inplace=True))
#         conv_block = nn.Sequential(conv_block1, conv_block2)
#         return conv_block

# ConvBlock
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y
# Encoding block
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        return y, y_conv
# Decoding block
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.up = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        return y, y_conv

def concatenate(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    # diffX = x2.size()[4] - x1.size()[4]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2))
                    # diffZ // 2, diffZ - diffZ//2))
    y = torch.cat([x2, x1], dim=1)
    return y

class sub_encoder(nn.Module):
    def __init__(self, in_ch, base_ch, mixstyle_layers, mixstyle_p, mixstyle_alpha):
        super(sub_encoder, self).__init__()
        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch*2)
        self.enc3 = enc_block(base_ch*2, base_ch*4)
        self.enc4 = enc_block(base_ch*4, base_ch*8)
        # if mixstyle_layers != None:
        #     self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='random')
        #     print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

    def forward(self, x, nodes_style):
        y, enc_conv_1 = self.enc1(x)
        if 'layer1' in self.mixstyle_layers and nodes_style!=None:
            y = self.mixstyle(y, nodes_style[0])
        y, enc_conv_2 = self.enc2(y)
        if 'layer2' in self.mixstyle_layers and nodes_style!=None:
            y = self.mixstyle(y, nodes_style[1])
        y, enc_conv_3 = self.enc3(y)
        if 'layer3' in self.mixstyle_layers and nodes_style!=None:
            y = self.mixstyle(y, nodes_style[2])
        y, enc_conv_4 = self.enc4(y)
        return y, enc_conv_1, enc_conv_2, enc_conv_3, enc_conv_4


class sub_decoder(nn.Module):
    def __init__(self, base_ch, cls_num, selected_classes,san_list = [0,1]):
        super(sub_decoder, self).__init__()
        self.dec1 = dec_block(base_ch*8,  base_ch*8)
        self.dec2 = dec_block(base_ch*16, base_ch*4)
        self.dec3 = dec_block(base_ch*8,  base_ch*2)
        self.dec4 = dec_block(base_ch*4,  base_ch)
        self.lastconv = double_conv(base_ch*2, base_ch)
        self.outconv = nn.Conv2d(base_ch, cls_num+1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.clssifier1 = nn.Sequential(
            nn.Conv2d(base_ch*8, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.clssifier1_up = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.clssifier2 = nn.Sequential(
            nn.Conv2d(base_ch*4, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.clssifier2_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier3 = nn.Sequential(
            nn.Conv2d(base_ch*2, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.clssifier3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier4 = nn.Sequential(
            nn.Conv2d(base_ch*1, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.clssifier4_up = nn.Sequential(
            nn.Softmax(dim=1)
        )
        self.san_list = san_list
        if 0 in self.san_list:
            self.SAN_stage_1 = SAN(inplanes=base_ch*8, selected_classes=selected_classes)
            self.SAW_stage_1 = SAW(selected_classes, dim=base_ch * 8, relax_denom=2.0, classifier=self.clssifier1,
                                   work=True)
        if 1 in self.san_list:
            self.SAN_stage_2 = SAN(inplanes=base_ch*4, selected_classes=selected_classes)
            self.SAW_stage_2 = SAW(selected_classes, dim=base_ch * 4, relax_denom=2.0, classifier=self.clssifier2,
                                   work=True)
        if 2 in self.san_list:
            self.SAN_stage_3 = SAN(inplanes=base_ch*2, selected_classes=selected_classes)
            self.SAW_stage_3 = SAW(selected_classes, dim=base_ch * 2, relax_denom=2.0, classifier=self.clssifier3,
                                   work=True)
        if 3 in self.san_list:
            self.SAN_stage_4 = SAN(inplanes=base_ch*1, selected_classes=selected_classes)
            self.SAW_stage_4 = SAW(selected_classes, dim=base_ch*1, relax_denom=2.0, classifier=self.clssifier4, work=True)

    def forward(self, x, e1, e2, e3, e4):
        y, ds1 = self.dec1(x)
        y1_d = self.clssifier1(ds1)
        y1 = self.clssifier1_up(y1_d)

        x_1_ori = None
        san1 = None
        saw_loss_lay1 = None
        if 0 in self.san_list:
            x_1_ori = y
            y = self.SAN_stage_1(y,y1_d)
            san1 = y
            saw_loss_lay1 = self.SAW_stage_1(y)

        y, ds2 = self.dec2(concatenate(y, e4))
        y2_d = self.clssifier2(ds2)
        y2 = self.clssifier2_up(y2_d)

        x_2_ori = None
        san2 = None
        saw_loss_lay2 = None
        if 1 in self.san_list:
            x_2_ori = y
            y = self.SAN_stage_2(y, y2_d)
            san2 = y
            saw_loss_lay2 = self.SAW_stage_2(y)

        y, ds3 = self.dec3(concatenate(y, e3))
        y3_d = self.clssifier3(ds3)
        y3 = self.clssifier3_up(y3_d)

        x_3_ori = None
        san3 = None
        saw_loss_lay3 = None
        if 2 in self.san_list:
            x_3_ori = y
            y = self.SAN_stage_3(y, y3_d)
            san3 = y
            saw_loss_lay3 = self.SAW_stage_3(y)


        y, ds4 = self.dec4(concatenate(y, e2))
        y4_d = self.clssifier4(ds4)
        y4 = self.clssifier4_up(y4_d)

        x_4_ori=None
        san4=None
        saw_loss_lay4=None
        if 3 in self.san_list:
            x_4_ori = y
            y = self.SAN_stage_4(y, y4_d)
            san4 = y
            saw_loss_lay4 = self.SAW_stage_4(y)

        y = self.lastconv(concatenate(y, e1))
        y = self.outconv(y)
        output = self.softmax(y)

        return [output, y4, y3, y2, y1],[x_1_ori,x_2_ori,x_3_ori,x_4_ori],[san1,san2,san3,san4],[saw_loss_lay1,saw_loss_lay2,saw_loss_lay3,saw_loss_lay4]

class aux_decoder(nn.Module):
    def __init__(self, base_ch, cls_num):
        super(aux_decoder, self).__init__()
        self.clssifier4 = nn.Sequential(
            double_conv(base_ch*8, base_ch),
            nn.Conv2d(base_ch, cls_num+1, 1),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier3 = nn.Sequential(
            double_conv(base_ch*4, base_ch),
            nn.Conv2d(base_ch, cls_num+1, 1),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier2 = nn.Sequential(
            double_conv(base_ch*2, base_ch),
            nn.Conv2d(base_ch, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier1 = nn.Sequential(
            double_conv(base_ch, base_ch),
            nn.Conv2d(base_ch, cls_num+1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, e1, e2, e3, e4):
        y4 = self.clssifier4(e4)
        y3 = self.clssifier3(e3)
        y2 = self.clssifier2(e2)
        y1 = self.clssifier1(e1)

        return y1, y2, y3, y4

class Unet(nn.Module):
    def __init__(self, input_nc, base_ch, output_nc, mixstyle_layers=['layer1', 'layer2', 'layer3'], san_lsit=[], selected_classes=[1,2,3,4,5,6,7,8,9,10,11], mixstyle_p=0.5, mixstyle_alpha=0.1):
        super(Unet, self).__init__()
        in_ch = input_nc
        cls_num = output_nc
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cls_num = cls_num
        self.san_lsit = san_lsit
        self.sub_encoders = sub_encoder(in_ch, base_ch, mixstyle_layers, mixstyle_p, mixstyle_alpha)
        self.global_decoder = sub_decoder(base_ch, cls_num, selected_classes,self.san_lsit)

    def forward(self, x, nodes_style=None, node_encoders=False):
        # if self.training:
        #     self.sub_encoders.mixstyle.apply(activate_mixstyle)
        # else:
        #     self.sub_encoders.mixstyle.apply(deactivate_mixstyle)
        e, e1, e2, e3, e4 = self.sub_encoders(x, nodes_style)
        # print(e.shape, e1.shape, e2.shape, e3.shape, e4.shape)
        # exit()
        [output, y4, y3, y2, y1],[x_1_ori,x_2_ori,x_3_ori,x_4_ori],[san1,san2,san3,san4],[saw_loss_lay1,saw_loss_lay2,saw_loss_lay3,saw_loss_lay4] = self.global_decoder(e, e1, e2, e3, e4)
        # if node_encoders:
        #     return output, e, e1, e2, e3, e4
        # if self.training or nodes_style==None:
        #     return [output,y1, y2, y3, y4],[e, e1, e2, e3, e4]
        # else:
        if len(self.san_lsit) == 0:
            return output
        else:
            return output,[x_1_ori,x_2_ori,x_3_ori,x_4_ori],[san1,san2,san3,san4],[saw_loss_lay1,saw_loss_lay2,saw_loss_lay3,saw_loss_lay4]

    def forward_SAN_SAW(self,x, nodes_style=None, node_encoders=False):
        # if self.training:
        #     self.sub_encoders.mixstyle.apply(activate_mixstyle)
        # else:
        #     self.sub_encoders.mixstyle.apply(deactivate_mixstyle)
        e, e1, e2, e3, e4 = self.sub_encoders(x, nodes_style)
        ([output, y4, y3, y2, y1],[x_1_ori,x_2_ori,x_3_ori,x_4_ori],
         [san1,san2,san3,san4],[saw_loss_lay1,saw_loss_lay2,saw_loss_lay3,saw_loss_lay4]) = self.global_decoder(e, e1, e2, e3, e4)
        return [output, y4, y3, y2, y1],[x_1_ori,x_2_ori,x_3_ori,x_4_ori],[san1,san2,san3,san4],[saw_loss_lay1,saw_loss_lay2,saw_loss_lay3,saw_loss_lay4]

    def description(self):
        return 'Multi-encoder U-Net (input channel = {0:d}) for {1:d}-organ segmentation (base channel = {2:d})'.format(self.in_ch, self.cls_num, self.base_ch)

