import copy
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
import math
try:
    from ..losses.losses import *
    from ..losses.lnconsistent_labels_loss import *
except:
    pass
backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'
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
    elif type == 'back':
        criterion = BackCELoss(class_num)
    else:
        raise NotImplementedError('loss type [%s] is not recoginized' % type)
    return criterion
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
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)



class DetailBranch(nn.Module):

    def __init__(self, input_nc):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(input_nc, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):

    def __init__(self,input_nc):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(input_nc, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        # feat = torch.mean(x, dim=(2, 3), keepdim=True)
        # print(x.shape)
        # print(feat.shape)
        feat = x
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self,input_nc):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock(input_nc)
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(nn.Module):
    def __init__(self,input_nc, n_classes, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        self.input_nc = input_nc
        self.aux_mode = aux_mode
        self.detail = DetailBranch(input_nc)
        self.segment = SegmentBranch(input_nc)
        self.bga = BGALayer()
        # self.softmax = nn.Softmax(dim=1)
        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=256)

        self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        ######
        if self.aux_mode=='eval':
            h_,w_ = size
            if h_%32!=0:
                new_h = math.ceil(h_/32)*32
                pad_h  = new_h-h_
            else: 
                pad_h=0
            if w_%32!=0:
                new_w = math.ceil(w_/32)*32
                pad_w  = new_w-w_
            else:
                pad_w=0
            
            x = torch.nn.functional.pad(x,(0,pad_w,0,pad_h),mode='reflect')    
        #####

        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        emb = self.proj_head(feat_head)
        logits = self.head(feat_head)
        '''
        sss
        '''
        # logits = self.softmax(logits)

        if self.aux_mode == 'train':
            return logits,emb
        elif self.aux_mode == 'eval':
            logits = logits[:,:,:h_,:w_]
            return logits,
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    def load_pretrain(self):
        # state = modelzoo.load_url(backbone_url)
        state = torch.load('segmentation/myseg/backbone_v2.pth')  # baidu server
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', ):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

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
        net.init_weights()
    return net

def define_net(input_nc, output_nc, net='unet', init_type='xavier_uniform', init_gain=1.0, gpu_ids=[],
               main_device=0, SAN_SAW=False, san_lsit=[0,1],base_ch=32):
    if net != 'bisenetv2':
        init_type = None
    if net == 'bisenetv2':
        net = BiSeNetV2(input_nc=input_nc, n_classes=output_nc)#input_nc=input_nc, output_nc=output_nc, base_ch=base_ch, san_lsit=san_lsit, selected_classes=list(range(1,output_nc+1)))
    else:
        net = BiSeNetV2(input_nc=input_nc, n_classes=output_nc)
        # raise NotImplementedError('model name [%s] is not recoginized'%net)
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=copy.deepcopy(gpu_ids),
                    main_device=main_device)
if __name__ == "__main__":
    #  x = torch.randn(16, 3, 1024, 2048)
    #  detail = DetailBranch()
    #  feat = detail(x)
    #  print('detail', feat.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  stem = StemBlock()
    #  feat = stem(x)
    #  print('stem', feat.size())
    #
    #  x = torch.randn(16, 128, 16, 32)
    #  ceb = CEBlock()
    #  feat = ceb(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 32, 16, 32)
    #  ge1 = GELayerS1(32, 32)
    #  feat = ge1(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 16, 16, 32)
    #  ge2 = GELayerS2(16, 32)
    #  feat = ge2(x)
    #  print(feat.size())
    #
    #  left = torch.randn(16, 128, 64, 128)
    #  right = torch.randn(16, 128, 16, 32)
    #  bga = BGALayer()
    #  feat = bga(left, right)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 128, 64, 128)
    #  head = SegmentHead(128, 128, 19)
    #  logits = head(x)
    #  print(logits.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  segment = SegmentBranch()
    #  feat = segment(x)[0]
    #  print(feat.size())
    #
    x = torch.randn(1, 3, 1024, 2048)
    model = BiSeNetV2(input_nc=3,n_classes=4)
    outs = model(x)
    for out in outs:
        print(out.size())
    #  print(logits.size())

    #  for name, param in model.named_parameters():
    #      if len(param.size()) == 1:
    #          print(name)


class ContrastLoss(nn.Module):
    def __init__(self, ignore_lb=255):
        super(ContrastLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.max_anchor = 512
        self.temperature = 0.07

    def _anchor_sampling(self, embs, labels):
        device = embs.device
        b_, c_, h_, w_ = embs.size()
        class_u = torch.unique(labels)
        class_u_num = len(class_u)
        if 0 in class_u:
            class_u_num = class_u_num - 1

        if class_u_num == 0:
            return None, None

        num_p_c = self.max_anchor // class_u_num

        embs = embs.permute(0, 2, 3, 1).reshape(-1, c_)

        labels = labels.view(-1)
        index_ = torch.arange(len(labels))
        index_ = index_.to(device)

        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
            # print(cls_)
            if cls_ != 0:
                mask_ = labels == cls_
                mask_ = mask_.to(device)
                labels = labels.to(device)
                cls_ = cls_.to(device)
                selected_index_ = torch.masked_select(index_, mask_)
                if len(selected_index_) > num_p_c:
                    sel_i_i = torch.arange(len(selected_index_))
                    sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]
                    selected_index_ = selected_index_[sel_i]
                #             print(selected_index_.size())
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                sampled_label_list.append(torch.ones(len(selected_index_)).to(device) * cls_)
        # print('&'*10)
        # print('sampled_list:',sampled_list)
        # print('sampled_label_list:',sampled_label_list)
        sampled_list = torch.cat(sampled_list, 0)
        sampled_label_list = torch.cat(sampled_label_list, 0)

        return sampled_list, sampled_label_list

    def forward(self, embs, labels, proto_mem, proto_mask):
        device = proto_mem.device
        anchors, anchor_labels = self._anchor_sampling(embs, labels)
        if anchors is None:
            loss = torch.tensor(0).to(device)
            return loss

            # print(anchors.size())
        # print(anchor_labels.size())
        # exit()

        # if self.args.kmean_num > 0:
        if 1 > 0:
            C_, km_, c_ = proto_mem.size()
            proto_labels = torch.arange(C_).unsqueeze(1).repeat(1, km_)
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx, proto_mask.bool())

            proto_labels = proto_labels.to(device)
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
        #
        #
        # else:
        #     C_, c_ = proto_mem.size()
        #     proto_labels = torch.arange(C_)
        #     proto_mem_ = proto_mem
        #     proto_labels = proto_labels
        #     proto_labels = proto_labels[sel_idx]
        #     proto_labels = proto_labels.to(device)
        #     proto_mask = proto_mask
        #     proto_idx = torch.arange(len(proto_mask))
        #     proto_idx = proto_idx.to(device)
        #     sel_idx = torch.masked_select(proto_idx, proto_mask.bool())
        #     proto_mem_ = proto_mem_[sel_idx]
        #     proto_labels = proto_labels[sel_idx]
        #     proto_labels = proto_labels.to(device)

        #        print(proto_mem_.size())
        #        print(proto_labels.size())
        #        exit()
        anchor_dot_contrast = torch.div(torch.matmul(anchors, proto_mem_.T), self.temperature)
        mask = anchor_labels.unsqueeze(1) == proto_labels.unsqueeze(0)
        mask = mask.float()
        mask = mask.to(device)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)

        # mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits) * mask
        #        print(exp_logits.size())
        #        print(logits.size())
        #        print(neg_logits.size())
        #        exit()
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        if torch.isnan(loss):
            print('!' * 10)
            print(torch.unique(logits))
            print(torch.unique(exp_logits))
            print(torch.unique(neg_logits))
            print(torch.unique(log_prob))
            print(torch.unique(mask.sum(1)))
            print(mask)
            print(torch.unique(anchor_labels))
            print(proto_labels)
            print(torch.unique(proto_labels))

            exit()
        #        print(loss)
        #        print('*'*10)
        return loss