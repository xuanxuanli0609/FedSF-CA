import pdb
# from cv2 import moments
import torch
import numpy as np
from .base_model import BaseModel
from . import bisenetv2
import torch.nn.functional as F
# from fedml_api.standalone.fedprox.optim import FedProx
# from fedml_api.standalone.scaffold.optim import Scaffold
# from fedml_api.standalone.feddyn.optim import FedDyn
# from fedml_api.standalone.fedavg.optim import FedAvg
from algorithm.fedavg.optim import FedAvg
from algorithm.fedprox.optim import FedProx
from algorithm.feddyn.optim import FedDyn
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
from collections import OrderedDict
from ..losses.losses import *
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

log_dir = "./logs"
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)
a_count = 1


class bisenetv2model(BaseModel):
    def name(self):
        return 'UnetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(net='unet')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['seg']
        self.model_names = ['']
        self.visual_names = ['image', 'out']
        opt.net = 'unet'
        self.net = bisenetv2.define_net(input_nc=opt.input_nc, output_nc=opt.output_nc, net=opt.net, \
                                       init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                       san_lsit=opt.san_list, SAN_SAW=opt.use_san_saw,base_ch=opt.bace_ch)

        if self.isTrain:
            self.visual_names.append('label')
            if opt.federated_algorithm == 'fedavg':
                self.optimizer = FedAvg(self.net.parameters(),  # 正式实验
                                        lr=opt.lr,
                                        alpha=0,
                                        momentum=0,
                                        eps=1e-5)
                # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)  # 平时跑模型

            elif opt.federated_algorithm == 'fedprox':
                self.optimizer = FedProx(self.net.parameters(),
                                         lr=opt.lr,
                                         mu=0.0001,
                                         # gmf = opt.gmf,
                                         momentum=0,
                                         nesterov=False,
                                         weight_decay=1e-4,
                                         alpha=0,
                                         eps=1e-5)
            elif opt.federated_algorithm == 'feddyn':
                self.optimizer = FedDyn(self.net.parameters(),
                                        lr=opt.lr,
                                        momentum=0,
                                        nesterov=False,
                                        weight_decay=1e-4,
                                        dyn_alpha=0.0001,
                                        alpha=0,
                                        eps=1e-5)
            elif opt.federated_algorithm == 'feddc':
                self.optimizer = torch.optim.SGD(self.net.parameters(),
                                                     lr=opt.lr,
                                                     weight_decay=1e-4)
            elif opt.federated_algorithm == 'fedddpm':
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)
            else:
                self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer)
        # weight = None
        # if isinstance(opt.loss_weight,list):
        #     weight=torch.Tensor(opt.loss_weight)
        #     weight=weight.to(self.gpu_ids[0])
        self.criterion = bisenetv2.define_loss(opt.loss_type, opt.focal_alpha, opt.output_nc)

        self.loss_type = opt.loss_type
        temperature = 0.05
        self.cont_loss_func = losses.NTXentLoss(temperature)

    def set_input(self, input):  # input(image,label,image_path)

        self.image = input['image'].to(self.gpu_ids[0])
        # if self.isTrain:
        self.label = input['label'].squeeze(1).type(torch.LongTensor)  # .to(self.gpu_ids[0])
        self.images_path = input['path']
        if self.opt.federated_algorithm == 'fedddpm':
            self.fake_image = input['fake_image'].to(self.gpu_ids[0])
            # self.fake_label = input['fake_label'].squeeze(1).type(torch.LongTensor).to(self.gpu_ids[0])
            # self.images = torch.cat((self.image, self.fake_image), dim=0)
            # self.labels = torch.cat((self.label, self.label), dim=0)
            # global a_count
            # writer.add_images("image",
            #                   self.images,
            #                   a_count, dataformats="NCHW")
            # writer.add_images("label",
            #                   torch.cat((input['label'], input['fake_label']), dim=0),
            #                   a_count, dataformats="NCHW")
            # # writer.add_images("fake_image",
            # #                   self.fake_image,
            # #                   a_count, dataformats="NCHW")
            # # writer.add_images("fake_label",
            # #                   input['fake_label'],
            # #                   a_count, dataformats="NCHW")
            # a_count = a_count + 1

    def forward(self):
        if self.net.aux_mode == 'Train':
            self.out,_ = self.net(self.image)
        else:
            self.out,_ = self.net(self.image)
        if np.isnan(np.sum(self.out.detach().cpu().numpy())):
            a = 1
        return self.out

    def set_flag(self):
        class_flag = [1]
        node_enabled_encoders = []
        for c in range(self.out.shape[1] - 1):
            if (c + 1) in np.unique(self.label):
                class_flag.append(1)
                node_enabled_encoders.append(c)
            else:
                class_flag.append(0)
        return class_flag

    def cal_loss(self):#模型计算损失
        if self.loss_type=='Inconsistent_Labels_loss':
            class_flag = self.set_flag()
            l1,l2 = self.criterion[0](self.out, self.label[:,None,...], class_flag)
            l3,l4 = self.criterion[1](self.out, self.label[:,None,...], class_flag)
            # print(l1,l2,l3,l4)
            loss_seg = l1+l2+l3+l4
        else:
            # print(self.out.shape)
            # print(self.label.shape)
            loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        return loss_seg.item()

    def get_loss(self):#模型计算损失
        if self.loss_type=='Inconsistent_Labels_loss':
            class_flag = self.set_flag()
            l1,l2 = self.criterion[0](self.out, self.label[:,None,...], class_flag)
            l3,l4 = self.criterion[1](self.out, self.label[:,None,...], class_flag)
            # print(l1,l2,l3,l4)
            loss_seg = l1+l2+l3+l4
        else:
            # print(self.out.shape)
            # print(self.label.shape)
            loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        return loss_seg

    def backword(self):
        # pdb.set_trace()
        l = self.cal_loss()
        if np.isnan(l):
            a = 1
        try:
            self.loss_seg.backward()
        except:
            a = 1
        return l

    def optimize_parameters(self):
        self.set_requires_grad(self.net, True)
        self.forward()
        self.optimizers[0].zero_grad()
        self.backword()

        # clip grad
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
        if self.opt.federated_algorithm != 'feddyn':
            self.optimizers[0].step()

    def get_Inconsistent_Labels_loss(self,out_fake):
        class_flag = [1]
        node_enabled_encoders = []
        for c in range(self.out.shape[1]-1):
            if (c + 1) in np.unique(self.label):
                class_flag.append(1)
                node_enabled_encoders.append(c)
            else:
                class_flag.append(0)
        l1,l2 = self.criterion[0](self.out, self.label[:,None,...], class_flag)
        l3,l4 = self.criterion[1](self.out, self.label[:,None,...], class_flag)
        loss_seg = 0.5*(l1+l2+l3+l4)
        l1, l2 = self.criterion[0](out_fake, self.label[:, None, ...], class_flag)
        l3, l4 = self.criterion[1](out_fake, self.label[:, None, ...], class_flag)
        loss_seg = loss_seg+0.5*(l1 + l2 + l3 + l4)
        return loss_seg

    def merge_label(self, label, class_flag):
        merged_label = torch.zeros_like(label)
        cc = 1
        for c, class_exist in enumerate(class_flag):
            if c > 0 and class_exist > 0:
                merged_label[label == c] = cc
        return merged_label

    def get_san_saw_loss(self,oris,sans,saw_loss):
        loss_sup = 0
        class_flag = self.set_flag()
        label_one = self.merge_label(self.label, class_flag).cuda()
        loss_in_lays = []
        variables = [self.net.global_decoder.SAN_stage_1.IN, self.net.global_decoder.SAN_stage_2.IN]
        if 2 in self.opt.san_list:
            variables.append(self.net.global_decoder.SAN_stage_3.IN)
        if 3 in self.opt.san_list:
            variables.append(self.net.global_decoder.SAN_stage_4.IN)
        for n in self.opt.san_list:
            label_ = F.adaptive_max_pool2d(label_one.float(), oris[n].size()[2:])  # [2, 1, 4, 32, 32]
            outs = []
            mask = torch.unsqueeze(label_, 1)
            out = oris[n] * mask
            out = variables[n](out)
            outs.append(out)
            outs = sum(outs)
            loss_in_lay = 0.1 * F.smooth_l1_loss(sans[n], outs)
            loss_in_lays.append(loss_in_lay)
            loss_sup += loss_in_lay + 0.1 * saw_loss[n]
        return loss_sup

    def fedddpm_optimize_parameters(self, round_idx):
        self.set_requires_grad(self.net, True)
        # forward
        # self.out = self.net(self.images[0].to(self.gpu_ids[0]))
        # out_fake = self.net(self.images[1].to(self.gpu_ids[0]))

        # compute loss
        # loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        if self.loss_type == 'Inconsistent_Labels_loss' and self.opt.use_san_saw:
            # output, [san1, san2, san3, san4], [saw_loss_lay1, saw_loss_lay2, saw_loss_lay3, saw_loss_lay4]
            self.out, oris, sans, saw_losses = self.net(self.image.to(self.gpu_ids[0]))
            out_fake, fake_oris, fake_sans, fake_saw_losses = self.net(self.fake_image.to(self.gpu_ids[0]))
            loss_seg = self.get_Inconsistent_Labels_loss(out_fake)
            if round_idx>30:
                loss_seg = loss_seg + self.get_san_saw_loss(oris,sans,saw_losses)
                loss_seg = loss_seg + self.get_san_saw_loss(fake_oris,fake_sans,fake_saw_losses)
        elif self.loss_type == 'Inconsistent_Labels_loss':
            self.out = self.net(self.image.to(self.gpu_ids[0]))
            out_fake = self.net(self.fake_image.to(self.gpu_ids[0]))
            loss_seg = self.get_Inconsistent_Labels_loss(out_fake)
        elif self.opt.use_san_saw:
            self.out, oris, sans, saw_losses = self.net(self.image.to(self.gpu_ids[0]))
            out_fake, fake_oris, fake_sans, fake_saw_losses = self.net(self.fake_image.to(self.gpu_ids[0]))
            loss_seg = 0.5 * (self.criterion(self.out, self.label) + self.criterion(out_fake, self.label)) \
                       + 0.5 * (dice_loss(self.out, self.label) * 10 + dice_loss(out_fake, self.label) * 10)
            if round_idx>30:
                loss_seg = loss_seg + self.get_san_saw_loss(oris,sans,saw_losses)
                loss_seg = loss_seg + self.get_san_saw_loss(fake_oris, fake_sans, fake_saw_losses)
        else:
            self.out = self.net(self.image.to(self.gpu_ids[0]))
            out_fake = self.net(self.fake_image.to(self.gpu_ids[0]))
            loss_seg = 0.5 * (self.criterion(self.out, self.label) + self.criterion(out_fake, self.label)) \
                       + 0.5 * (dice_loss(self.out, self.label) * 10 + dice_loss(out_fake, self.label) * 10)

        # zero grad
        self.optimizers[0].zero_grad()
        # backward
        loss_seg.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
        self.optimizers[0].step()

    def feddc_optimize_parameters(self, alpha, local_update_last, global_update_last, global_model_param, hist_i):
        self.set_requires_grad(self.net, True)
        # print('data: ', np.mean(self.image.cpu().numpy()), np.var(self.image.cpu().numpy()))
        # self.forward()
        # loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        if self.loss_type == 'Inconsistent_Labels_loss' and self.opt.use_san_saw:
            # output, [san1, san2, san3, san4], [saw_loss_lay1, saw_loss_lay2, saw_loss_lay3, saw_loss_lay4]
            self.out,oris,sans,saw_losses = self.net(self.image.to(self.gpu_ids[0]))
            loss_seg = self.get_loss()
            loss_seg = loss_seg + self.get_san_saw_loss(oris,sans,saw_losses)
        elif self.loss_type == 'Inconsistent_Labels_loss':
            self.out = self.net(self.image.to(self.gpu_ids[0]))
            loss_seg = self.get_loss()
        else:
            self.out = self.net(self.image.to(self.gpu_ids[0]))
            loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        ## Get f_i estimate
        loss_f_i = loss_seg

        state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device="cuda:0",)
        state_update_diff = Variable(state_update_diff, requires_grad=True)
        local_parameter = None
        # for param in self.net.parameters():
        #     if not isinstance(local_parameter, torch.Tensor):
        #         # Initially nothing to concatenate
        #         local_parameter = param.reshape(-1)
        #     else:
        #         local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
        for param_keys in self.net.state_dict():
            if not isinstance(local_parameter, torch.Tensor):
                # Initially nothing to concatenate
                local_parameter = self.net.state_dict()[param_keys].reshape(-1)
            else:
                local_parameter = torch.cat((local_parameter, self.net.state_dict()[param_keys].reshape(-1)), 0)
        loss_cp = alpha / 2 * torch.sum(
            (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
        loss_cg = torch.sum(local_parameter * state_update_diff)

        loss = loss_f_i + loss_cp + loss_cg
        # print(loss_f_i,loss_cp,loss_cg)
        self.optimizers[0].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(),
                                       max_norm=10)  # Clip gradients to prevent exploding
        self.optimizers[0].step()

    def feseg_optimize_parameters(self, round_idx, prototypes, proto_mask, global_model):
        self.set_requires_grad(self.net, True)
        # print('data: ', np.mean(self.image.cpu().numpy()), np.var(self.image.cpu().numpy()))
        # self.forward()
        # loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        if self.loss_type == 'Inconsistent_Labels_loss':
            self.out, emb = self.net(self.image.to(self.gpu_ids[0]))
            loss_seg = self.get_loss()
            # print(loss_seg)
        else:
            self.out, emb = self.net(self.image.to(self.gpu_ids[0]))
            loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10

        # loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        if round_idx > 2000:
            feat_head = emb
            labels_ = self.label
            _, _, h, w = feat_head.size()

            labels_1 = labels_.unsqueeze(1)
            labels_1 = F.interpolate(labels_1.float(), size=(h, w), mode='nearest')
            labels_1 = labels_1.squeeze(1)
            # print(feat_head.size())
            # print(labels_1.size())
            # print(prototypes.size())
            # print(proto_mask.size())
            # exit()
            if round_idx >= 0:
                proto_mask_tmp = proto_mask.sum(1) < 1
            else:
                proto_mask_tmp = proto_mask < 1
            for ii, bo in enumerate(proto_mask_tmp):
                if bo:
                    labels_1[labels_1 == ii] = 0
            criteria_contrast = bisenetv2.ContrastLoss()
            loss_con = criteria_contrast(feat_head, labels_1, prototypes, proto_mask)
            loss_seg += loss_con

            if round_idx >= 0:
                device = prototypes.device
                with torch.no_grad():
                    logits_t, feat_head_t = global_model(self.image)
                labels_2 = F.interpolate(logits_t.float(), size=(h, w), mode='bilinear')
                labels_2 = torch.softmax(labels_2, dim=1)
                props, labels_2 = torch.max(labels_2, dim=1)
                #                        print(props.max())
                #                        print(props.min())

                mask_ = props < 0.8
                labels_2[mask_] = 0

                for ii, bo in enumerate(proto_mask_tmp):
                    if bo:
                        labels_2[labels_2 == ii] = 0

                loss_con_2 = criteria_contrast(feat_head, labels_2, prototypes, proto_mask)
                loss_seg += loss_con_2

        # zero grad
        self.optimizers[0].zero_grad()
        # backward
        loss_seg.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
        self.optimizers[0].step()


    def set_learning_rate(self, lr):
        for param_group in self.optimizers[0].param_groups:
            param_group['lr'] = lr

    def extract_contour_embedding(self, contour_list, embeddings):
        average_embeddings_list = []
        for contour in contour_list:
            contour = contour.to(embeddings.device)
            contour_embeddings = contour * embeddings
            average_embeddings = torch.sum(contour_embeddings, (-1, -2)) / torch.sum(contour, (-1, -2))
            # print (contour.shape)
            # print (embeddings.shape)
            # print (contour_embeddings.shape)
            # print (average_embeddings.shape)
            average_embeddings_list.append(average_embeddings)
        return average_embeddings_list


    @torch.no_grad()
    def get_protos(self, model, global_round, trainloader_eval):
        model.eval()
        tmp_ = []
        label_list = []
        label_mask_list = []
        for batch_idx, data in enumerate(trainloader_eval):
            image = data['image'].to(self.gpu_ids[0])
            # if self.isTrain:
            label = data['label'].squeeze(1).type(torch.LongTensor)  # .to(self.gpu_ids[0])
            images_path = data['path']
            for n in range(image.shape[0]):
                images, labels = image[n][None,].to(self.device), label[n][None,].to(self.device)
                logits, feat_head = model(images)

                _, _, h, w = feat_head.size()
                labels_2 = F.interpolate(logits.float(), size=(h, w), mode='bilinear')
                labels_2 = torch.softmax(labels_2, dim=1)
                props, labels_2 = torch.max(labels_2, dim=1)
                #                        print(props.max())
                #                        print(props.min())
                mask_ = props < 0.8
                labels_2[mask_] = 0

                feat_head = feat_head.unsqueeze(1)

                labels = labels.unsqueeze(1)
                labels = F.interpolate(labels.float(), size=(h, w), mode='nearest')
                labels = labels.unsqueeze(1)

                #            print(labels.size())
                #            print(labels_2.size())
                #            exit()

                labels_2 = labels_2.unsqueeze(1).unsqueeze(1)

                # labels_2[labels!=255]=labels

                labels = torch.where(labels.float() != 0, labels.float(), labels_2.float())
                unique_l = torch.unique(labels.cpu()).numpy().tolist()
                label_list.extend(unique_l)
                one_hot_ = torch.zeros(self.opt.output_nc).to(self.device)
                for ll in unique_l:
                    ll = int(ll)
                    if ll != 0:
                        one_hot_[ll] = 1
                label_mask_list.append(one_hot_)

                class_ = torch.arange(self.opt.output_nc).to(self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                weight_ = class_ == labels
                weight_ = weight_ / (weight_.sum(3, keepdim=True).sum(4, keepdim=True) + 1e-5)
                out = weight_ * feat_head
                out = out.sum(-1).sum(-1)
                tmp_.append(out)
        tmp_ = torch.cat(tmp_, 0)
        tmp_ = tmp_.permute(1, 0, 2)

        #        print(tmp_.size())
        #        tmp_ = sum(tmp_)/len(tmp_)
        label_mask_ = torch.stack(label_mask_list, 1)
        return tmp_, label_list, label_mask_