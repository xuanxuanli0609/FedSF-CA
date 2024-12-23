import torch
import torch.nn as nn

class ContrastLoss(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self, embs, labels):
        device = embs.device
        b_, c_, h_, w_ = embs.size()
        class_u = torch.unique(labels)
        class_u_num = len(class_u)
        if 255 in class_u:
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
            #     print(cls_)
            if cls_ != 255:
                mask_ = labels == cls_
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
        sampled_list = torch.cat(sampled_list, 0)
        sampled_label_list = torch.cat(sampled_label_list, 0)

        return sampled_list, sampled_label_list

    def forward(self, embs, labels, proto_mem, proto_mask):
        device = proto_mem.device
        anchors, anchor_labels = self._anchor_sampling(embs, labels)
        if anchors is None:
            loss = torch.tensor(0).to(device)
            return loss

        if self.args.kmean_num > 0:
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
        else:
            C_, c_ = proto_mem.size()
            proto_labels = torch.arange(C_)
            proto_mem_ = proto_mem
            proto_labels = proto_labels
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
            proto_mask = proto_mask
            proto_idx = torch.arange(len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = torch.masked_select(proto_idx, proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)

        anchor_dot_contrast = torch.div(torch.matmul(anchors, proto_mem_.T), self.temperature)
        mask = anchor_labels.unsqueeze(1) == proto_labels.unsqueeze(0)
        mask = mask.float()
        mask = mask.to(device)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits) * mask
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
        return loss