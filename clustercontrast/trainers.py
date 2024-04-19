from __future__ import print_function, absolute_import

import copy
import time
from .utils.meters import AverageMeter
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import nn
from opengait.data.transform import get_transform
from opengait.utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
import numpy as np
import cv2
from collections import OrderedDict
import collections
from label_refinement.sil_score import compute_dist_cluster
from label_refinement.sil_score import label_refinement



def compute_soft_label(soft_labels, labels, features, mu_s=0.3):
    N = len(labels)
    C = features.size(0)
    onehot_labels = torch.full(size=(N, C), fill_value=0).cuda()
    labels_lis = [label for label in labels if label != -1]
    for i, label in enumerate(labels):
        onehot_labels[i][label] = 1

    for i, label in enumerate(labels):
        labels_lis[i] = mu_s * soft_labels[i] + (1 - mu_s) * onehot_labels[i]
    refinement_pseudo_labels = torch.stack(labels_lis)
    refinement_pseudo_labels = ts2np(refinement_pseudo_labels)
    return refinement_pseudo_labels


# def extract_probabilities(features, centers, temp=30):
#     features = F.normalize(features, dim=1)
#     logits = temp*features.mm(centers.t().clone())
#     prob = F.softmax(logits, 1)
#     return prob
#
# def compute_soft(ema_labels, labels):
#     ema_labels = torch.stack(ema_labels)
#     # ema_labels = ema_labels.cuda()
#     labels = labels.cuda()
#     ones_tensor = torch.ones_like(labels)
#     for i, label in enumerate(labels):
#         ones_tensor[i] = 0.5 * ema_labels[i] + 0.5*labels[i]
#     return ones_tensor
#
# def compute_model_soft(soft_labels, refine_label, pseudo_labels):
#     # ema_labels = torch.stack(ema_labels)
#     # ema_labels = ema_labels.cuda()
#     # labels = labels.cuda()
#     onehot_labels= collections.defaultdict(list)
#     for i, label in enumerate(pseudo_labels):
#         if label==-1:
#             continue
#         onehot_labels[i] = 0.3 * soft_labels[i] + 0.7 * refine_label[i]
#     return onehot_labels
#
# def compute_dist_martix(labels, label_centers, features, sig):
#     dist_martix = []
#     for i, label in enumerate(labels):
#         if label == -1:
#             continue
#         dist_cluster = compute_dist_cluster(features[i], label_centers, sig)
#         dist_martix.append(dist_cluster)
#
#     return dist_martix

class ClusterContrastTrainer(object):
    def __init__(self, encoder, ema_encoder=None, memory=None, aug_memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.aug_memory = aug_memory
        self.ema_encoder = ema_encoder

        # self.criterion_ce = CrossEntropyLabelSmooth(self.num_cluster).cuda()

    def train(self, args, epoch, data_loader, optimizer, pseudo_labeled_dataset, refinement_pseudo_labeled_dataset
              , print_freq, train_iters):
        self.encoder.train()
        self.ema_encoder.train()
        # torch.autograd.set_detect_anomaly(True)
        batch_time = AverageMeter()  # 每个batch处理数据的时间
        data_time = AverageMeter()  # 每个batch加载数据的时间
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):

            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            ipts, ema_ipts, labels, fnames, _ = self._inputs_pretreament(inputs, pseudo_labeled_dataset,
                                                                               refinement_pseudo_labeled_dataset)

            f_out= self._forward(ipts)
            refinement_labels, _ = label_refinement(labels, f_out, f_out, self.memory.features, 0.4,
                                                 30)
            refinement_labels = [refinement_labels[key] for key in refinement_labels.keys()]
            refinement_labels = torch.stack(refinement_labels)

            with torch.no_grad():
                ema_out = self.ema_forward(ema_ipts)

            memory_loss = self.memory(f_out, ema_out, labels, refinement_labels, args.use_refine_label, args.use_aug)

            # 总的损失
            loss = memory_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            self._updata_ema(self.encoder, self.ema_encoder, epoch * len(data_loader)+i)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.4f} ({:.4f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

        # return loss


    def pre_inputs(self, inputs, index_lis):
        inputs[0][0] = [elements for index, elements in enumerate(inputs[0][0]) if index not in index_lis]
        inputs[1] = [elements for index, elements in enumerate(inputs[1]) if index not in index_lis]
        inputs[2] = [elements for index, elements in enumerate(inputs[2]) if index not in index_lis]
        inputs[3] = [elements for index, elements in enumerate(inputs[3]) if index not in index_lis]
        inputs[4] = [elements for index, elements in enumerate(inputs[4]) if index not in index_lis]
        return inputs

    def _inputs_pretreament(self, inputs, pseudo_labeled_dataset, refinement_pseudo_labeled_dataset):

        _, labels, _, _, fnames, _ =inputs
        index_lis = []
        refinement_lables = []
        for i, fname in enumerate(fnames):
            if fname in pseudo_labeled_dataset  and fname in refinement_pseudo_labeled_dataset:
                labels[i]= pseudo_labeled_dataset[fname]
                refinement_lables.append(refinement_pseudo_labeled_dataset[fname])
            else:
                index_lis.append(i)
        inputs = self.pre_inputs(inputs, index_lis)
        l = int(len(inputs[1]) / 2)
        ema_inputs = copy.deepcopy(inputs)

        ipts = self.encoder.inputs_pretreament(inputs, use_leg=False)
        ema_ipts = self.encoder.aug_inputs_pretreament(ema_inputs, use_leg=False)

        # ipts = self.encoder.q_inputs_pretreament(inputs, use_leg=False, length=l)
        # ema_ipts = self.encoder.k_inputs_pretreament(ema_inputs, use_leg=False, length=l)

        # 对输入数据进行数据增强
        # aug_inputs = copy.deepcopy(inputs)
        # aug_ipts = self.encoder.dataaug_inputs_pretreament(aug_inputs, use_leg=False)

        # labels = torch.tensor(inputs[1][:l])
        # refinement_lables=refinement_lables[:l]
        labels = torch.tensor(inputs[1])
        refinement_lables = torch.stack(refinement_lables)
        fnames = inputs[4]
        return ipts, ema_ipts, labels.cuda(), fnames, refinement_lables

    def _updata_ema(self, model, ema_model, global_step, alpha = 0.999):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        if global_step >= 99:
            alpha = min(1 - 1 / (global_step - 98), alpha)

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            # ema_param.data.add_(1 - alpha, param.data)
            ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data

    def _forward(self, inputs):
        with autocast(enabled=self.encoder.engine_cfg['enable_float16']):
            retval = self.encoder(inputs)
            inference_feat = retval['inference_feat']
            outputs = inference_feat['embeddings']
            outputs = outputs.view(outputs.size(0), -1)
            outputs = outputs.to(torch.float32)

        return outputs

    def ema_forward(self, ema_inputs):
        with autocast(enabled=self.encoder.engine_cfg['enable_float16']):
            ema_retval = self.ema_encoder(ema_inputs)
            ema_inference_feat = ema_retval['inference_feat']
            ema_outputs = ema_inference_feat['embeddings']
            ema_outputs = ema_outputs.view(ema_outputs.size(0), -1)
            ema_outputs = ema_outputs.to(torch.float32)
        return ema_outputs



