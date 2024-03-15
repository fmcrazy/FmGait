# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os
import torch
import collections
import time
from datetime import timedelta
import sys
import numpy as np
import math
from sklearn.cluster import DBSCAN
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
import argparse
from opengait.modeling import models
from opengait.utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from opengait.utils import get_msg_mgr
from torch.nn import init
from opengait.utils.common import ts2np

from clustercontrast.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap
from clustercontrast.utils.data import IterLoader
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from examples.utils_function import get_loader
from examples.utils_function import initialization
from examples.utils_function import inference
from examples.utils_function import ccr_inference
from examples.utils_function import leg_ccr_inference
from examples.utils_function import save_ckp
from examples.utils_function import run_test
from examples.utils_function import load_ckpt
from examples.utils_function import generate_cluster_features
from examples.utils_function import vis_plot
from examples.utils_function import t_sne
from examples.utils_function import typ_t_sne
from examples.utils_function import vie_t_sne
from label_refinement.two_epoch  import compute_label_iou_matrix
from label_refinement.two_epoch  import compute_sample_softlabels
from label_refinement.two_epoch import compute_hard_label
from label_refinement.two_epoch import extract_probabilities
from label_refinement.sil_score import label_refinement
from label_refinement.sil_score import compute_label_centers
import gc


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8754'
os.environ['LOCAL_RANK'] = '0'


start_epoch = best_mAP = 0


def cluster_and_memory(cfgs, data_cfg, model, epoch, args, use_leg=False):
    with torch.no_grad():
        print('==> Create pseudo labels for unlabeled data')
        cluster_loader = get_loader(cfgs, data_cfg, train=True, inference=True)

        seqs_data = []
        for path in cluster_loader.dataset.seqs_info:
            seqs_data.append(path[-1][0])
        rank = torch.distributed.get_rank()

        # 计算全局特征
        if use_leg:
            features, labels, label_typ, label_vie = leg_ccr_inference(model, cluster_loader, rank)
        else:
            features, labels, label_typ, label_vie = ccr_inference(model, cluster_loader, rank)
        features = torch.cat([features[f].unsqueeze(0) for f in sorted(seqs_data)], 0)  # (8107,15872)
        labels = [labels[f] for f in sorted(seqs_data)]
        label_typ = [label_typ[f] for f in sorted(seqs_data)]
        label_vie = [label_vie[f] for f in sorted(seqs_data)]

        del cluster_loader

        # 可视化服装问题的图
        if epoch == args.epochs - 1:
            typ_t_sne(features, labels, label_typ, person=3, stride=1)

        # 可视化角度问题
        # vie_t_sne(features, labels, label_vie, person=5, stride=1)

        # 可视化训练效果
        # if epoch == args.epochs - 1:
        #     t_sne(features, labels)

        features_array = F.normalize(features, dim=1).cpu().numpy()
        feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1, knn_method='faiss-gpu')
        del features_array

        s = time.time()
        pseudo_labels = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=args.eps, cluster_num=args.k2)
        pseudo_labels = pseudo_labels.astype(np.intp)
        del feat_dists, feat_nbrs

        print('cluster cost time: {}'.format(time.time() - s))
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

    # cluster_features = generate_cluster_features(pseudo_labels, features)
    alpha = 0.1 * math.tanh(epoch - args.epochs / 2)
    cluster_features = compute_label_centers(pseudo_labels, features, alpha)  # 使用轮廓分数计算聚类质心

    refinement_pseudo_labels = label_refinement(pseudo_labels, features, cluster_features, beta=0.8)
    refinement_pseudo_labels = torch.stack(refinement_pseudo_labels)
    refinement_pseudo_labels = ts2np(refinement_pseudo_labels)
    refinement_pseudo_labeled_dataset = OrderedDict()
    for i, (fname, label , refinement_label) in enumerate(zip(sorted(seqs_data), labels, refinement_pseudo_labels)):
        if label !=-1:
            refinement_pseudo_labeled_dataset[fname] = refinement_label

    num_features = features.size(-1)
    pseudo_labeled_dataset = OrderedDict()
    fnames = []
    for i, (fname, label) in enumerate(zip(sorted(seqs_data), pseudo_labels)):
        if label != -1:
            pseudo_labeled_dataset[fname] = label.item()
            fnames.append(fname)

    # if use_leg:
    #     return features, pseudo_labels, pseudo_labeled_dataset, fnames



    memory = ClusterMemory(num_features, num_cluster, temp=args.temp,
                           momentum=args.momentum, use_hard=args.use_hard).cuda()
    memory.features = F.normalize(cluster_features, dim=1).cuda()



    print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

    return memory.features, pseudo_labels, memory, pseudo_labeled_dataset, fnames, seqs_data, refinement_pseudo_labeled_dataset




def main():

    args = parser.parse_args()
    gc.collect()

    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True  # 告诉CuDNN在运行时根据硬件和输入数据的大小来选择最佳的卷积算法

    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))

    # 下载配置文件
    pretrained_cfgs = config_loader('configs/gaitset/gaitset.yaml')  # 源数据集预训练模型的配置文件
    cfgs = config_loader('configs/gaitset/gaitset_OUMVLP.yaml')  # 目标数据集的配置文件


    initialization(cfgs, training=True)
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(args)

    # 下载数据集
    rank = torch.distributed.get_rank()
    data_cfg = cfgs['data_cfg']
    iters = args.iters if (args.iters > 0) else None


    # 创建模型
    model_cfg = pretrained_cfgs['model_cfg']
    Model = getattr(models, model_cfg['model'])
    model = Model(pretrained_cfgs, training=False)

    # BN层实现分布式训练
    # if cfgs['trainer_cfg']['sync_BN']:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 将模型的标准归一化层换成同步归一化层，以适应分布式训练的需求
    # if cfgs['trainer_cfg']['fix_BN']:
    #     model.fix_BN()

    # 实现分布训练
    # model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    del params


    # trainer = ClusterContrastTrainer(model)

    # print("==> Test {} with the {} which pretrained in {}  ".format(cfgs['data_cfg']['dataset_name'], pretrained_cfgs['model_cfg']['model'], pretrained_cfgs['data_cfg']['dataset_name']))
    # test_loader = get_loader(cfgs, data_cfg, train=False, inference=True)
    # run_test(model, test_loader, cfgs)

    # stroed_loss = []
    # stroed_nm_acc=[]
    # stroed_bg_acc=[]
    # stroed_cl_acc=[]


    for epoch in range(args.epochs):
        trainer = ClusterContrastTrainer(model)

        # 使用全局特征
        cluster_features, pseudo_labels, memory, pseudo_labeled_dataset, global_fnames, seqs_data, refinement_dataset = cluster_and_memory(cfgs, data_cfg, model,
                                                                                                 epoch, args, use_leg=False)

        # 使用腿的特征
        leg_features, leg_pseudo_labels, leg_memory, leg_pseudo_labeled_dataset, leg_fnames, _, leg_refinement_dataset= cluster_and_memory(cfgs,
                                                                                                                 data_cfg,
                                                                                                                 model,
                                                                                                                 epoch,
                                                                                                                 args,
                                                                                                                 use_leg=True)
        del leg_features
        right_fnames = set(global_fnames) & set(leg_fnames)

        trainer.memory = memory
        trainer.leg_memory = leg_memory


        train_loader = get_loader(cfgs, data_cfg, train=True, inference=False)
        train_loader = IterLoader(train_loader, length=iters)

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, pseudo_labeled_dataset, leg_pseudo_labeled_dataset, right_fnames, cluster_features,
                      refinement_dataset, leg_refinement_dataset,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        # 测试
        test_loader = get_loader(cfgs, data_cfg, train=False, inference=True)  # 测试数据集
        result_dict = run_test(model, test_loader, cfgs)
        del test_loader , train_loader
        del trainer.memory,trainer.leg_memory, trainer, leg_memory, memory, cluster_features
        # if epoch == 10:
        #     save_ckp(model, pretrained_cfgs, cfgs, epoch+1)

        # stroed_loss.append(loss.cpu().detach().numpy())
        # stroed_nm_acc.append(result_dict['scalar/test_accuracy/NM@R1'])
        # stroed_bg_acc.append(result_dict['scalar/test_accuracy/BG@R1'])
        # stroed_cl_acc.append(result_dict['scalar/test_accuracy/CL@R1'])

        lr_scheduler.step()

    # 利用matlab画准确率的图
    # vis_plot(stroed_nm_acc, stroed_bg_acc, stroed_cl_acc, args.epochs)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")

    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001,  # 修改：将学习率0.00035改成0.0001
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--step-size', type=int, default=5)   # 将学习率衰减由20改为5

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")
    main()
