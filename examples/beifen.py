# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os
import torch
import collections
import time
from datetime import timedelta
import sys
import numpy as np
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

from clustercontrast.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap
from clustercontrast.utils.data import IterLoader
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from examples.utils_function import get_loader
from examples.utils_function import initialization
from examples.utils_function import inference
from examples.utils_function import ccr_inference
from examples.utils_function import save_ckp
from examples.utils_function import run_test
from examples.utils_function import load_ckpt
from examples.utils_function import generate_cluster_features
from examples.utils_function import vis_plot
# from examples.label_refine import compute_label_iou_matrix
# from examples.label_refine import compute_sample_softlabels
# from examples.label_refine import compute_hard_label


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1899'
os.environ['LOCAL_RANK'] = '0'


start_epoch = best_mAP = 0

# class LabelRefine():
#     def __init__(self, epoch, pseudo_label, label_center):
#         self.epoch = epoch
#         self.pseudo_label = pseudo_label
#         self.label_center = label_center
#
#     def update_label(self):
#         if self.epoch==0:
#             self.pseudo_labels_pervious = torch.tensor(self.pseudo_labels)
#             self.pseudo_centers_pervious = self.label_centers.clone().detach().requires_grad_(False)
#             self.hat_labels = torch.zeros(self.pseudo_label.size(0), self.label_center.size(0))
#
#         else:
#             hard_labels = compute_hard_label(self.pseudo_labels_pervious, self.pseudo_centers_pervious)
#
#             pseudo_labels_current = torch.tensor(self.pseudo_labels)
#             pseudo_centers_current = self.label_centers.clone().detach().requires_grad_(False)
#             iou_mat = compute_label_iou_matrix(self.pseudo_labels_pervious, pseudo_labels_current)
#             norm_iou_mat = (iou_mat.t() / iou_mat.t().sum(0)).t()
#
#             self.hat_labels = hard_labels.mm(norm_iou_mat)
#
#             self.pseudo_labels_pervious = pseudo_labels_current
#             self.pseudo_centers_pervious = pseudo_centers_current


def main():

    args = parser.parse_args()

    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True  # 告诉CuDNN在运行时根据硬件和输入数据的大小来选择最佳的卷积算法

    # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))  # 指定输出的日志文件
    # print("==========\nArgs:{}\n==========".format(args))

    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))

    # # 下载配置文件
    # pretrained_cfgs = config_loader('../configs/gaitgl/gaitgl_OUMVLP.yaml')  # 源数据集预训练模型的配置文件
    # cfgs = config_loader('../configs/gaitgl/gaitgl.yaml')  # 目标数据集的配置文件

    # 下载配置文件
    pretrained_cfgs = config_loader('../configs/gaitset/gaitset_OUMVLP.yaml')  # 源数据集预训练模型的配置文件
    cfgs = config_loader('../configs/gaitset/gaitset.yaml')  # 目标数据集的配置文件

    initialization(cfgs, training=True)
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(args)

    # 下载数据集
    rank = torch.distributed.get_rank()
    data_cfg = cfgs['data_cfg']
    iters = args.iters if (args.iters > 0) else None
    train_loader = get_loader(cfgs, data_cfg, train=True, inference=True)
    test_loader = get_loader(cfgs, data_cfg, train=False, inference=True)  # 测试数据集

    # 创建模型
    model_cfg = pretrained_cfgs['model_cfg']
    Model = getattr(models, model_cfg['model'])
    model = Model(pretrained_cfgs, training=False, num_class=len(train_loader))
    ema_model = Model(pretrained_cfgs, training=False, num_class=len(train_loader))
    for param in ema_model.parameters():
        param.detach_()

    # BN层实现分布式训练
    # if cfgs['trainer_cfg']['sync_BN']:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 将模型的标准归一化层换成同步归一化层，以适应分布式训练的需求
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()

    # 实现分布训练
    # model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # Trainer

    trainer = ClusterContrastTrainer(model, ema_model)

    # 使用tensoboard
    # tag = "{}-{}-{}".format(args.k1, args.k2, args.iters)
    # writer = SummaryWriter('./log/{}'.format(tag))

    # print("==> Test {} with the {} which pretrained in {}  ".format(cfgs['data_cfg']['dataset_name'], pretrained_cfgs['model_cfg']['model'], pretrained_cfgs['data_cfg']['dataset_name']))
    # run_test(model, test_loader, cfgs)

    stroed_loss = []
    stroed_nm_acc=[]
    stroed_bg_acc=[]
    stroed_cl_acc=[]

    for epoch in range(args.epochs):

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_loader(cfgs, data_cfg, train=True,  inference=True)

            seqs_data = []
            for path in cluster_loader.dataset.seqs_info:
                seqs_data.append(path[-1][0])
            rank = torch.distributed.get_rank()
            features, _ = ccr_inference(model, cluster_loader, rank)
            features = torch.cat([features[f].unsqueeze(0) for f in sorted(seqs_data)], 0)  # (8107,15872)

            features_array = F.normalize(features, dim=1).cpu().numpy()
            feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1, knn_method='faiss-gpu')
            del features_array

            s = time.time()
            pseudo_labels = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=args.eps, cluster_num=args.k2)
            pseudo_labels = pseudo_labels.astype(np.intp)

            print('cluster cost time: {}'.format(time.time() - s))
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate new dataset and calculate cluster centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        label_centers = cluster_features


        # 伪标签细化， https://github.com/2han9x1a0release/RLCC
        # refine_label = LabelRefine(epoch, pseudo_labels, label_centers)
        # LabelRefine.update_label()
        #
        # current_pseudo_labels = compute_hard_label(pseudo_labels, label_centers)
        # alpha = 0.9
        # updated_labels = alpha * current_pseudo_labels + (1 - alpha) * refine_label.hat_labels

        # updated_labels_dataset = OrderedDict()
        # for i, (fname, label, updated_label) in enumerate(zip(sorted(seqs_data), pseudo_labels, updated_labels)):
        #     if label!=-1:
        #         updated_labels_dataset[fname]=updated_label

        num_features = features.size(-1)
        del cluster_loader, features

        # 用聚类中心初始化分类器的权重
        # model.module.classifier.weight.data[:num_cluster].copy_(F.normalize(label_centers, dim=1).float().cuda())
        # ema_model.module.classifier.weight.data[:num_cluster].copy_(F.normalize(label_centers, dim=1).float().cuda())

        # Create hybrid memory
        memory = ClusterMemory(num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = OrderedDict()
        for i, (fname, label) in enumerate(zip(sorted(seqs_data), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset[fname] = label.item()

        train_loader = get_loader(cfgs, data_cfg, train=True, inference=False)
        train_loader = IterLoader(train_loader, length=iters)

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader.new_epoch()
        updated_labels_dataset = []
        loss = trainer.train(epoch, train_loader, optimizer,label_centers, pseudo_labeled_dataset,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        # 参数的时间平均更新


        # if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
        result_dict = run_test(model, test_loader, cfgs)

        # writer.add_scalars("NM-loss/acc", {'loss':loss, 'acc':result_dict['scalar/test_accuracy/NM@R1']}, epoch)
        # writer.add_scalars("BG-loss/acc", {'loss':loss, 'acc':result_dict['scalar/test_accuracy/BG@R1']}, epoch)
        # writer.add_scalars("CL-loss/acc", {'loss':loss, 'acc':result_dict['scalar/test_accuracy/CL@R1']}, epoch)

        # save_ckp(model, pretrained_cfgs, cfgs, epoch+1)


        stroed_loss.append(loss.cpu().detach().numpy())
        stroed_nm_acc.append(result_dict['scalar/test_accuracy/NM@R1'])
        # stroed_bg_acc.append(result_dict['scalar/test_accuracy/BG@R1'])
        # stroed_cl_acc.append(result_dict['scalar/test_accuracy/CL@R1'])


        lr_scheduler.step()

    # writer.close()

    # 利用matlab画准确率的图
    vis_plot(stroed_nm_acc, args.epochs)


    # model = load_ckpt(model,pretrained_cfgs, cfgs)
    # run_test(model, test_loader, cfgs)


    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    # parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
    #                     choices=datasets.names())
    # parser.add_argument('-b', '--batch-size', type=int, default=2)
    # parser.add_argument('-j', '--workers', type=int, default=1)
    # parser.add_argument('--height', type=int, default=256, help="input height")
    # parser.add_argument('--width', type=int, default=128, help="input width")
    # parser.add_argument('--num-instances', type=int, default=4,
    #                     help="each minibatch consist of "
    #                          "(batch_size // num_instances) identities, and "
    #                          "each identity has num_instances instances, "
    #                          "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")
    # model
    # parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                     choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00005,  # 修改：将学习率0.00035改成0.0001
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--iters', type=int, default=250)
    parser.add_argument('--step-size', type=int, default=15)   # 将学习率衰减由20改为5

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    # working_dir = osp.dirname(osp.abspath(__file__))
    # parser.add_argument('--data-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'data'))
    # parser.add_argument('--logs-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'logs'))
    # parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")
    main()
