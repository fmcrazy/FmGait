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
# from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from opengait.utils import get_msg_mgr
from torch.nn import init


from clustercontrast.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap
from clustercontrast.utils.data import IterLoader
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from examples.utils_function import initialization
from examples.utils_function import inference
from examples.utils_function import ccr_inference
from examples.utils_function import leg_ccr_inference
from examples.utils_function import save_ckp
from examples.utils_function import run_test
from examples.utils_function import load_ckpt
from examples.utils_function import generate_cluster_features
from examples.utils_function import vis_plot
from examples.utils_function import oumvlp_vis_plot
from examples.utils_function import t_sne
from examples.utils_function import typ_t_sne
from examples.utils_function import vie_t_sne
from examples.utils_function import cluster_and_memory
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9691'
os.environ['LOCAL_RANK'] = '0'
os.environ["OMP_NUM_THREADS"] = "1"

def main():

    args = parser.parse_args()

    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True  # 告诉CuDNN在运行时根据硬件和输入数据的大小来选择最佳的卷积算法

    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))

    # 下载配置文件
    cfgs = config_loader('./configs/gaitgl/gaitgl_OU_CA.yaml')  # 目标数据集的配置文件

    initialization(cfgs, training=True)
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(args)

    iters = args.iters if (args.iters > 0) else None


    # 创建模型
    model_cfg = cfgs['model_cfg']
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training=True)

    ema_model = Model(cfgs, training=True)
    for param in ema_model.parameters():
        param.requires_grad = False

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
    trainer = ClusterContrastTrainer(model)

    # Model.run_test(model)

    stroed_nm_acc=[]
    stroed_bg_acc=[]
    stroed_cl_acc=[]


    for epoch in range(args.epochs):

        pseudo_labels, memory, pseudo_labeled_dataset, refinement_dataset = cluster_and_memory(model, epoch, args, use_leg=False)

        trainer.memory = memory
        trainer.ema_encoder = ema_model

        train_loader = IterLoader(model.train_loader, length=iters)

        train_loader.new_epoch()

        trainer.train(args, epoch, train_loader, optimizer, pseudo_labeled_dataset,
                      refinement_dataset, print_freq=args.print_freq, train_iters=len(train_loader))

        # 测试
        result_dict = Model.run_test(model)
        # save_ckp(model, cfgs, optimizer, lr_scheduler, epoch+1)

        # stroed_loss.append(loss.cpu().detach().numpy())
        if cfgs['data_cfg']['dataset_name'] == 'CASIA-B':
            stroed_nm_acc.append(result_dict['scalar/test_accuracy/NM@R1'])
            stroed_bg_acc.append(result_dict['scalar/test_accuracy/BG@R1'])
            stroed_cl_acc.append(result_dict['scalar/test_accuracy/CL@R1'])

        elif cfgs['data_cfg']['dataset_name'] == 'OUMVLP':
            stroed_nm_acc.append(result_dict['scalar/test_accuracy/NM@R1'])
        else:
            stroed_nm_acc.append(result_dict['scalar/test_accuracy/Rank-1'])

        lr_scheduler.step()

    # 利用matlab画准确率的图
    if cfgs['data_cfg']['dataset_name'] == 'CASIA-B':
        vis_plot(stroed_nm_acc, stroed_bg_acc, stroed_cl_acc, args.epochs)

    else:
        oumvlp_vis_plot(stroed_nm_acc, args.epochs)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")

    # 需要调整的参数
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")
    parser.add_argument('--k', type=int, default=2,
                        help="hyperparameter for outline")

    parser.add_argument('--refine_weight', type=float, default=0.4,
                        help="sigmoid function")
    parser.add_argument('--sig', type=int, default=30,
                        help="sigmoid function")
    parser.add_argument('--aug_weight', type=float, default=0.4,
                        help="sigmoid function")
    parser.add_argument('--center_sig', type=int, default=5,
                        help="sigmoid function")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=5)  # 将学习率衰减由20改为5
    parser.add_argument('--use_hard', type=bool, default=True)
    parser.add_argument('--use_refine_label', type=bool, default=True)
    parser.add_argument('--use_aug', type=bool, default=True)

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001,  # 修改：将学习率0.00035改成0.0001
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0005)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    # parser.add_argument('--no-cam', action="store_true")
    parser.add_argument('--local_rank', type=int, default=0,
                        help="passed by torch.distributed.launch module")
    main()
