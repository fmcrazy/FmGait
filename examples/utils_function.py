# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import copy
import os.path as osp

from opengait.utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
import torch.utils.data as tordata
from tqdm import tqdm
from torch.cuda.amp import autocast

from opengait.data.collate_fn import CollateFn
from opengait.data.dataset import DataSet
import opengait.data.sampler as Samplers
from opengait.utils import Odict, mkdir, ddp_all_gather
from opengait.utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from opengait.evaluation import evaluator as eval_functions
from opengait.utils import NoOp
from label_refinement.sil_score import label_refinement
from label_refinement.sil_score import compute_label_centers
from label_refinement.sil_score import compute_label_centers_my
import os
import torch
import collections
import time
import numpy as np
import math
import torch.nn.functional as F

from opengait.utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
from collections import OrderedDict

from opengait.utils import get_msg_mgr
from opengait.utils.common import ts2np

from clustercontrast.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from sklearn.cluster import DBSCAN


import os
import torch
from sklearn.manifold import TSNE
import matplotlib

# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt



def cluster_and_memory(model, epoch, args, use_leg=False):
    with torch.no_grad():
        print('==> Create pseudo labels for unlabeled data')
        seqs_data = [path[-1][0] for path in model.inference_train_loader.dataset.seqs_info ]
        rank = torch.distributed.get_rank()
        features, aug_features, out_shape= model.ccr_inference(rank, use_leg)
        aug_features = torch.cat([aug_features[f].unsqueeze(0) for f in sorted(seqs_data)], 0)
        features = torch.cat([features[f].unsqueeze(0) for f in sorted(seqs_data)], 0)  # (8107,15872)

        features_array = F.normalize(features, dim=1).cpu().numpy()
        feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1, knn_method='faiss-gpu')
        del features_array
        #
        s = time.time()
        pseudo_labels = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=args.eps, cluster_num=args.k2)
        pseudo_labels = pseudo_labels.astype(np.intp)

        del feat_dists, feat_nbrs

    # 使用聚类中样本的平均值计算质心
    # cluster_features = generate_cluster_features(pseudo_labels, features)

    # # 使用轮廓分数计算质心
    # alpha = 0.1 * math.tanh(epoch - args.epochs / 2)
    # cluster_features, pseudo_labels = compute_label_centers(pseudo_labels, features, alpha)  # 使用轮廓分数计算聚类质心
    # #
    # 使用样本平均距离的权重定义质心
    cluster_features = compute_label_centers_my(pseudo_labels, features, args.center_sig)

    refinement_pseudo_labels, dist_martix = label_refinement(pseudo_labels, features, aug_features, cluster_features, args.sig, beta=0.8)
    pseudo_labeled_dataset = OrderedDict()
    for i, (fname, label) in enumerate(zip(sorted(seqs_data), pseudo_labels)):
        if label != -1:
            pseudo_labeled_dataset[fname] = label.item()

    refinement_pseudo_labeled_dataset = OrderedDict()
    for i, (fname, label) in enumerate(zip(sorted(seqs_data), pseudo_labels)):
        if label !=-1:
            refinement_pseudo_labeled_dataset[fname] = refinement_pseudo_labels[i]

    # refinement_pseudo_labeled_dataset = OrderedDict()
    # for i, (fname, label, refinement_label) in enumerate(zip(sorted(seqs_data), pseudo_labels, refinement_pseudo_labels)):
    #     if label != -1:
    #         refinement_pseudo_labeled_dataset[fname] = refinement_label

    labels_weight = OrderedDict()
    for i, (fname, label) in enumerate(zip(sorted(seqs_data), pseudo_labels)):
        if label != -1:
            labels_weight[fname] = dist_martix[i][label]

    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    num_features = features.size(-1)
    memory = ClusterMemory(num_features, num_cluster, temp=args.temp,
                           momentum=args.momentum, use_hard=args.use_hard).cuda()
    memory.features = F.normalize(cluster_features, dim=1).cuda()

    # 使用数据增强初始化一个memory bank
    # aug_memory = compute_aug_memory(aug_features, pseudo_labels, seqs_data, args)

    print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

    return pseudo_labels, memory, pseudo_labeled_dataset, refinement_pseudo_labeled_dataset, labels_weight

def compute_aug_memory(features, pseudo_labels, seqs_data, args):
    features = torch.cat([features[f].unsqueeze(0) for f in sorted(seqs_data)], 0)
    cluster_features = compute_label_centers_my(pseudo_labels, features, args.center_sig)
    num_features = features.size(-1)
    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

    memory = ClusterMemory(num_features, num_cluster, temp=args.temp,
                           momentum=args.momentum, use_hard=args.use_hard).cuda()

    memory.features = F.normalize(cluster_features, dim=1).cuda()
    return memory

# def extract_probabilities(features, centers, temp):
#     features = F.normalize(features, p=2, dim=1)
#     print("Extracting prob...")
#     print("################ PROB #################")
#     logits = temp * features.mm(centers.t())
#     prob = F.softmax(logits, 1)
#     print("################ PROB #################")
#
#     return prob


def compute_part_feature(features, part_num):
    all_part_features = []
    part_index = math.ceil(features.size(1)/part_num)
    for i in range(part_num):
        if (i+1)*part_index<=features.size(1):
            part_features = features[:,i*part_index:(i+1)*part_index,:]
        else:
            part_features = features[:,i * part_index:features.size(1), :]
        part_features = part_features.view(part_features.size(0), -1)
        part_features = part_features.to(torch.float32)
        features_array = F.normalize(part_features, dim=1).cpu().numpy()
        all_part_features.append(features_array)

    return all_part_features


def compute_knn(all_part_features, k=15):
    all_part_knn = []
    for i, part_features in enumerate(all_part_features):
        _, feat_nbrs = get_dist_nbr(features=part_features, k=k, knn_method='faiss-gpu')
        all_part_knn.append(feat_nbrs)

    return all_part_knn

def compute_iou(all_part_knn, seq_data):
    all_part_knn = [list(s) for s in zip(*all_part_knn)]
    reliability = pseudo_labeled_dataset = OrderedDict()
    for (i, sample_knn), fname in zip(enumerate(all_part_knn), sorted(seq_data)):
        sample_knn = [set(s) for s in sample_knn]
        intersection = set.intersection(*sample_knn)
        union = set.union(*sample_knn)
        reliability[fname] = len(intersection)/len(union)
    return reliability


def vis_plot(nm_acc, bg_acc, cl_acc, epoch):
    nm_acc = [round(num, 1) for num in nm_acc]
    bg_acc = [round(num, 1) for num in bg_acc]
    cl_acc = [round(num, 1) for num in cl_acc]
    i = range(epoch)
    plt.figure(figsize=(10, 4))
    # 绘制nm准确率曲线
    plt.plot(i, nm_acc, 'b', label='nm_acc')
    for s, label in enumerate(nm_acc):
        plt.text(s, label, label, ha='right', va='bottom')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('NM_ACC')
    plt.legend()

    # 绘制bg准确率曲线
    plt.plot(i, bg_acc, 'r', label='bg_acc')
    for s, label in enumerate(bg_acc):
        plt.text(s, label, label, ha='right', va='bottom')
    plt.ylabel('BG_ACC')
    plt.legend()

    # 绘制cl准确率曲线
    plt.plot(i, cl_acc, 'orange', label='cl_acc')
    for s, label in enumerate(cl_acc):
        plt.text(s, label, label, ha='right', va='bottom')
    plt.ylabel('CL_ACC')
    plt.legend()


    # 显示图形
    plt.show()

def oumvlp_vis_plot(nm_acc, epoch):
    nm_acc = [round(num, 1) for num in nm_acc]
    i = range(epoch)
    plt.figure(figsize=(10, 4))
    # 绘制nm准确率曲线
    plt.plot(i, nm_acc, 'b', label='nm_acc')
    for s, label in enumerate(nm_acc):
        plt.text(s, label, label, ha='right', va='bottom')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('NM_ACC')

    # 显示图形
    plt.show()

def t_sne(features, labels):
    # 画id标签：创建T-SNE模型，将高维数据映射到2维
    features = features.cpu()
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_result = tsne.fit_transform(features)

    # 绘制T-SNE结果
    plt.figure(figsize=(17, 17))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')

    # 在每个数据点附近添加标签
    for i, label in enumerate(labels):
        if label % 3 == 0:
            plt.annotate(label, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=10)

    plt.title("T-SNE Visualization")
    plt.colorbar()
    plt.show()

def typ_t_sne(features, labels, label_typs, person, stride):
    # 画类型标签：创建T-SNE模型，将高维数据映射到2维
    features = features.cpu()
    res = 110 * person
    features = features[0:res:stride]
    labels = labels[0:res:stride]
    label_typs = label_typs[0:res:stride]
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_result = tsne.fit_transform(features)

    # 绘制T-SNE结果
    plt.figure(figsize=(17, 17))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')

    # 在每个数据点附近添加标签
    for i, (label_typ, label) in enumerate(zip(label_typs, labels)):
        # if label % 3 == 0:
        plt.annotate(label_typ, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=10)

    plt.title("T-SNE Visualization")
    plt.colorbar()
    plt.show()

def vie_t_sne(features, labels, label_vies, person, stride):
    # 画角度标签：创建T-SNE模型，将高维数据映射到2维
    features = features.cpu()
    res = 110 * person
    features = features[0:res:stride]
    labels = labels[0:res:stride]
    label_vies = label_vies[0:res:stride]
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_result = tsne.fit_transform(features)

    # 绘制T-SNE结果
    plt.figure(figsize=(17, 17))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')

    # 在每个数据点附近添加标签
    for i, (label_vie, label) in enumerate(zip(label_vies, labels)):
        plt.annotate(label_vie, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=10)

    plt.title("T-SNE Visualization")
    plt.colorbar()
    plt.show()

def get_loader(cfgs: object, data_cfg: object, train: object, inference: object = False,) -> object:

    sampler_cfg = cfgs['evaluator_cfg']['sampler'] if inference else cfgs['trainer_cfg']['sampler']
    dataset = DataSet(data_cfg, train)

    Sampler = get_attr_from([Samplers], sampler_cfg['type'])
    vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
        'sample_type', 'type'])
    sampler = Sampler(dataset, **vaild_args)

    loader =tordata.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=CollateFn(dataset.label_set, sampler_cfg),
        num_workers=data_cfg['num_workers'])
    return loader

def initialization(cfgs, training):
    log_to_file = True
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, log_to_file)

    msg_mgr.log_info(engine_cfg)
    seed = torch.distributed.get_rank()
    init_seeds(seed)

# 实现opengait的获取特征的方法
def inference(model, trainset,  rank):
    model.eval()
    total_size = len(trainset)
    if rank == 0:
        pbar = tqdm(total=total_size, desc='Transforming')  # 创建一个名字为Transforming的进度条
    else:
        pbar = NoOp()
    batch_size = trainset.batch_sampler.batch_size
    rest_size = total_size
    info_dict = Odict()

    count = 0
    for inputs in trainset:
        count += 1
        ipts = model.inputs_pretreament(inputs, use_leg=False)
        with autocast(enabled=model.engine_cfg['enable_float16']):
            retval = model.forward(ipts)
            inference_feat = retval['inference_feat']
            for k, v in inference_feat.items():
                inference_feat[k] = ddp_all_gather(v, requires_grad=False)
            del retval
        for k, v in inference_feat.items():
            inference_feat[k] = ts2np(v)
        info_dict.append(inference_feat)
        rest_size -= batch_size
        if rest_size >= 0:
            update_size = batch_size
        else:
            update_size = total_size % batch_size
            # break
        pbar.update(update_size)

        # if count == 6:
        #     break
    pbar.close()
    for k, v in info_dict.items():
        v = np.concatenate(v)[:total_size]
        info_dict[k] = v
    return info_dict

# 计算腿的特征
def leg_ccr_inference(model, trainset,  rank):
    total_size = len(trainset)
    if rank == 0:
        pbar = tqdm(total=total_size, desc='Extract Features')  # 创建一个名字为Transforming的进度条
    else:
        pbar = NoOp()
    batch_size = trainset.batch_sampler.batch_size
    rest_size = total_size
    features = OrderedDict()
    labels = OrderedDict()
    label_typs = OrderedDict()
    label_vies = OrderedDict()
    nm_labels = OrderedDict()

    for inputs in trainset:
        ipts = model.leg_inputs_pretreament(inputs)
        with autocast(enabled=model.engine_cfg['enable_float16']):
            retval, classifer_out = model.forward(ipts)
            inference_feat = retval['inference_feat']
            outputs = inference_feat['embeddings']
            outputs = ddp_all_gather(outputs, requires_grad=False)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = outputs.to(torch.float32)
            _, real_labels, types, vies, fnames, _ = inputs
            for fname, output, pid in zip(fnames, outputs, real_labels):
                features[fname] = output
                labels[fname] = pid

            # 获取不同穿衣情况的标签用于画图
            for fname, label, typ in zip(fnames, real_labels, types):
                typ = typ[0:2]
                label_typ = "{}{}".format(label, typ)
                label_typs[fname] = label_typ

            # 获取不同角度标签用于画图
            for fname, label, typ, vie in zip(fnames, real_labels, types, vies):
                typ = typ[0:2]
                label_typ = "{}{}{}".format(label, typ, vie)
                label_vies[fname] = label_typ

            # # 只提取nm的特征
            # for fname, typ in zip(fnames, types):
            #     if typ[:2] == 'nm':
            #         features[fname] = output
            #         nm_labels[fname] = typ
        rest_size -= batch_size
        if rest_size >= 0:
            update_size = batch_size
        else:
            update_size = total_size % batch_size
        pbar.update(update_size)
    pbar.close()
    return features.cuda(), labels, label_typs, label_vies

# 实现CCR中的获取features的方法
def ccr_inference(model, trainset,  rank):
    total_size = len(trainset)
    if rank == 0:
        pbar = tqdm(total=total_size, desc='Extract Features')  # 创建一个名字为Transforming的进度条
    else:
        pbar = NoOp()
    batch_size = trainset.batch_sampler.batch_size[0] * trainset.batch_sampler.batch_size[1]
    rest_size = total_size
    features = OrderedDict()
    labels = OrderedDict()
    label_typs = OrderedDict()
    label_vies = OrderedDict()
    nm_labels = OrderedDict()

    for inputs in trainset:
        ipts = model.inputs_pretreament(inputs)
        with autocast(enabled=model.engine_cfg['enable_float16']):
            retval, classifer_out = model.forward(ipts)
            inference_feat = retval['inference_feat']
            outputs = inference_feat['embeddings']
            outputs = ddp_all_gather(outputs, requires_grad=False)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = outputs.to(torch.float32)
            _, real_labels, types, vies, fnames, _ = inputs
            for fname, output, pid in zip(fnames, outputs, real_labels):
                features[fname] = output
                labels[fname] = pid

            # 获取不同穿衣情况的标签用于画图
            for fname, label, typ in zip(fnames, real_labels, types):
                typ = typ[0:2]
                label_typ = "{}{}".format(label, typ)
                label_typs[fname] = label_typ

            # 获取不同角度标签用于画图
            for fname, label, typ, vie in zip(fnames, real_labels, types, vies):
                typ = typ[0:2]
                label_typ = "{}{}{}".format(label, typ, vie)
                label_vies[fname] = label_typ

            # # 只提取nm的特征
            # for fname, typ in zip(fnames, types):
            #     if typ[:2] == 'nm':
            #         features[fname] = output
            #         nm_labels[fname] = typ
        rest_size -= batch_size
        if rest_size >= 0:
            update_size = batch_size
        else:
            update_size = total_size % batch_size
        pbar.update(update_size)
    pbar.close()
    return features, labels, label_typs, label_vies

# 实现CCR中的获取腿的features的方法
def ccr_leg_inference(model, trainset,  rank):
    total_size = len(trainset)
    if rank == 0:
        pbar = tqdm(total=total_size, desc='Extract Features')  # 创建一个名字为Transforming的进度条
    else:
        pbar = NoOp()
    batch_size = trainset.batch_sampler.batch_size
    rest_size = total_size
    features = OrderedDict()
    labels = OrderedDict()
    label_typs = OrderedDict()
    label_vies = OrderedDict()
    nm_labels = OrderedDict()

    for inputs in trainset:
        ipts = model.inputs_pretreament(inputs)
        with autocast(enabled=model.engine_cfg['enable_float16']):
            retval, classifer_out = model.forward(ipts)
            inference_feat = retval['inference_feat']
            outputs = inference_feat['embeddings']
            outputs = ddp_all_gather(outputs, requires_grad=False)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = outputs.to(torch.float32)
            _, real_labels, types, vies, fnames, _ = inputs
            for fname, output, pid in zip(fnames, outputs, real_labels):
                features[fname] = output
                labels[fname] = pid

            # 获取不同穿衣情况的标签用于画图
            for fname, label, typ in zip(fnames, real_labels, types):
                typ = typ[0:2]
                label_typ = "{}{}".format(label, typ)
                label_typs[fname] = label_typ

            # 获取不同角度标签用于画图
            for fname, label, typ, vie in zip(fnames, real_labels, types, vies):
                typ = typ[0:2]
                label_typ = "{}{}{}".format(label, typ, vie)
                label_vies[fname] = label_typ

            # # 只提取nm的特征
            # for fname, typ in zip(fnames, types):
            #     if typ[:2] == 'nm':
            #         features[fname] = output
            #         nm_labels[fname] = typ
        rest_size -= batch_size
        if rest_size >= 0:
            update_size = batch_size
        else:
            update_size = total_size % batch_size
        pbar.update(update_size)
    pbar.close()
    return features, labels, label_typs, label_vies

def split_leg(seq):
    h = seq.size(0)
    w = seq.size(1)
    split =h//2
    seq[0:split,:] = 0
    return seq


@torch.no_grad()
def generate_cluster_features(labels, features):
    '''使用聚类的质心存储'''
    centers = collections.defaultdict(list)  # 当访问字典中不存在的键时会产生一个空列表作为默认值
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers


@torch.no_grad()
def generate_all_features(labels, features):
    '''使用所有样本的特征存储'''
    for i, label in enumerate(labels):
        if label == -1:
            features = torch.cat((features[:i], features[i+1:]))

    return features


def save_ckp(model, cfgs, optimizer, scheduler, epoch):
    dataset_name = cfgs['data_cfg']['dataset_name']
    save_path = osp.join('udr_output/', dataset_name,
                         cfgs['model_cfg']['model'])
    if torch.distributed.get_rank() == 0:
        mkdir(osp.join(save_path, "checkpoints/"))
        save_name = cfgs['evaluator_cfg']['save_name']
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(checkpoint,
                   osp.join(save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, epoch)))

def run_test(model, cfgs):
    rank = torch.distributed.get_rank()
    model.eval()
    model.training = False
    with torch.no_grad():
        info_dict = inference(model, model.test_loader, rank)

    model.train()
    model.training = True
    if rank == 0:
        loader = model.test_loader
        label_list = loader.dataset.label_list
        types_list = loader.dataset.types_list
        views_list = loader.dataset.views_list

        info_dict.update({
            'labels': label_list, 'types': types_list, 'views': views_list})

        if 'eval_func' in cfgs["evaluator_cfg"].keys():
            eval_func = cfgs['evaluator_cfg']["eval_func"]
        else:
            eval_func = 'identification'
        eval_func = getattr(eval_functions, eval_func)
        valid_args = get_valid_args(
            eval_func, cfgs["evaluator_cfg"], ['metric'])
        try:
            dataset_name = cfgs['data_cfg']['test_dataset_name']
        except:
            dataset_name = cfgs['data_cfg']['dataset_name']
        return eval_func(info_dict, dataset_name, **valid_args)

def load_ckpt(model, pretrained_cfgs, cfgs, epoch):
    load_ckpt_strict = cfgs.engine_cfg['restore_ckpt_strict']
    dataset_name = pretrained_cfgs['data_cfg']['dataset_name'] + 'to' + cfgs['data_cfg']['dataset_name']
    save_path = osp.join('udr_output/', dataset_name,
                         cfgs['model_cfg']['model'], cfgs['evaluator_cfg']['save_name'])
    save_path = osp.join(save_path, 'checkpoints/{}-{:0>5}.pt'.format(cfgs['evaluator_cfg']['save_name'], epoch))
    save_name = osp.join('../', save_path)
    checkpoint = torch.load(save_name, map_location=torch.device(
        "cuda", model.device))
    model_state_dict = checkpoint['model']
    if not load_ckpt_strict:
        model.msg_mgr.log_info("-------- Restored Params List --------")
        model.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
            set(model.state_dict().keys()))))

    model.load_state_dict(model_state_dict, strict=load_ckpt_strict)

