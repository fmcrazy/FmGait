'''
论文名称：Confidence-guided Centroids for Unsupervised Person Re-Identification
论文地址：https://arxiv.org/abs/2211.11921
'''
import copy

import torch
import collections
import numpy as np
import torch.nn.functional as F
from opengait.utils.common import ts2np

def compute_label_centers(labels, features, alpha): # features是张量
    # 检查生成的伪标签是否是连续的
    is_continuous(labels)

    centers = collections.defaultdict(list)  # 当访问字典中不存在的键时会产生一个空列表作为默认值
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    label_centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]  # 一个294维的列表，里面的元素是15872维的张量
    sil_score = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        dist_a = compute_dist_a(features[i], centers[labels[i]])
        nearest_cluster_index= compute_nearest_cluster_index(features[i], labels[i], label_centers)
        dist_b = compute_dist_a(features[i], centers[nearest_cluster_index])
        sil_score[i] = (dist_b - dist_a)/max(dist_a,dist_b)

    # 检查会不会出现对于一个聚类来说其中的所有样本的sil_score都不满足的情况，对于这种情况的样本将其伪标签设置为-1，让其不参与训
    new_labels = label_check(labels, sil_score, alpha)

    new_centers = collections.defaultdict(list)
    for i, label in enumerate(new_labels):
        if label == -1 or sil_score[i] < alpha:
            continue
        new_centers[new_labels[i]].append(features[i])

    new_label_centers = [
        torch.stack(new_centers[idx], dim=0).mean(0) for idx in sorted(new_centers.keys())
    ]

    new_label_centers = torch.stack(new_label_centers, dim=0)
    num_cluster = len(set(new_labels)) - (1 if -1 in new_labels else 0)
    return new_label_centers, new_labels

def compute_label_centers_my(labels, features, sig): # features是张量
    centers = collections.defaultdict(list)  # 当访问字典中不存在的键时会产生一个空列表作为默认值
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    label_centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]  # 一个294维的列表，里面的元素是15872维的张量
    distance = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        dist = compute_dist_a(features[i], centers[labels[i]])
        distance[i] = dist   # distance中存储了样本和聚类中其他样本的平均距离

    cluster_distance = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1 :
            continue
        cluster_distance[label].append(distance[i])

    for idx in sorted(centers.keys()):
        # sigmoid_distance =  [ torch.sigmoid(sig*(dist-min(cluster_distance[idx]))) for dist in cluster_distance[idx] ]
        # cluster_distance = torch.stack(cluster_distance)
        sigmoid_distance = keep_top_k(torch.tensor(cluster_distance[idx]), k=1)
        dist_total = sum(sigmoid_distance)
        norm_distance = [ sigmoid_dist/dist_total for sigmoid_dist in sigmoid_distance]
        centers[idx] = [ f * w for f, w in zip(centers[idx], norm_distance)]
        centers[idx] = sum(centers[idx])
    centers = [
        centers[idx] for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers)
    return centers

def is_continuous(labels):
    for i, label in enumerate(sorted(set(labels))):
        if i-1 !=label:
            print("The generated labels is not continuous")
            label = i

def label_check(labels, sil_score, alpha):
    num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    check_lis = [0] * num_cluster
    for i, label in enumerate(labels):
        if label == -1 or sil_score[i] < alpha:
            continue
        check_lis[label] = 1
    if 0 in check_lis:
        index_lis = [idx for idx, num in enumerate(check_lis) if num == 0 ]
        for i, label in enumerate(labels):
            if label in index_lis:
                labels[i] = -1
            else:
                count = find_index(index_lis, labels[i])
                labels[i] = labels[i] - count
        return labels
    else:
        return labels

def find_index(lst, num):
    if num < lst[0]:
        return 0

    if len(lst) ==1 and num > lst[0]:
        return 1

    for i in range(len(lst) - 1):
        if lst[i] < num < lst[i + 1]:
            return i+1
    return len(lst)

# def compute_dist_a(anchor, clusters):
#     clusters = torch.stack(clusters)
#     dist_a = torch.norm(clusters - anchor, dim=1)
#     avg_dist_a = torch.mean(dist_a)
#     return avg_dist_a

def compute_dist_a(anchor, clusters):
    clusters = torch.stack(clusters)
    dist_a = cosine_similarity(anchor, clusters)
    avg_dist_a = torch.mean(dist_a)
    return avg_dist_a


def cosine_similarity(x, y):
    cos_similarities = F.cosine_similarity(x.unsqueeze(0), y, dim=1)
    # cos_distances = 1 - cos_similarities
    return cos_similarities


def compute_nearest_cluster_index(anchor, label, label_centers):
    label_centers = torch.stack(label_centers)
    distance = cosine_similarity(anchor, label_centers)
    distance[label] = float('inf')
    index = torch.argmin(distance).item()

    return index

def compute_dist_cluster(anchor, label_centers, sig):
    distance = cosine_similarity(anchor, label_centers)
    # sigmoid_distance = torch.sigmoid(sig*(distance-torch.mean(distance)))
    # # sigmoid_distance = [ torch.sigmoid(dist) for dist in distance ]
    # sigmoid_distance = keep_top_k(distance, k=10)
    sigmoid_distance = torch.sigmoid(30 * (distance - torch.mean(distance)))
    norm_distance = sigmoid_distance/torch.sum(sigmoid_distance)

    return norm_distance

def compute_aug_dist_cluster(anchor, label_centers, sig):
    distance = cosine_similarity(anchor, label_centers)
    # sigmoid_distance = torch.sigmoid(sig*(distance-torch.mean(distance)))
    # # sigmoid_distance = [ torch.sigmoid(dist) for dist in distance ]
    sigmoid_distance = keep_top_k(distance, k=2)
    # sigmoid_distance = torch.sigmoid(30 * (sigmoid_distance - torch.mean(sigmoid_distance)))
    norm_distance = sigmoid_distance/torch.sum(sigmoid_distance)

    return norm_distance


def keep_top_k(tensor, k):
    # 找到张量中的最大值及其索引
    values, indices = torch.topk(tensor, k)

    # 将除了最大的 k 个元素外的所有元素设为 0
    result = torch.zeros_like(tensor)
    # indices = torch.unravel_index(indices, tensor.shape)
    result[indices] = values

    return result

def compute_dist_martix(labels, label_centers, features, sig):
    labels_weight = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        dist_cluster = compute_dist_cluster(features[i], label_centers, sig)
        labels_weight[i] = dist_cluster
    # dist_martix = torch.stack(dist_martix, dim=0)

    return labels_weight

def compute_aug_dist_martix(labels, label_centers, features, sig):
    labels_weight = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        dist_cluster = compute_aug_dist_cluster(features[i], label_centers, sig)
        labels_weight[i] = dist_cluster
    # dist_martix = torch.stack(dist_martix, dim=0)

    return labels_weight

def extract_probabilities(features, centers, temp=30):
    features = F.normalize(features, dim=1)
    logits = temp*features.mm(centers.t())
    prob = F.softmax(logits, 1)
    return prob

def label_refinement(labels, features, aug_features, label_centers, sig, beta):
    dist_martix = compute_dist_martix(labels, label_centers, features, sig)
    aug_labels = compute_dist_martix(labels, label_centers, aug_features, sig)
    # 将labels变为独热编码
    N = len(labels)
    C = len(set(labels)) - (1 if -1 in labels else 0)
    onehot_labels = torch.full(size=(N, C), fill_value=0).cuda()
    labels_lis = collections.defaultdict(list)
    for i in range(N):
        index = labels[i]
        if index != -1:
            onehot_labels[i][index] = 1
    for i, label in enumerate(labels):
        # labels_lis[i] = beta * onehot_labels[i] + (1 - beta) * dist_martix[i]
        if label == -1:
            continue
        labels_lis[i] = 0.4 * onehot_labels[i] + 0.6 * dist_martix[i]
    # refinement_pseudo_labels = torch.stack(labels_lis)
    # refinement_pseudo_labels = ts2np(labels_lis)


    return labels_lis, dist_martix