import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:   # 判断是否需要计算梯度
            grad_inputs = grad_outputs.mm(ctx.features)  # inputs的梯度 = outputs的梯度*features

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()  # 存在在memory bank中的特征需要归一化

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):  # inputs 是batch的特征， indexes是伪标签， features是memory bank中的存储的特征
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)  # 将 inputs 和 targets 存储在上下文对象 ctx 中，以备反向传播使用。这两个张量将用于计算反向传播时的梯度
        outputs = inputs.mm(ctx.features.t())  # batch的特征和memory bank的特征进行相似度计算

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))  # inputs 是batch的特征， indexes是伪标签， features是memory bank中的存储的特征


class SoftEntropySmooth(nn.Module):
    def __init__(self, epsilon=0.1):
        super(SoftEntropySmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, soft_targets, targets):
        log_probs = self.logsoftmax(inputs)
        # targets = torch.zeros_like(log_probs).scatter_(
        #     1, targets.unsqueeze(1), 1)
        soft_targets = F.softmax(soft_targets, dim=1)
        smooth_targets = (1 - self.epsilon) * targets + \
            self.epsilon * soft_targets
        loss = (- smooth_targets.detach() * log_probs).mean(0).sum()
        return loss

class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features  # 每个特征的维度
        self.num_samples = num_samples   # 聚类的数量

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.soft_ce_loss = SoftEntropySmooth(epsilon=0.4).cuda()

        self.register_buffer('features', torch.zeros(num_samples, num_features)) # 注册一个缓冲区，并且不需要求梯度（反向传播）


    def forward(self, inputs, ema_inputs, targets, refinement_labels=None, use_refine_label=False):

        # if use_ema:
        #     inputs = F.normalize(inputs, dim=1).cuda()

        inputs = F.normalize(inputs, dim=1).cuda()
        ema_inputs = F.normalize(ema_inputs).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
        outputs /= self.temp

        regression = ema_inputs.mm(self.features.t())
        regression /= self.temp

        if use_refine_label:
            refinement_labels = refinement_labels.cuda()
            # loss = F.cross_entropy(outputs, refinement_labels)
            loss = self.soft_ce_loss(outputs, regression, refinement_labels)
        else:
            loss = F.cross_entropy(outputs, targets)
        return loss
