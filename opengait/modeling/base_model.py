"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import copy

import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from torch.nn import init

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from opengait.modeling import backbones
from opengait.modeling.loss_aggregator import LossAggregator
from opengait.data.transform import get_transform
from opengait.data.collate_fn import CollateFn
from opengait.data.dataset import DataSet
import opengait.data.sampler as Samplers
from opengait.utils import Odict, mkdir, ddp_all_gather
from opengait.utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from opengait.evaluation import evaluator as eval_functions
from opengait.utils import NoOp
from opengait.utils import get_msg_mgr
from collections import OrderedDict

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training, num_class=0):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.num_class = num_class
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:  # 表示使用混合精度
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()

        self.msg_mgr.log_info(cfgs['data_cfg'])
        self.inference_train_loader = self.get_loader(cfgs['data_cfg'], train=True, inference=True)
        self.train_loader = self.get_loader(cfgs['data_cfg'], train=True, inference=False)
        self.test_loader = self.get_loader(cfgs['data_cfg'], train=False, inference=True)

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))  # 将模型和数据移动到指定设备

        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)


    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, train=True, inference=True):
        # sampler_cfg = self.cfgs['evaluator_cfg']['sampler'] if inference  else self.cfgs['trainer_cfg']['sampler']
        if inference and train:
            sampler_cfg = self.cfgs['inference_cfg']['sampler']
        if not inference and train:
            sampler_cfg = self.cfgs['trainer_cfg']['sampler']
        if inference and not train:
            sampler_cfg = self.cfgs['evaluator_cfg']['sampler']

        dataset = DataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader


    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']
        # save_name = osp.join('../', save_name)
        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        # checkpoint = torch.load(save_name, map_location=torch.device(
        #     "cpu"))   # 修改：将cuda改成cpu
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def split(self, fra):
        fra[:,:32,:] = 0
        return fra

    def inputs_pretreament(self, inputs, use_leg):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, fanme_batch, seqL_batch = inputs
        trf_cfgs = self.engine_cfg['transform']
        seq_trfs = get_transform(trf_cfgs)
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        if use_leg:
            seqs = [np2var(np.asarray([self.split(trf(fra)) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]
        else:
            seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def q_inputs_pretreament(self, inputs, use_leg, length):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, fanme_batch, seqL_batch = inputs
        trf_cfgs = self.engine_cfg['transform']
        seq_trfs = get_transform(trf_cfgs)
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        # batch_size = int(len(seqs_batch[0])/2)
        if use_leg:
            seqs = [np2var(np.asarray([self.split(trf(fra)) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]
        else:
            seqs = [np2var(np.asarray([trf(fra) for fra in seq[0:length]]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch[0:length]
        vies = vies_batch[:length]
        labs_batch = labs_batch[0:length]

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def k_inputs_pretreament(self, inputs, use_leg, length):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, fanme_batch, seqL_batch = inputs
        # trf_cfgs = self.engine_cfg['transform']
        trf_cfgs = [{'type': 'DA4GaitSSB'}]
        seq_trfs = get_transform(trf_cfgs)
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        if use_leg:
            seqs = [np2var(np.asarray([self.split(trf(fra)) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]
        else:
            seqs = [np2var(np.asarray([trf(fra) for fra in seq[length:]]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch[length:]
        vies = vies_batch[length:]
        labs_batch = labs_batch[length:]

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def aug_inputs_pretreament(self, inputs, use_leg):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, fanme_batch, seqL_batch = inputs
        trf_cfgs = [{'type':'DA4GaitSSB'}]
        seq_trfs = get_transform(trf_cfgs)
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        if use_leg:
            seqs = [np2var(np.asarray([self.split(trf(fra)) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]
        else:
            seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def split_leg(self, seq):
        h = seq.size(0)
        w = seq.size(1)
        split = h // 2
        seq[0:split, :] = 0
        return seq

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')  # 创建一个名字为Transforming的进度条
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs,use_leg=False)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
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
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    def ccr_inference(self, rank, use_leg=False, use_aug=False):
        total_size = len(self.inference_train_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Extract Features')
        else:
            pbar = NoOp()
        batch_size = self.inference_train_loader.batch_sampler.batch_size
        rest_size = total_size
        features = OrderedDict()
        aug_features = OrderedDict()
        labels = OrderedDict()
        label_typs = OrderedDict()
        part_features = OrderedDict()

        for inputs in self.inference_train_loader:
            if use_aug:
                aug_inputs = copy.deepcopy(inputs)
                aug_features = self.aug_pre(aug_inputs, aug_features, use_leg)
            ipts = self.inputs_pretreament(inputs, use_leg)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                del retval
                outputs = inference_feat['embeddings']
                part_outputs = outputs
                part_outputs = part_outputs.to(torch.float32)
                out_shape = [outputs.size(1), outputs.size(2)]
                # outputs = ddp_all_gather(outputs, requires_grad=False)
                outputs = outputs.view(outputs.size(0), -1)
                outputs = outputs.to(torch.float32)
                _, real_labels, types, vies, fnames, _ = inputs
                for fname, output, pid in zip(fnames, outputs, real_labels):
                    features[fname] = output
                    labels[fname] = pid

                for fname, output, pid in zip(fnames, part_outputs, real_labels):
                    part_features[fname] = output
                    labels[fname] = pid

                # 根据穿衣情况开始画图
                for fname, label, typ in zip(fnames, real_labels, types):
                    typ = typ[0:2]
                    label_typ = "{}{}".format(label, typ)
                    label_typs[fname] = label_typ

            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        return features, out_shape, part_features

    def q_ccr_inference(self, rank, use_leg=False, use_aug=True):
        total_size = len(self.inference_train_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Extract Features')
        else:
            pbar = NoOp()
        batch_size = self.inference_train_loader.batch_sampler.batch_size
        rest_size = total_size
        features = OrderedDict()
        aug_features = OrderedDict()
        labels = OrderedDict()
        label_typs = OrderedDict()

        for inputs in self.inference_train_loader:
            if use_aug:
                aug_inputs = copy.deepcopy(inputs)
                aug_features = self.aug_pre(aug_inputs, aug_features, use_leg)
            ipts = self.q_inputs_pretreament(inputs, use_leg)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                del retval
                outputs = inference_feat['embeddings']
                out_shape = [outputs.size(1), outputs.size(2)]
                # outputs = ddp_all_gather(outputs, requires_grad=False)
                outputs = outputs.view(outputs.size(0), -1)
                outputs = outputs.to(torch.float32)
                _, real_labels, types, vies, fnames, _ = inputs
                for fname, output, pid in zip(fnames, outputs, real_labels):
                    features[fname] = output
                    labels[fname] = pid

                # 根据穿衣情况开始画图
                for fname, label, typ in zip(fnames, real_labels, types):
                    typ = typ[0:2]
                    label_typ = "{}{}".format(label, typ)
                    label_typs[fname] = label_typ

            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        return features, aug_features, out_shape

    def k_ccr_inference(self, rank, use_leg=False, use_aug=True):
        total_size = len(self.inference_train_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Extract Features')
        else:
            pbar = NoOp()
        batch_size = self.inference_train_loader.batch_sampler.batch_size
        rest_size = total_size
        features = OrderedDict()
        aug_features = OrderedDict()
        labels = OrderedDict()
        label_typs = OrderedDict()

        for inputs in self.inference_train_loader:
            if use_aug:
                aug_inputs = copy.deepcopy(inputs)
                aug_features = self.aug_pre(aug_inputs, aug_features, use_leg)
            ipts = self.k_inputs_pretreament(inputs, use_leg)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                del retval
                outputs = inference_feat['embeddings']
                out_shape = [outputs.size(1), outputs.size(2)]
                # outputs = ddp_all_gather(outputs, requires_grad=False)
                outputs = outputs.view(outputs.size(0), -1)
                outputs = outputs.to(torch.float32)
                _, real_labels, types, vies, fnames, _ = inputs
                for fname, output, pid in zip(fnames, outputs, real_labels):
                    features[fname] = output
                    labels[fname] = pid

                # 根据穿衣情况开始画图
                for fname, label, typ in zip(fnames, real_labels, types):
                    typ = typ[0:2]
                    label_typ = "{}{}".format(label, typ)
                    label_typs[fname] = label_typ

            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        return features, aug_features, out_shape

    def aug_pre(self, inputs, features, use_leg=False):
        ipts = self.aug_inputs_pretreament(inputs, use_leg)
        with autocast(enabled=self.engine_cfg['enable_float16']):
            retval = self.forward(ipts)
            inference_feat = retval['inference_feat']
            del retval
            outputs = inference_feat['embeddings']
            out_shape = [outputs.size(1), outputs.size(2)]
            outputs = outputs.view(outputs.size(0), -1)
            outputs = outputs.to(torch.float32)
            _, real_labels, types, vies, fnames, _ = inputs
            for fname, output in zip(fnames, outputs):
                features[fname] = output
        return features

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for inputs in model.train_loader:
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = BaseModel.run_test(model)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        model.eval()
        model.training = False
        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        model.train()
        model.training = True
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
                eval_func = model.cfgs['evaluator_cfg']["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, model.cfgs["evaluator_cfg"], ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            return eval_func(info_dict, dataset_name, **valid_args)


