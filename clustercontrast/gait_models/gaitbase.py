import os
import argparse
import torch
import torch.nn as nn
from opengait.modeling import models
from opengait.utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
import torch.distributed as dist
import os.path as osp


cfgs = config_loader('../configs/gaitbase/gaitbase_oumvlp.yaml')
engine_cfg = cfgs['evaluator_cfg']
output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
model_cfg = cfgs['model_cfg']
Model = getattr(models, model_cfg['model'])
model = Model(cfgs, training=False)


