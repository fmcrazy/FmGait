# import sys
# sys.path.append('/code/lxl/Gait/OpenGait/opengait')

from .common import get_ddp_module, ddp_all_gather
from .common import Odict, Ntuple
from .common import get_valid_args
from .common import is_list_or_tuple, is_bool, is_str, is_list, is_dict, is_tensor, is_array, config_loader, init_seeds, handler, params_count
from .common import ts2np, ts2var, np2var, list2var
from .common import mkdir, clones
from .common import MergeCfgsDict
from .common import get_attr_from
from .common import NoOp
from .msg_manager import get_msg_mgr



# import os
#
# # 获取当前模块的文件路径
# current_module_path = os.path.abspath(__file__)
# print("Current module path:", current_module_path)
#
# # 获取当前模块的包路径（去掉最后一个文件名）
# current_package_path = os.path.dirname(current_module_path)
# print("Current package path:", current_package_path)
