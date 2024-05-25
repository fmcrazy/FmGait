import torch
import math
def compute_part_feature(features, part_num):
    all_part_features = []
    part_index = math.ceil(features.size(1)/part_num)
    for i in range(part_num):
        part_features = torch.zeros_like(features).cuda()
        if (i+1)*part_index<=features.size(1):
            part_features[:,i*part_index:(i+1)*part_index,:] = features[:,i*part_index:(i+1)*part_index,:]
        else:
            part_features[:,i*part_index:(i+1)*part_index,:] = features[:,i * part_index:features.size(1), :]
        part_features = part_features.view(part_features.size(0), -1)
        part_features = part_features.to(torch.float32)
        all_part_features.append(part_features)
    all_part_features = torch.stack(all_part_features)
    return all_part_features