import torch.nn.functional as F
import torch
from torch.distributions import Categorical

'''
论文名称：Refining Pseudo Labels with Clustering Consensus over Generations for Unsupervised Object Re-identification
论文地址：https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Refining_Pseudo_Labels_With_Clustering_Consensus_Over_Generations_for_Unsupervised_CVPR_2021_paper.pdf
代码地址：https://github.com/2han9x1a0release/RLCC
'''
class LabelRefine():
    def __init__(self):
        pass

    def update_label(self, epoch, pseudo_label, label_center, features, model):
        self.epoch = epoch
        self.pseudo_label = pseudo_label
        self.label_center = label_center

        if self.epoch==0:
            self.pseudo_labels_pervious = torch.tensor(self.pseudo_label)
            self.pseudo_centers_pervious = self.label_center.clone().detach().requires_grad_(False)
            self.pseudo_centers_pervious = torch.tensor(self.pseudo_centers_pervious)
            self.sample_soft_labels = torch.zeros(self.pseudo_labels_pervious.size(0), self.pseudo_centers_pervious.size(0))
            self.sample_soft_labels = self.sample_soft_labels.to(model.device)

        else:
            probs_perv = extract_probabilities(features, self.pseudo_centers_pervious, 30).cuda()
            # hard_labels = compute_hard_label(self.pseudo_labels_pervious, self.pseudo_centers_pervious)

            # if epoch!=1:
            #     N = probs_perv.size(0)  # 样本的数量
            #     C = probs_perv.size(1)  # 类别的数量
            #     onehot_labels = torch.full(size=(N, C), fill_value=0)  # 创建一个大小为（N,C）的张量，并用0填充
            #     onehot_labels = onehot_labels.to(model.device)
            #     for i in range(N):
            #         index = self.pseudo_labels_pervious[i]
            #         if index != -1:
            #             onehot_labels[i][index] = 1
            #     probs_perv = 0.9 * onehot_labels + (1.0 - 0.9) * probs_perv

            self.pseudo_labels_current = torch.tensor(self.pseudo_label)
            self.pseudo_centers_current = self.label_center.clone().detach().requires_grad_(False)
            iou_mat = compute_label_iou_matrix(self.pseudo_labels_pervious, self.pseudo_labels_current)
            norm_iou_mat = (iou_mat.t() / iou_mat.t().sum(0)).t()
            # norm_iou_mat = norm_iou_mat.to(torch.long)
            # probs_perv = probs_perv.to(torch.long)
            # probs_perv = probs_perv.to(model.device)
            norm_iou_mat = norm_iou_mat.to(model.device)
            all_softlabels = probs_perv.mm(norm_iou_mat)

            beta = 0.5
            prob_soft_labels = all_softlabels.cuda()
            hard_iou_labels = compute_sample_softlabels(
                self.pseudo_labels_pervious, self.pseudo_labels_current, "iou", "original"
            ).cuda()
            self.sample_soft_labels = beta * hard_iou_labels + (1.0 - beta) * prob_soft_labels
            self.sample_soft_labels = self.sample_soft_labels.to(model.device)
            self.pseudo_labels_pervious = self.pseudo_labels_current
            self.pseudo_centers_pervious = self.pseudo_centers_current

def extract_probabilities(features, centers, temp):
    features = F.normalize(features, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)
    print("Extracting prob...")
    print("################ PROB #################")
    logits = temp * features.mm(centers.t())
    prob = F.softmax(logits, 1)
    print("################ PROB #################")

    return prob

@torch.no_grad()
def compute_label_transform_matrix(labels_t1, labels_t2, a, b):
    assert labels_t1.size(1) == labels_t2.size(1) # make sure sample num are equal
    sample_num = labels_t1.size(1)
    class_num_t1 = labels_t1.unique().size(0)
    class_num_t2 = labels_t2.unique().size(0)
    if a > 0 :
        class_num_t1 = class_num_t1 -1
    if b > 0 :
        class_num_t2 = class_num_t2 -1
    dual_labels = torch.cat((labels_t1, labels_t2),0).t()
    label_tran_mat = torch.zeros(class_num_t1, class_num_t2)
    for x in dual_labels:
        label_tran_mat[x[0].item(), x[1].item()] += 1
    return label_tran_mat


@torch.no_grad()
def compute_inclass_distributiions(labels_t1, labels_t2, dis_type="original"):
    label_tran_mat = compute_label_transform_matrix(labels_t1, labels_t2)
    if dis_type=="softmax":
    	return torch.nn.functional.softmax(label_tran_mat, 0)
    else:
        return label_tran_mat / label_tran_mat.sum(0)


# Method 1 as class weights
@torch.no_grad()
def compute_class_stablization_by_entropy(labels_t1, labels_t2):
    distributions = compute_inclass_distributiions(labels_t1, labels_t2, dis_type="softmax")
    return Categorical(probs = distributions.t()).entropy()

@torch.no_grad()
def compute_hard_label(labels, labels_center):
    N = labels.size(0)
    C = labels_center.size(0)
    onehot_labels = torch.full(size=(N, C), fill_value=0)
    for i in range(N):
        index = labels[i]
        if index != -1:
            onehot_labels[i][index] = 1
    # onehot_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1)
    return onehot_labels

@torch.no_grad()
def compute_label_iou_matrix(labels_t1, labels_t2):
    count_of_minus_ones = sum(1 for x in labels_t1 if x == -1)
    count_of_minus_twos = sum(1 for x in labels_t2 if x == -1)
    labels_t1 = labels_t1.unsqueeze(0)
    labels_t2 = labels_t2.unsqueeze(0)
    class_num_t1 = labels_t1.unique().size(0)
    class_num_t2 = labels_t2.unique().size(0)
    if count_of_minus_ones > 0:
        class_num_t1 = class_num_t1 - 1

    if count_of_minus_twos > 0:
        class_num_t2 = class_num_t2 -1
    dual_labels = torch.cat((labels_t1, labels_t2),0).t()
    label_union_mat_1 = torch.zeros(class_num_t1, class_num_t2)
    label_union_mat_2 = torch.zeros(class_num_t1, class_num_t2).t()
    for x in dual_labels:
        if x[0] != -1:
            label_union_mat_1[x[0].item()] += 1
        if x[1] != -1:
            label_union_mat_2[x[1].item()] += 1
    label_inter_mat = compute_label_transform_matrix(labels_t1, labels_t2, count_of_minus_ones, count_of_minus_twos)
    label_union_mat = label_union_mat_1 + label_union_mat_2.t() - label_inter_mat
    return label_inter_mat / label_union_mat




@torch.no_grad()
def compute_sample_weights(labels_t1, labels_t2):
    ioumat = torch.nn.functional.softmax(compute_label_iou_matrix(labels_t1, labels_t2), 1)
    return torch.index_select(torch.index_select(ioumat, 0, labels_t1[0])[0], 0, labels_t2[0])


@torch.no_grad()
def compute_class_softlabels(labels_t1, labels_t2, matr_type="trans", distr_type="original"):
    assert labels_t1.size(0) == labels_t2.size(0) # make sure sample num are equal
    if matr_type == "trans":
        matr = compute_label_transform_matrix(labels_t1, labels_t2)
    else:
        matr = compute_label_iou_matrix(labels_t1, labels_t2)
    if distr_type=="original":
        return (matr.t() / matr.t().sum(0)).t()
    else:
        return torch.nn.functional.softmax(matr, 1)

@torch.no_grad()
def extract_probabilities(features, centers, temp):
    features = F.normalize(features, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)
    logits = temp * features.mm(centers.t())
    prob = F.softmax(logits, 1)
    return prob

@torch.no_grad()
def compute_sample_softlabels(labels_t1, labels_t2, matr_type="trans", distr_type="original"):
    class_softlabels = compute_class_softlabels(labels_t1, labels_t2, matr_type, distr_type)
    return torch.index_select(class_softlabels, 0, labels_t1[0])