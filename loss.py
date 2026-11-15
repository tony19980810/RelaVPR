from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from torch import nn
import torch
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
import torch.nn.functional as F
#  The loss function call (this method will be called at each training iteration)



def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss
class LinearDistillLoss2(nn.Module):
    def __init__(self, student_dim=3072, teacher_dim=4096):
        super().__init__()
        self.student_to_teacher = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_feat, teacher_feat):
       
        student_expanded = self.student_to_teacher(student_feat)
       
        return F.mse_loss(student_expanded, teacher_feat)

def CKALoss(SH,TH):
    eps=1e-8
    dT = TH.size(-1)
    dS = SH.size(-1)
    SH = SH.view(-1, dS).to(SH.device, torch.float64)
    TH = TH.view(-1, dT).to(SH.device, torch.float64)


    SH = SH - SH.mean(0, keepdim=True)
    TH = TH - TH.mean(0, keepdim=True)

    num = torch.norm(SH.t().matmul(TH), 'fro')
    den1 = torch.norm(SH.t().matmul(SH), 'fro') + eps
    den2 = torch.norm(TH.t().matmul(TH), 'fro') + eps

    return 1 - num / torch.sqrt(den1 * den2)

class LinearDistillLoss(nn.Module):
    def __init__(self, student_dim=3072, teacher_dim=6144):
        super().__init__()
        self.student_to_teacher = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_feat, teacher_feat):
       
        student_expanded = self.student_to_teacher(student_feat)
      
        return F.mse_loss(student_expanded, teacher_feat)








def distillation_loss(student_feature, teacher_feature):
    # 对教师特征进行平均池化，使其维度与学生特征一致

    # 计算蒸馏损失，这里使用均方误差损失
    loss = F.mse_loss(student_feature, teacher_feature)
    return loss


class RKDConfidenceHardLoss_final(nn.Module):
    def __init__(
        self,
        student_dim=3072,
        teacher_dim=6144,
        topk_ratio=0.1,       # 最近负样本比例
        tau=0.03,
        low_th=0.3,           # 保留接口；当前未使用
        sampling_ratio=0.9,   # 论文中的 rho
        hard_ratio=0.1,
        lambda_hard=0.001,
        debug=False,
    ):
        super().__init__()

        self.topk_ratio = topk_ratio
        self.tau = tau
        self.low_th = low_th
        self.sampling_ratio = sampling_ratio
        self.hard_ratio = hard_ratio
        self.lambda_hard = lambda_hard
        self.debug = debug

        self.student_to_teacher = nn.Linear(student_dim, teacher_dim)

    def forward(self, student, teacher, labels):
        B = labels.size(0)
        if B < 2:
            raise ValueError("RKD requires at least two samples.")
        if not (0 < self.sampling_ratio <= 1):
            raise ValueError("sampling_ratio must be in (0, 1].")

        eye = torch.eye(B, dtype=torch.bool, device=labels.device)
        labels_ = labels.view(-1, 1)
        pos_mask = (labels_ == labels_.t()) & ~eye
        neg_mask = labels_ != labels_.t()

        # ===============================
        # Teacher distances and confidence
        # ===============================
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[~eye].mean().clamp_min(1e-12)
            t_d = t_d / mean_td

            k = max(1, int(self.topk_ratio * B))
            if (neg_mask.sum(dim=1) < k).any():
                raise ValueError(
                    f"Each anchor needs at least {k} negative samples."
                )

            t_d_neg = t_d.masked_fill(~neg_mask, float("inf"))
            topk_neg = torch.topk(
                t_d_neg, k=k, largest=False, dim=1
            ).values
            neg_ref = topk_neg.mean(dim=1, keepdim=True)

            pos_confidence = torch.sigmoid((neg_ref - t_d) / self.tau)
            neg_confidence = torch.sigmoid((t_d - neg_ref) / self.tau)

            confidence = torch.where(
                pos_mask,
                pos_confidence,
                torch.where(
                    neg_mask,
                    neg_confidence,
                    torch.zeros_like(t_d),
                ),
            )

           
            m = max(1, int(self.sampling_ratio * (B - 1)))
            sampling_weights = confidence + (~eye).to(confidence.dtype) * 1e-12
            sampling_weights = sampling_weights.masked_fill(eye, 0.0)

            sampled_idx = torch.multinomial(
                sampling_weights,
                num_samples=m,
                replacement=False,
            )  # (B, m)

        # ===============================
        # Student distances
        # ===============================
        d = pdist(student, squared=False)
        mean_d = d[~eye].mean().clamp_min(1e-12)
        d = d / mean_d

        # ===============================
        # URSF: sampled, unweighted RKD
        # ===============================
        loss_mat = F.smooth_l1_loss(d, t_d, reduction="none")
        loss_rkd = loss_mat.gather(dim=1, index=sampled_idx).mean()

        # ===============================
        # HAA: hard anchor selection
        # ===============================
        with torch.no_grad():
            diff = torch.abs(d - t_d)
            hardness = diff.mean(dim=1)

            k_hard = max(1, int(self.hard_ratio * B))
            hard_idx = torch.topk(
                hardness, k=k_hard, largest=True
            ).indices

        student_hard = student[hard_idx]
        teacher_hard = teacher[hard_idx]
        
        student_proj = self.student_to_teacher(student_hard)
        student_proj = F.normalize(student_proj, dim=1)
        teacher_hard = F.normalize(teacher_hard, dim=1)

        cos_sim = F.cosine_similarity(
            student_proj, teacher_hard, dim=1
        )
        loss_cos = 1 - cos_sim.mean()

        loss = loss_rkd + self.lambda_hard * loss_cos

       

        return loss

def pdist_batch(x, squared=False, eps=1e-12):
    """
    x: (B, N, C)
    return: (B, N, N)
    """
    x_square = x.pow(2).sum(dim=2, keepdim=True)           # (B, N, 1)
    prod = torch.bmm(x, x.transpose(1, 2))                 # (B, N, N)

    res = (x_square + x_square.transpose(1, 2) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    # zero diagonal
    B, N, _ = res.shape
    eye = torch.eye(N, device=x.device).bool().unsqueeze(0)
    res = res.masked_fill(eye, 0)

    return res









def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res






        




def Normalize(feat):
    return (feat - feat.mean(dim=0)) / (feat.std(dim=0, unbiased=False) + 1e-5)
    









def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss





