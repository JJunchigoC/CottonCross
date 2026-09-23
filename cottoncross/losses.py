"""Training objectives.

The supervised terms all read labels that the label-first generator produced for free.
`TARGET_*` names the packing used by `dataset.SynthDataset`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import CH_CENTER, CH_EMBED, CH_JOINT, CH_MASK, CH_MOMENTS, CH_OVERLAP

T_MASK = slice(0, 1)
T_CENTER = slice(1, 2)
T_JOINT = slice(2, 3)
T_MOMENTS = slice(3, 7)
T_ORIVALID = slice(7, 8)
T_OVERLAP = slice(8, 9)
T_INSTANCE = slice(9, 10)
TARGET_CHANNELS = 10

DEFAULT_WEIGHTS = dict(mask=1.0, center=1.0, joint=1.0, topology=0.30, orientation=0.15,
                       overlap=0.30, embedding=0.20, deep=0.20)


# ------------------------------------------------------------------ topology (clDice)
def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x, iterations=6):
    def opening(v):
        return F.max_pool2d(soft_erode(v), 3, 1, 1)

    sk = F.relu(x - opening(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - opening(x))
        sk = sk + F.relu(delta - sk * delta)
    return sk


def dice_loss(p, y, eps=1.0):
    dims = (1, 2, 3)
    return (1 - (2 * (p * y).sum(dims) + eps) / (p.sum(dims) + y.sum(dims) + eps)).mean()


def cldice_loss(p, y, iterations=6):
    """Topology-preserving tubular loss (Shit et al., CVPR 2021)."""
    sp, sy = soft_skeleton(p, iterations), soft_skeleton(y, iterations)
    dims = (1, 2, 3)
    precision = ((sp * y).sum(dims) + 1) / (sp.sum(dims) + 1)
    recall = ((sy * p).sum(dims) + 1) / (sy.sum(dims) + 1)
    return (1 - 2 * precision * recall / (precision + recall + 1e-6)).mean()


# ------------------------------------------------------------------ instance embedding
def embedding_loss(embed, instance_id, centerline, delta_var=0.3, delta_dist=1.2,
                   max_pixels=768):
    """Discriminative embedding loss over the free synthetic instance labels.

    Pixels on the same fiber are pulled inside a `delta_var` ball around their centroid;
    different fibers are pushed `delta_dist` apart.  At inference the two ports of one
    crossing are paired partly by comparing these embeddings, which is what lets the
    linker follow a fiber through an overlap instead of guessing from angles alone.
    """
    device = embed.device
    total = embed.new_zeros(())
    used = 0
    for b in range(embed.shape[0]):
        ids = instance_id[b, 0]
        on = (centerline[b, 0] > 0.5) & (ids > 0)
        labels = torch.unique(ids[on])
        if labels.numel() < 1:
            continue
        means, terms = [], []
        for k in labels:
            sel = on & (ids == k)
            idx = sel.nonzero(as_tuple=False)
            if idx.shape[0] < 4:
                continue
            if idx.shape[0] > max_pixels:
                idx = idx[torch.randperm(idx.shape[0], device=device)[:max_pixels]]
            vecs = embed[b, :, idx[:, 0], idx[:, 1]].t()          # (n, d)
            mu = F.normalize(vecs.mean(0, keepdim=True), dim=1)
            means.append(mu)
            terms.append(F.relu((vecs - mu).norm(dim=1) - delta_var).square().mean())
        if not terms:
            continue
        variance = torch.stack(terms).mean()
        loss = variance
        if len(means) > 1:
            mu = torch.cat(means, 0)
            distance = torch.cdist(mu, mu)
            off = ~torch.eye(len(means), dtype=torch.bool, device=device)
            loss = loss + F.relu(delta_dist - distance[off]).square().mean()
        total = total + loss
        used += 1
    return total / max(used, 1)


# ------------------------------------------------------------------ supervised bundle
def supervised_loss(logits, target, weights=None, aux=None):
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    mask_t = target[:, T_MASK]
    center_t = target[:, T_CENTER]
    joint_t = target[:, T_JOINT]

    mask_bce = F.binary_cross_entropy_with_logits(
        logits[:, CH_MASK], mask_t, pos_weight=logits.new_tensor(8.0))
    center_bce = F.binary_cross_entropy_with_logits(
        logits[:, CH_CENTER], center_t, pos_weight=logits.new_tensor(12.0))
    mask_p = torch.sigmoid(logits[:, CH_MASK])
    center_p = torch.sigmoid(logits[:, CH_CENTER])
    l_mask = mask_bce + dice_loss(mask_p, mask_t)
    l_center = center_bce + 0.5 * dice_loss(center_p, (center_t > 0.5).float())

    # Gaussian shoulders are soft targets, so the heatmap uses a weighted soft BCE.
    raw = F.binary_cross_entropy_with_logits(logits[:, CH_JOINT], joint_t, reduction="none")
    l_joint = (raw * (1.0 + 50.0 * joint_t)).mean()

    parts = dict(mask=float(l_mask.detach()), center=float(l_center.detach()),
                 joint=float(l_joint.detach()))
    total = w["mask"] * l_mask + w["center"] * l_center + w["joint"] * l_joint

    if w["topology"] > 0:
        l_topo = cldice_loss(mask_p, mask_t)
        total = total + w["topology"] * l_topo
        parts["topology"] = float(l_topo.detach())
    if w["orientation"] > 0:
        valid = target[:, T_ORIVALID]
        diff = (torch.tanh(logits[:, CH_MOMENTS]) - target[:, T_MOMENTS]).square()
        l_ori = (diff * valid).sum() / (4 * valid.sum() + 1)
        total = total + w["orientation"] * l_ori
        parts["orientation"] = float(l_ori.detach())
    if w["overlap"] > 0:
        cls = target[:, T_OVERLAP].squeeze(1).long().clamp(0, 2)
        l_ovl = F.cross_entropy(logits[:, CH_OVERLAP], cls,
                                weight=logits.new_tensor([0.2, 1.0, 6.0]))
        total = total + w["overlap"] * l_ovl
        parts["overlap"] = float(l_ovl.detach())
    if w["embedding"] > 0:
        embed = F.normalize(logits[:, CH_EMBED], dim=1)
        l_emb = embedding_loss(embed, target[:, T_INSTANCE], (center_t > 0.5).float())
        total = total + w["embedding"] * l_emb
        parts["embedding"] = float(l_emb.detach())
    if w["deep"] > 0 and aux and aux.get("deep"):
        l_deep = logits.new_zeros(())
        for z in aux["deep"]:
            size = z.shape[-2:]
            m = F.adaptive_max_pool2d(mask_t, size)
            c = F.adaptive_max_pool2d(center_t, size)
            j = F.adaptive_max_pool2d(joint_t, size)
            l_deep = l_deep + F.binary_cross_entropy_with_logits(
                z, torch.cat([m, c, j], 1),
                pos_weight=logits.new_tensor([8.0, 12.0, 30.0]).view(1, 3, 1, 1))
        l_deep = l_deep / len(aux["deep"])
        total = total + w["deep"] * l_deep
        parts["deep"] = float(l_deep.detach())

    parts["supervised"] = float(total.detach())
    return total, parts


# ------------------------------------------------------------------ unlabelled real
def consistency_loss(student_logits, teacher_logits, confident, orientation_weight=0.05):
    """Balanced foreground/background consistency on confidently pseudo-labelled pixels."""
    tp = torch.sigmoid(teacher_logits[:, 0:3])
    raw = F.binary_cross_entropy_with_logits(student_logits[:, 0:3], tp, reduction="none")
    fg = confident & (tp > 0.5)
    bg = confident & (tp <= 0.5)
    zero = raw.sum() * 0
    loss = 0.5 * (raw[fg].mean() if fg.any() else zero) + 0.5 * (raw[bg].mean() if bg.any() else zero)
    if orientation_weight > 0:
        valid = (tp[:, 0:1] > 0.9).float()
        diff = (torch.tanh(student_logits[:, CH_MOMENTS])
                - torch.tanh(teacher_logits[:, CH_MOMENTS])).square()
        loss = loss + orientation_weight * (diff * valid).sum() / (4 * valid.sum() + 1)
    return loss


def equivariance_loss(model, images, transform="flip_x"):
    """Predictions must transform with the image; a label-free structural constraint.

    Only the three probability maps are compared, so no orientation-channel sign
    bookkeeping is needed.  One transform is drawn per step to keep the cost at two
    extra forward passes.
    """
    z = model(images)
    if transform == "flip_x":
        back = torch.flip(model(torch.flip(images, [-1])), [-1])
    elif transform == "flip_y":
        back = torch.flip(model(torch.flip(images, [-2])), [-2])
    else:
        back = torch.rot90(model(torch.rot90(images, 1, (-2, -1))), -1, (-2, -1))
    return F.mse_loss(torch.sigmoid(z[:, 0:3]), torch.sigmoid(back[:, 0:3]))
