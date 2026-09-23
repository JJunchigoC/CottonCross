"""Backbones under comparison, and CFX-Net (ours).

Every backbone is wrapped by the SAME multi-task head and trained with the SAME loss
set and budget, so Table 1 isolates the architecture.  The head emits

    0        fiber occupancy            (sigmoid)
    1        centerline                 (sigmoid)
    2        crossing heatmap           (sigmoid)
    3:7      cos2t, sin2t, cos4t, sin4t (tanh)   local orientation moments
    7:10     overlap multiplicity       (softmax over {none, one fiber, >=2 fibers})
    10:18    instance embedding         (l2-normalised)

Orientation is undirected, hence the second-order moments; the fourth-order pair keeps
information at right-angle crossings where the second-order pair cancels.

CFX-Net adds two modules to a residual U-Net:

* **DCM** - a dilated context module at the bottleneck.  A 35 mm fiber is far longer
  than any receptive field we can afford at full resolution; dilation buys continuity
  evidence across the gap that a crossing opens in the centerline.
* **OCA** - orientation-gated crossing attention at full resolution.  A bank of
  oriented strip convolutions measures directional energy; the *second* largest
  directional response is, by construction, large only where two differently oriented
  structures coexist, i.e. exactly at a crossing.  That signal gates the decoder
  features and is fed directly to the crossing head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

OUT_CHANNELS = 18
CH_MASK = slice(0, 1)
CH_CENTER = slice(1, 2)
CH_JOINT = slice(2, 3)
CH_MOMENTS = slice(3, 7)
CH_OVERLAP = slice(7, 10)
CH_EMBED = slice(10, 18)
EMBED_DIM = 8


def gn(ch):
    groups = next(g for g in (8, 4, 2, 1) if ch % g == 0)
    return nn.GroupNorm(groups, ch)


class Conv(nn.Sequential):
    def __init__(self, cin, cout, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            gn(cout), nn.SiLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(Conv(cin, cout), Conv(cout, cout))


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(Conv(cin, cout), nn.Conv2d(cout, cout, 3, padding=1, bias=False), gn(cout))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


def up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


# ------------------------------------------------------------------ plain U-Net family
class UNet(nn.Module):
    block = DoubleConv

    def __init__(self, base=32, depth=4, in_ch=3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(self.block(prev, c))
            prev = c
        self.bottom = self.block(prev, prev * 2)
        self.dec = nn.ModuleList()
        prev = prev * 2
        for c in reversed(chs):
            self.dec.append(self.block(prev + c, c))
            prev = c
        self.out_channels = base

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottom(x)
        for d, s in zip(self.dec, reversed(skips)):
            x = d(torch.cat([up(x, s), s], 1))
        return x


class ResUNet(UNet):
    block = ResBlock


class AttentionGate(nn.Module):
    """Attention U-Net gate (Oktay et al., 2018)."""

    def __init__(self, gate_ch, skip_ch, inter=None):
        super().__init__()
        inter = inter or max(skip_ch // 2, 8)
        self.wg = nn.Conv2d(gate_ch, inter, 1)
        self.wx = nn.Conv2d(skip_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, gate, skip):
        a = torch.sigmoid(self.psi(F.silu(self.wg(gate) + self.wx(skip))))
        return skip * a


class AttentionUNet(nn.Module):
    def __init__(self, base=32, depth=4, in_ch=3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(DoubleConv(prev, c))
            prev = c
        self.bottom = DoubleConv(prev, prev * 2)
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = prev * 2
        for c in reversed(chs):
            self.gates.append(AttentionGate(prev, c))
            self.dec.append(DoubleConv(prev + c, c))
            prev = c
        self.out_channels = base

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottom(x)
        for g, d, s in zip(self.gates, self.dec, reversed(skips)):
            u = up(x, s)
            x = d(torch.cat([u, g(u, s)], 1))
        return x


class UNetPP(nn.Module):
    """U-Net++ (Zhou et al., 2018) with nested dense skip pathways, depth 4."""

    def __init__(self, base=32, depth=4, in_ch=3):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth)]
        self.nodes = nn.ModuleDict()
        for i in range(depth):
            self.nodes[f"{i}_0"] = DoubleConv(in_ch if i == 0 else chs[i - 1], chs[i])
        for j in range(1, depth):
            for i in range(depth - j):
                self.nodes[f"{i}_{j}"] = DoubleConv(chs[i] * j + chs[i + 1], chs[i])
        self.out_channels = base

    def forward(self, x):
        cache = {}
        for i in range(self.depth):
            src = x if i == 0 else F.avg_pool2d(cache[f"{i - 1}_0"], 2)
            cache[f"{i}_0"] = self.nodes[f"{i}_0"](src)
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                prev = [cache[f"{i}_{k}"] for k in range(j)]
                deep = up(cache[f"{i + 1}_{j - 1}"], prev[0])
                cache[f"{i}_{j}"] = self.nodes[f"{i}_{j}"](torch.cat(prev + [deep], 1))
        return cache[f"0_{self.depth - 1}"]


class DSCBlock(nn.Module):
    """DSCNet encoder block: standard conv fused with the two snake branches.

    Uses the authors' released `DSConv_pro` operator (MIT, vendored unmodified under
    `cottoncross/third_party/`), so the baseline is the published operator rather than a
    re-implementation.
    """

    def __init__(self, cin, cout, kernel_size=9, extend_scope=1.0):
        super().__init__()
        from .third_party.dsconv_pro import DSConv_pro

        # The released DSConv normalises with out_channels // 4 groups, so each snake
        # branch needs a channel count that is a positive multiple of four.
        unit = max(4, (cout // 3) // 4 * 4)
        if cout - 2 * unit < 4:
            raise ValueError(f"DSCNet needs at least 12 channels per level, got {cout}")
        # The vendored operator builds its sampling grid on a fixed device at construction.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.plain = Conv(cin, cout - 2 * unit)
        self.snake_x = DSConv_pro(cin, unit, kernel_size, extend_scope, morph=0, if_offset=True, device=dev)
        self.snake_y = DSConv_pro(cin, unit, kernel_size, extend_scope, morph=1, if_offset=True, device=dev)
        self.norm = nn.Sequential(gn(cout), nn.SiLU(inplace=True))
        self.fuse = Conv(cout, cout)

    def forward(self, x):
        z = torch.cat([self.plain(x), self.snake_x(x), self.snake_y(x)], 1)
        return self.fuse(self.norm(z))


class DSCNet(UNet):
    block = DSCBlock

    def __init__(self, base=24, depth=4, in_ch=3):
        super().__init__(base, depth, in_ch)


# ------------------------------------------------------------------------- ours
class DilatedContext(nn.Module):
    """Parallel dilated branches at the bottleneck; long-range continuity evidence."""

    def __init__(self, ch, rates=(1, 2, 4, 8)):
        super().__init__()
        unit = max(ch // len(rates), 8)
        self.branches = nn.ModuleList([Conv(ch, unit, dilation=r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, unit, 1), nn.SiLU(inplace=True))
        self.fuse = Conv(unit * (len(rates) + 1), ch)

    def forward(self, x):
        parts = [b(x) for b in self.branches]
        g = self.pool(x).expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.fuse(torch.cat(parts + [g], 1)) + x


class OrientedBank(nn.Module):
    """Depthwise strip convolutions along 4 orientations (0, 45, 90, 135 degrees)."""

    def __init__(self, ch, length=11):
        super().__init__()
        self.h = nn.Conv2d(ch, ch, (1, length), padding=(0, length // 2), groups=ch, bias=False)
        self.v = nn.Conv2d(ch, ch, (length, 1), padding=(length // 2, 0), groups=ch, bias=False)
        self.d1 = nn.Conv2d(ch, ch, length, padding=length // 2, groups=ch, bias=False)
        self.d2 = nn.Conv2d(ch, ch, length, padding=length // 2, groups=ch, bias=False)
        eye = torch.eye(length)
        self.register_buffer("m1", eye.view(1, 1, length, length))
        self.register_buffer("m2", torch.flip(eye, [1]).view(1, 1, length, length))

    def forward(self, x):
        pad = self.d1.padding[0]
        d1 = F.conv2d(x, self.d1.weight * self.m1, padding=pad, groups=x.shape[1])
        d2 = F.conv2d(x, self.d2.weight * self.m2, padding=pad, groups=x.shape[1])
        return [self.h(x), d1, self.v(x), d2]


class CrossingAttention(nn.Module):
    """Orientation-gated crossing attention (OCA).

    `top2` - the second largest directional energy - is the crossing descriptor: a
    single fiber excites one orientation, a crossing excites two.

    Two details matter in practice.  The per-orientation energy is the channel MAXIMUM,
    not the mean, so a few direction-selective channels are not diluted by the rest.  And
    the four energies are standardised per image before sorting: the raw activations have
    an arbitrary scale and a large positive mean, which leaves any downstream gain badly
    conditioned - in an earlier version the gain simply collapsed to zero and the
    descriptor was never used.
    """

    def __init__(self, ch, length=11, version="v2"):
        super().__init__()
        if version not in ("v1", "v2"):
            raise ValueError(f"unknown OCA version {version!r}")
        self.version = version
        self.bank = OrientedBank(ch, length)
        self.descr = nn.Sequential(nn.Conv2d(4, ch // 2, 1), gn(ch // 2), nn.SiLU(inplace=True))
        self.gate = nn.Sequential(nn.Conv2d(ch + ch // 2, ch, 1), nn.Sigmoid())
        self.mix = nn.Conv2d(ch + ch // 2, ch, 1)

    def forward(self, x):
        pool = (lambda r: F.relu(r).mean(1)) if self.version == "v1" \
            else (lambda r: F.relu(r).amax(1))
        energies = torch.stack([pool(r) for r in self.bank(x)], 1)          # (B,4,H,W)
        if self.version == "v2":
            mean = energies.mean(dim=(1, 2, 3), keepdim=True)
            std = energies.std(dim=(1, 2, 3), keepdim=True) + 1e-5
            energies = (energies - mean) / std
        ordered, _ = torch.sort(energies, dim=1, descending=True)
        top1, top2 = ordered[:, 0:1], ordered[:, 1:2]
        descriptor = torch.cat([top1, top2, top1 - top2, energies.mean(1, keepdim=True)], 1)
        d = self.descr(descriptor)
        z = torch.cat([x, d], 1)
        if self.version == "v1":
            return self.mix(z) + x * self.gate(z), top2
        # v2 is a gated residual update rather than a replacement: the identity path
        # survives, so attention can only add orientation evidence, never erase weak
        # fiber evidence the decoder already found.
        return x + self.mix(z) * self.gate(z), descriptor


class CFXNet(nn.Module):
    """Ours: residual U-Net + dilated context + orientation-gated crossing attention."""

    def __init__(self, base=32, depth=4, in_ch=3, strip=11, use_oca=True, use_dcm=True,
                 deep_supervision=True, oca_version="v2"):
        super().__init__()
        self.use_oca = use_oca
        self.oca_version = oca_version
        self.deep_supervision = deep_supervision
        chs = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(nn.Sequential(ResBlock(prev, c), ResBlock(c, c)))
            prev = c
        self.bottom = (nn.Sequential(ResBlock(prev, prev * 2), DilatedContext(prev * 2))
                       if use_dcm else ResBlock(prev, prev * 2))
        self.dec = nn.ModuleList()
        prev = prev * 2
        for c in reversed(chs):
            self.dec.append(ResBlock(prev + c, c))
            prev = c
        self.oca = CrossingAttention(base, strip, oca_version) if use_oca else None
        self.out_channels = base
        self.deep = nn.ModuleList([nn.Conv2d(c, 3, 1) for c in reversed(chs[1:])])

    def forward(self, x, want_aux=False):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottom(x)
        outs = []
        for d, s in zip(self.dec, reversed(skips)):
            x = d(torch.cat([up(x, s), s], 1))
            outs.append(x)
        if self.oca is not None:
            x, crossing_prior = self.oca(x)
        else:
            channels = 1 if self.oca_version == "v1" else 4
            crossing_prior = x.new_zeros((x.shape[0], channels, x.shape[2], x.shape[3]))
        if want_aux:
            deep = ([h(o) for h, o in zip(self.deep, outs[:-1])]
                    if self.deep_supervision else [])
            return x, dict(crossing_prior=crossing_prior, deep=deep)
        return x


# --------------------------------------------------------------------------- wrapper
BACKBONES = dict(unet=UNet, attunet=AttentionUNet, unetpp=UNetPP, resunet=ResUNet,
                 dscnet=DSCNet, cfxnet=CFXNet)


class FiberModel(nn.Module):
    """Backbone + shared multi-task head."""

    def __init__(self, arch="cfxnet", base=32, depth=4, in_ch=3, strip=11, **extra):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(BACKBONES)}")
        self.arch = arch
        kwargs = dict(base=base, depth=depth, in_ch=in_ch)
        if arch == "cfxnet":
            kwargs["strip"] = strip
            kwargs.update({k: v for k, v in extra.items()
                           if k in ("use_oca", "use_dcm", "deep_supervision",
                                    "oca_version")})
        self.backbone = BACKBONES[arch](**kwargs)
        ch = self.backbone.out_channels
        self.head = nn.Conv2d(ch, OUT_CHANNELS, 1)
        if arch == "cfxnet":
            # The OCA descriptor is injected into the crossing logit as an explicit prior.
            # v2 uses a 1x1 convolution rather than v1's single scalar: the four
            # descriptor channels carry different evidence and need their own signs and
            # scales.  Zero initialisation means training starts exactly where it would
            # without the prior.  v1 is kept so its archived checkpoints still load.
            if self.backbone.oca_version == "v1":
                self.prior_gain = nn.Parameter(torch.zeros(1))
            else:
                self.prior = nn.Conv2d(4, 1, 1)
                nn.init.zeros_(self.prior.weight)
                nn.init.zeros_(self.prior.bias)

    def forward(self, x, want_aux=False):
        if self.arch == "cfxnet":
            feat, aux = self.backbone(x, want_aux=True)
            z = self.head(feat)
            if self.backbone.oca is None:
                prior = torch.zeros_like(z[:, 2:3])
            elif self.backbone.oca_version == "v1":
                prior = self.prior_gain * aux["crossing_prior"]
            else:
                prior = self.prior(aux["crossing_prior"])
            z = torch.cat([z[:, :2], z[:, 2:3] + prior, z[:, 3:]], 1)
            return (z, aux) if want_aux else z
        z = self.head(self.backbone(x))
        return (z, {}) if want_aux else z


def activate(logits):
    """Map raw logits to the interpretable prediction bundle."""
    return dict(
        mask=torch.sigmoid(logits[:, CH_MASK]),
        center=torch.sigmoid(logits[:, CH_CENTER]),
        joint=torch.sigmoid(logits[:, CH_JOINT]),
        moments=torch.tanh(logits[:, CH_MOMENTS]),
        overlap=torch.softmax(logits[:, CH_OVERLAP], 1),
        embed=F.normalize(logits[:, CH_EMBED], dim=1),
    )


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
