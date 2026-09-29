"""
hydronet.py - HydroNet: a pretrained YOLO11-s + P2 detector, adapted to underwater debris.

Design rule: every new part starts as a NO-OP, so at step 0 the model is exactly the
COCO-pretrained network, and each addition can only change it as far as training finds useful.

  SPD stem          space-to-depth + 3x3 conv, initialised to reproduce pretrained layer 0 exactly,
                    but able to use the full 2x2 pixel detail (6x6 window) - more capacity for tiny objects
  Physics condition 3 macro maps (veiling, absorption, sharpness) + 5 global numbers, computed from the
                    pixels (no learning), encoded by a tiny depthwise-separable CNN into per-position
                    scale/shift (FiLM) for backbone P2/P3/P4. Zero-initialised.
  Aux mask head     training-only foreground mask from the P2 neck output (BCE + Dice). Removed at inference.
  Adaptive NWD      label assignment uses max(CIoU, scale-adaptive NWD): kinder to tiny boxes,
                    never harsher than the baseline on any box.
  Optional          DySample at the P3->P2 upsample; EMA attention (zero-gated) on the P2/P3 outputs.

Ultralytics 8.4 already floors tiny boxes to a minimum candidate size during assignment, so no extra
tiny-object fallback is added (it would be redundant).
"""
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.torch_utils import unwrap_model

DEFAULT_OPTS = dict(physics=True, spd=True, aux=True, nwd=True, dysample=False, ema_attn=False,
                    mask_weight=1.0, nwd_cmin=8.0, nwd_kappa=0.5)


# ----------------------------------------------------------------------------- building blocks
class DWSep(nn.Module):
    """Depthwise-separable conv: 3x3 per channel (where), then 1x1 across channels (what). ~8x cheaper."""

    def __init__(self, c1, c2):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


def _box(t, k):
    """Separable k x k mean filter with edge replication (no fake borders)."""
    p = k // 2
    t = F.avg_pool2d(F.pad(t, (p, p, 0, 0), mode="replicate"), (1, k), 1)
    return F.avg_pool2d(F.pad(t, (0, 0, p, p), mode="replicate"), (k, 1), 1)


def _copy_layer_attrs(src, dst):
    for a in ("i", "f", "type", "np"):
        if hasattr(src, a):
            setattr(dst, a, getattr(src, a))
    return dst


class SPDStem(nn.Module):
    """Space-to-depth stem replacing pretrained layer 0 (3x3 conv, stride 2).

    The 2x2 pixel blocks are stacked into channels (nothing discarded), then a 3x3 conv at half resolution
    sees a 6x6 full-resolution window. Its weights are set so the output equals the pretrained conv
    exactly at initialisation; extra weights start at zero and learn only if detail helps.
    """

    def __init__(self, conv0: Conv):
        super().__init__()
        w0 = conv0.conv.weight.data  # (c2, c1, 3, 3), stride 2, padding 1
        c2, c1 = w0.shape[:2]
        assert conv0.conv.stride == (2, 2) and conv0.conv.kernel_size == (3, 3), "layer 0 is not a 3x3/s2 conv"
        self.conv = Conv(c1 * 4, c2, 3, 1)
        k = torch.zeros(c2, c1 * 4, 3, 3, dtype=w0.dtype)
        # full-res tap offset d in {-1,0,+1} -> (kernel index at half res, sub-pixel phase)
        tap = {0: (0, 1), 1: (1, 0), 2: (1, 1)}
        for c in range(c1):
            for iy, (ky, a) in tap.items():
                for ix, (kx, b) in tap.items():
                    k[:, c * 4 + a * 2 + b, ky, kx] = w0[:, c, iy, ix]
        self.conv.conv.weight.data.copy_(k)
        self.conv.bn.load_state_dict(conv0.bn.state_dict())
        self.conv.act = conv0.act
        _copy_layer_attrs(conv0, self)

    def forward(self, x):
        return self.conv(F.pixel_unshuffle(x, 2))


class PhysicsCondition(nn.Module):
    """Underwater condition maps -> FiLM scale/shift for chosen backbone layers.

    Maps (computed at 1/4 resolution, no learning, letterbox padding excluded):
      veiling    15x15 min of min(G,B)             backscatter; ignores red, which water removes anyway
      absorption 31x31 mean of log((G+B)/2 / R)    water colour / light path; large window averages out objects
      sharpness  Laplacian energy over ~28 px       blur from forward scattering (a window statistic, not edges)
    Global (per image): Shades-of-Gray colour cast (3), mean veiling, mean sharpness.
    """

    N_IN = 8  # 3 maps + 5 global

    def __init__(self, levels, width=32):
        super().__init__()
        self.levels = [tuple(int(v) for v in lv) for lv in levels]  # (layer index, channels, stride)
        self.register_buffer("mu", torch.zeros(self.N_IN))
        self.register_buffer("sd", torch.ones(self.N_IN))
        self.register_buffer("calibrated", torch.zeros(()))
        self.register_buffer("lap", torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3))
        self.enc4 = DWSep(self.N_IN, width)
        self.enc8 = DWSep(width, width)
        self.heads = nn.ModuleList(nn.Conv2d(width, 2 * c, 1) for _, c, _ in self.levels)
        for h in self.heads:  # zero start: FiLM is the identity until training says otherwise
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self._acc = None

    @torch.no_grad()
    def raw(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            pad = ((x[:, 0] == x[:, 1]) & (x[:, 1] == x[:, 2])).unsqueeze(1).float()  # letterbox grey
            x4 = F.avg_pool2d(x, 4)
            pad = (F.avg_pool2d(pad, 4) > 0.5).float()
            keep = 1.0 - pad
            r, g, b = x4[:, 0:1], x4[:, 1:2], x4[:, 2:3]
            v = torch.minimum(g, b) * keep + pad  # padding -> 1 so the min ignores it
            v = -F.max_pool2d(F.pad(-v, (7, 7, 7, 7), mode="replicate"), 15, 1)
            v = _box(v, 5)
            a = torch.log((g + b) / 2 + 0.02) - torch.log(r + 0.02)
            a = (_box(a * keep, 31) / (_box(keep, 31) + 1e-6)).clamp(-2.0, 4.0)
            y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            lap = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), self.lap.float())
            s = torch.log(_box(F.avg_pool2d(lap * lap, 4), 7) + 1e-5)
            n = keep.sum((2, 3)).clamp(min=1.0)
            cast = ((x4.clamp(min=0) ** 6 * keep).sum((2, 3)) / n) ** (1 / 6)
            cast = 3 * cast / cast.sum(1, keepdim=True).clamp(min=1e-6)
            glob = torch.cat([cast, (v * keep).sum((2, 3)) / n, (s * keep).sum((2, 3)) / n], 1)
            return torch.cat([v, a, s], 1), glob, pad

    def forward(self, x):
        maps, glob, pad = self.raw(x)
        mu, sd = self.mu.float(), self.sd.float()
        maps = (maps - mu[:3].view(1, 3, 1, 1)) / sd[:3].view(1, 3, 1, 1) * (1.0 - pad)
        glob = (glob - mu[3:]) / sd[3:]
        inp = torch.cat([maps, glob[:, :, None, None].expand(-1, -1, *maps.shape[-2:])], 1)
        inp = inp.to(self.enc4.pw.weight.dtype)
        f4 = self.enc4(inp)
        f8 = self.enc8(F.avg_pool2d(f4, 2))
        out = []
        for (li, _, s), head in zip(self.levels, self.heads):
            if s <= 4:
                f = f4 + F.interpolate(f8, size=f4.shape[-2:], mode="bilinear", align_corners=False)
            elif s == 8:
                f = f8
            else:
                f = F.avg_pool2d(f8, s // 8)
            gamma, beta = head(f).chunk(2, 1)
            out.append((li, 0.5 * torch.tanh(gamma), beta))
        return out

    # --- fixed dataset-level normalisation (keeps absolute turbidity differences between images)
    @torch.no_grad()
    def calib_add(self, imgs):
        maps, glob, pad = self.raw(imgs)
        keep = (1.0 - pad).bool().expand_as(maps)
        m = [maps[:, i][keep[:, i]].double() for i in range(3)]
        acc = self._acc or {"m": [[0.0, 0.0, 0] for _ in range(3)], "g": []}
        for i in range(3):
            acc["m"][i][0] += m[i].sum().item()
            acc["m"][i][1] += (m[i] ** 2).sum().item()
            acc["m"][i][2] += m[i].numel()
        acc["g"].append(glob.double().cpu())
        self._acc = acc

    @torch.no_grad()
    def calib_finish(self):
        acc, mu, sd = self._acc, torch.zeros(self.N_IN), torch.ones(self.N_IN)
        for i, (s1, s2, n) in enumerate(acc["m"]):
            mu[i] = s1 / max(n, 1)
            sd[i] = max((s2 / max(n, 1) - mu[i].item() ** 2) ** 0.5, 1e-3)
        g = torch.cat(acc["g"])
        mu[3:], sd[3:] = g.mean(0).float(), g.std(0).clamp(min=1e-3).float()
        self.mu.copy_(mu)
        self.sd.copy_(sd)
        self.calibrated.fill_(1)
        self._acc = None


class AuxMaskHead(nn.Module):
    """Training-only foreground mask at stride 4. Teaches which pixels are object - boxes of ropes, nets
    and bags are mostly seabed. Discarded at inference (zero cost)."""

    def __init__(self, c):
        super().__init__()
        self.body = DWSep(c, c)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.constant_(self.out.bias, -3.5)  # prior: objects cover a few % of pixels

    def forward(self, x):
        return self.out(self.body(x))


class DySampleUp(nn.Module):
    """DySample (Liu et al., ICCV 2023), 'lp' style, static scope 0.25. Learns where to sample when
    upsampling, so features snap to object edges. Starts as ~bilinear upsampling."""

    def __init__(self, c, scale=2, groups=4):
        super().__init__()
        self.scale, self.groups = scale, groups
        self.offset = nn.Conv2d(c, 2 * groups * scale**2, 1)
        nn.init.normal_(self.offset.weight, 0, 0.001)
        nn.init.zeros_(self.offset.bias)
        h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
        pos = torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2)
        self.register_buffer("init_pos", pos.repeat(1, groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos.to(x.dtype)
        b, _, h, w = offset.shape
        offset = offset.reshape(b, 2, -1, h, w)
        ch = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        cw = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([cw, ch], indexing="ij")).transpose(1, 2).contiguous()[None, :, None]
        norm = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / norm - 1
        s = self.scale
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), s).reshape(b, 2, -1, s * h, s * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        out = F.grid_sample(x.reshape(b * self.groups, -1, h, w), coords, mode="bilinear",
                            align_corners=False, padding_mode="border")
        return out.reshape(b, -1, s * h, s * w)


class EMAAttention(nn.Module):
    """Efficient Multi-scale Attention (Ouyang et al., ICASSP 2023), wrapped in a zero-initialised gate
    so it starts as a no-op. Not the same thing as the EMA of weights that Ultralytics also uses."""

    def __init__(self, c, factor=8):
        super().__init__()
        self.g = factor
        cg = c // factor
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1 = nn.Conv2d(cg, cg, 1)
        self.conv3 = nn.Conv2d(cg, cg, 3, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        gx = x.reshape(b * self.g, -1, h, w)
        xh, xw = self.pool_h(gx), self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1(torch.cat([xh, xw], 2))
        xh, xw = torch.split(hw, [h, w], 2)
        x1 = self.gn(gx * xh.sigmoid() * xw.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3(gx)
        a1 = F.softmax(x1.mean((2, 3)).unsqueeze(1), -1)
        a2 = F.softmax(x2.mean((2, 3)).unsqueeze(1), -1)
        wts = (a1 @ x2.reshape(b * self.g, c // self.g, -1) + a2 @ x1.reshape(b * self.g, c // self.g, -1))
        att = (gx * wts.reshape(b * self.g, 1, h, w).sigmoid()).reshape(b, c, h, w)
        return x + self.alpha * (att - x)


# ----------------------------------------------------------------------------- model
class HydroDetectionModel(DetectionModel):
    """DetectionModel with FiLM physics conditioning, optional post-layer blocks and an aux mask head.
    Hooks live inside the forward loop, so no layer index changes and all pretrained weights load as-is."""

    def _predict_once(self, x, profile=False, embed=None):
        phys = getattr(self, "hydro_physics", None)
        film = {li: (g, b) for li, g, b in phys(x)} if phys is not None else {}
        post = getattr(self, "hydro_post", None)
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if m.i in film:
                g, b = film[m.i]
                if g.shape[-2:] != x.shape[-2:]:
                    g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    b = F.interpolate(b, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x * (1 + g.to(x.dtype)) + b.to(x.dtype)
            if post is not None and str(m.i) in post:
                x = post[str(m.i)](x)
            y.append(x if m.i in self.save else None)
            if m.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        aux = getattr(self, "hydro_aux", None)
        if aux is not None and self.training and isinstance(x, dict):
            x["aux_mask"] = aux(y[self.hydro_aux_src])
        return x

    @torch.no_grad()
    def _probe(self, idx):
        shapes, hooks = {}, []
        for i in idx:
            hooks.append(self.model[i].register_forward_hook(lambda m, a, o, i=i: shapes.__setitem__(i, o.shape)))
        was = self.training
        self.eval()
        p = next(self.parameters())
        self._predict_once(torch.zeros(1, 3, 128, 128, device=p.device, dtype=p.dtype))
        self.train(was)
        for h in hooks:
            h.remove()
        return {i: (s[1], 128 // s[-1]) for i, s in shapes.items()}  # channels, stride

    def hydro_setup(self, **opts):
        o = {**DEFAULT_OPTS, **opts}
        info = self._probe((2, 4, 6, 16, 19, 22))
        assert info[19][1] == 4 and info[2][1] == 4, "expected the yolo11s-p2 layout (P2 at layers 2 and 19)"
        if o["spd"]:
            self.model[0] = SPDStem(self.model[0])
        if o["physics"]:
            self.hydro_physics = PhysicsCondition([(i, info[i][0], info[i][1]) for i in (2, 4, 6)])
        if o["aux"]:
            assert 19 in self.save
            self.hydro_aux, self.hydro_aux_src = AuxMaskHead(info[19][0]), 19
        if o["dysample"]:
            assert isinstance(self.model[17], nn.Upsample), "layer 17 should be the P3->P2 upsample"
            self.model[17] = _copy_layer_attrs(self.model[17], DySampleUp(info[16][0]))
        if o["ema_attn"]:
            self.hydro_post = nn.ModuleDict({str(i): EMAAttention(info[i][0]) for i in (19, 22)})
        self.hydro_opts = o
        self.criterion = None
        return self

    def init_criterion(self):
        return HydroLoss(self)


# ----------------------------------------------------------------------------- training objective
def adaptive_nwd(gt, pd, cmin=8.0, kappa=0.5):
    """Scale-adaptive Normalized Wasserstein Distance similarity for xyxy pixel boxes, shape (N,).

    Boxes are 2D Gaussians. Tolerance C = max(cmin, kappa * sqrt(area)):
    tiny objects get a fixed pixel tolerance (NWD), larger ones a size-relative one (IoU-like).
    kappa = 0.5 matches IoU's sensitivity to small shifts on large boxes (IoU ~ 1 - 2*shift/side).
    """
    gw, gh = (gt[:, 2] - gt[:, 0]).clamp(min=1e-3), (gt[:, 3] - gt[:, 1]).clamp(min=1e-3)
    pw, ph = (pd[:, 2] - pd[:, 0]).clamp(min=1e-3), (pd[:, 3] - pd[:, 1]).clamp(min=1e-3)
    d2 = ((gt[:, :2] + gt[:, 2:]) / 2 - (pd[:, :2] + pd[:, 2:]) / 2).pow(2).sum(-1)
    w2 = d2 + ((gw - pw) ** 2 + (gh - ph) ** 2) / 4
    c = (kappa * (gw * gh).sqrt()).clamp(min=cmin)
    return torch.exp(-(w2 + 1e-9).sqrt() / c)


class HydroAssigner(TaskAlignedAssigner):
    """Task-aligned assigner scoring matches by max(CIoU, adaptive NWD): never harsher than the stock
    assigner on any box, and still informative for tiny boxes that don't overlap yet."""

    def __init__(self, *args, cmin=8.0, kappa=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmin, self.kappa = cmin, kappa

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        iou = super().iou_calculation(gt_bboxes, pd_bboxes)
        return torch.maximum(iou, adaptive_nwd(gt_bboxes, pd_bboxes, self.cmin, self.kappa).to(iou.dtype))


def mask_loss(logits, masks):
    t = (masks.to(logits.device) > 0).float()
    t = t.unsqueeze(1) if t.dim() == 3 else t
    if t.shape[-2:] != logits.shape[-2:]:
        t = F.interpolate(t, size=logits.shape[-2:], mode="nearest")
    lg = logits.float()
    bce = F.binary_cross_entropy_with_logits(lg, t)
    p = lg.sigmoid()
    dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


class HydroLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        o = model.hydro_opts
        if o["nwd"]:
            a = self.assigner
            self.assigner = HydroAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta,
                                          stride=a.stride, eps=a.eps, topk2=a.topk2,
                                          cmin=o["nwd_cmin"], kappa=o["nwd_kappa"])
        self.mask_weight = o["mask_weight"] if o["aux"] else 0.0

    def loss(self, preds, batch):
        aux = preds.pop("aux_mask", None)
        total, items = super().loss(preds, batch)
        m = torch.zeros((), device=total.device)
        if aux is not None and self.mask_weight > 0 and batch.get("masks") is not None:
            m = self.mask_weight * mask_loss(aux, batch["masks"])
        items = dict(items)
        items["mask_loss"] = m.detach()
        return torch.cat([total, (m * preds["boxes"].shape[0]).view(1).to(total.dtype)]), items


# ----------------------------------------------------------------------------- trainer
class HydroTrainer(DetectionTrainer):
    """Builds HydroNet, gives the training set masks, and calibrates physics normalisation once."""

    hydro_opts = dict(DEFAULT_OPTS)
    calib_images = 64

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            HydroDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1))
        if isinstance(weights, HydroDetectionModel):  # resuming / continuing our own checkpoint
            model.hydro_setup(**weights.hydro_opts)
            model.load(weights)
        else:  # COCO-pretrained start: load first, so the SPD stem copies the pretrained layer 0
            if weights:
                model.load(weights)
            model.hydro_setup(**self.hydro_opts)
        return model

    def _opts(self):
        return getattr(unwrap_model(self.model), "hydro_opts", None) or self.hydro_opts

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        cfg = copy(self.args)
        if mode == "train" and (self._opts()["aux"] or self.args.copy_paste > 0):
            cfg.task = "segment"  # load polygons -> masks for the aux head and copy-paste
        return build_yolo_dataset(cfg, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def _build_train_pipeline(self):
        super()._build_train_pipeline()
        phys = getattr(unwrap_model(self.model), "hydro_physics", None)
        if phys is None or phys.calibrated.item() > 0:
            return
        ds = self.train_loader.dataset
        n = min(self.calib_images, len(ds))
        idx = torch.randperm(len(ds))[:n].tolist()
        LOGGER.info(f"HydroNet: calibrating physics normalisation on {n} training images...")
        for i in range(0, n, 8):
            imgs = torch.stack([ds[j]["img"] for j in idx[i:i + 8]]).to(self.device).float() / 255
            phys.calib_add(imgs)
        phys.calib_finish()
        names = ("veiling", "absorption", "sharpness", "cast_R", "cast_G", "cast_B", "mean_veil", "mean_sharp")
        LOGGER.info("HydroNet: " + ", ".join(f"{k} {m:.3f}±{s:.3f}" for k, m, s in
                                             zip(names, phys.mu.tolist(), phys.sd.tolist())))
