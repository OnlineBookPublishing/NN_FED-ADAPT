"""
Paper 1 experiment harness: negative transfer under a single-site domain shift.

One script runs every method on identical data, splits, seeds, shift and evaluation:
    fedavg, fedprox, fedbn, ditto, cfl, fed_adapt

What it logs (CSV, appended, so interrupted sweeps can resume):
    <out>/per_round.csv   seed, method, severity, split, round, client, acc
                          split = "val" every round for every client; "test" once at the end
    <out>/events.csv      Fed-ADAPT isolations and CFL splits (round, client/cluster info)
    <out>/signatures.csv  Fed-ADAPT detector score for every client, every round

Quick start (Colab):
    !python fl_harness.py --smoke                                  # ~1-3 min sanity check
    !python fl_harness.py --methods fedavg --severities 0.5,1.0,1.5 --seeds 0   # is the shift harmful?
    !python fl_harness.py                                          # full sweep (long)
    !python plot_results.py --out results                          # figure + table
"""
import argparse
import csv
import functools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print = functools.partial(print, flush=True)  # show progress immediately

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

NUM_CLIENTS = 10
NUM_CLASSES = 9
BATCH_SIZE = 64
EVAL_BATCH = 2048
SPLIT_SEED = 0  # client partition is fixed across methods and seeds, so only training randomness varies
ALL_METHODS = ["fedavg", "fedprox", "fedbn", "ditto", "cfl", "fed_adapt"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATA: PathMNIST held on the GPU as uint8; per-client train / val / test
# =============================================================================
_RAW = None
_SHIFT_CACHE = {}


# Client partition. "iid": every client gets a random, equal share (original setting).
# "dirichlet": label-skewed (non-IID) clients. Each client draws its class mix from Dirichlet(alpha)
# (smaller alpha = more different clients); every client still gets the same number of images as in the
# IID setting. Validation and test follow each client's own training class mix. Set from the command line.
PARTITION = "iid"
DIR_ALPHA = 0.5
NORM = "bn"  # "bn" = BatchNorm (original); "gn" = GroupNorm (no batch statistics). Set from the command line.


def _largest_remainder(total, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = w * total
    out = np.floor(raw).astype(int)
    rem = int(total - out.sum())
    if rem > 0:
        out[np.argsort(-(raw - out))[:rem]] += 1
    return out


def _dirichlet_split(y, target, n_per, seed):
    """Give each client n_per indices whose class mix follows its row of `target` as closely as the
    available images allow (a class that runs out is topped up from the remaining classes)."""
    rng = np.random.default_rng(seed)
    C, K = target.shape
    pools = {k: list(rng.permutation(np.where(y == k)[0])) for k in range(K)}
    parts = []
    for c in range(C):
        avail = np.array([len(pools[k]) for k in range(K)])
        take = np.minimum(_largest_remainder(n_per, target[c]), avail)
        deficit = n_per - take.sum()
        while deficit > 0:
            left = avail - take
            w = np.where(left > 0, target[c] + 1e-3, 0.0)
            take = take + np.minimum(_largest_remainder(deficit, w), left)
            deficit = n_per - take.sum()
        idx = []
        for k in range(K):
            idx += pools[k][:take[k]]
            pools[k] = pools[k][take[k]:]
        parts.append(np.array(idx, dtype=np.int64))
    return parts


def _load_raw():
    global _RAW
    if _RAW is None:
        from medmnist import PathMNIST
        print(f"Loading PathMNIST (first run downloads ~200 MB)... partition = {PARTITION}"
              + (f" (alpha = {DIR_ALPHA})" if PARTITION == "dirichlet" else ""))
        _RAW = {}
        target = None
        for split in ("train", "val", "test"):
            ds = PathMNIST(split=split, download=True)
            x = torch.from_numpy(ds.imgs).permute(0, 3, 1, 2).contiguous()  # N,3,28,28 uint8
            y = torch.from_numpy(ds.labels).view(-1).long()
            n = len(y) // NUM_CLIENTS
            if PARTITION == "iid":
                g = torch.Generator().manual_seed(SPLIT_SEED * 10 + {"train": 0, "val": 1, "test": 2}[split])
                perm = torch.randperm(len(y), generator=g)
                idx = [perm[i * n:(i + 1) * n] for i in range(NUM_CLIENTS)]
            else:
                y_np = y.numpy()
                if target is None:  # class mix per client, drawn once and shared by all splits
                    target = np.random.default_rng(SPLIT_SEED).dirichlet([DIR_ALPHA] * NUM_CLASSES, size=NUM_CLIENTS)
                parts = _dirichlet_split(y_np, target, n, SPLIT_SEED * 10 + {"train": 0, "val": 1, "test": 2}[split])
                if split == "train":  # val/test follow the class mix each client actually received
                    target = np.array([np.bincount(y_np[p], minlength=NUM_CLASSES) / len(p) for p in parts])
                    glob = np.bincount(y_np, minlength=NUM_CLASSES) / len(y_np)
                    tv = [0.5 * np.abs(t - glob).sum() for t in target]
                    print(f"  non-IID: distance of each client's class mix from the overall mix "
                          f"(0 = same, 1 = completely different): {np.round(tv, 2).tolist()}")
                    for c in range(NUM_CLIENTS):
                        print(f"    client {c}: class counts {np.bincount(y_np[parts[c]], minlength=NUM_CLASSES).tolist()}")
                idx = [torch.from_numpy(p) for p in parts]
            _RAW[split] = [(x[i].to(DEV), y[i].to(DEV)) for i in idx]
            print(f"  {split}: {n} images per client")
    return _RAW


def gaussian_blur(x, sigma, k=5):
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    w = torch.exp(-0.5 * (coords / sigma) ** 2)
    w = w / w.sum()
    C = x.shape[1]
    out = F.pad(x, (k // 2,) * 4, mode="reflect")
    out = F.conv2d(out, w.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(out, w.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def scanner_shift(x_u8, s):
    """Fixed, deterministic 'new scanner / stain protocol' shift of severity s (0 = none, sensible range 0-2).
    Every image gets the same transformation, like a real hardware change:
      colour cast (R down, B up), lower contrast, gamma (darker), fixed Gaussian blur.
    The constants are design choices -- state them in the paper."""
    if s <= 0:
        return x_u8.clone()
    x = x_u8.float() / 255
    gain = torch.tensor([1 - 0.30 * s, 1 + 0.10 * s, 1 + 0.25 * s], device=x.device).view(1, 3, 1, 1)
    x = (x * gain).clamp(0, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * max(0.05, 1 - 0.4 * s) + m).clamp(0, 1)
    x = x.pow(1 + 0.6 * s)
    x = gaussian_blur(x, 1.5 * s).clamp(0, 1)
    return (x * 255).round().to(torch.uint8)


def get_shifted(severity, cid):
    """Client cid's data under the scanner shift of this severity (cached)."""
    raw = _load_raw()
    key = (severity, cid)
    if key not in _SHIFT_CACHE:
        _SHIFT_CACHE[key] = {split: (scanner_shift(raw[split][cid][0], severity), raw[split][cid][1])
                             for split in raw}
    return _SHIFT_CACHE[key]


class Env:
    """Which data each client sees in a given round: the clients in `shifted` (default: client 0)
    switch to the shifted data from shift_round on; everyone else keeps their original data."""

    def __init__(self, severity, shift_round, shifted=(0,)):
        self.raw = _load_raw()
        self.severity = severity
        self.shift_round = shift_round
        self.shifted = tuple(shifted)

    def data(self, split, cid, rnd):
        if cid in self.shifted and rnd >= self.shift_round:
            return get_shifted(self.severity, cid)[split]
        return self.raw[split][cid]


# =============================================================================
# MODEL, TRAINING, EVALUATION
# =============================================================================
def norm_layer(ch):
    # GroupNorm with 8 groups has the same number of parameters as BatchNorm but no running statistics,
    # so nothing about a client's data distribution is averaged at the server except through the weights.
    return nn.GroupNorm(8, ch) if NORM == "gn" else nn.BatchNorm2d(ch)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), norm_layer(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def new_model():
    return SimpleCNN().to(DEV)


def to_input(x_u8):
    return x_u8.float().mul_(2 / 255).sub_(1)  # == ToTensor + Normalize(0.5, 0.5)


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def avg_states(states, keys):
    return {k: torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype) for k in keys}


def bn_keys(model):
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            keys.update(f"{name}.{k}" for k in m.state_dict())
    return keys


def train_local(model, x, y, epochs, lr, prox_ref=None, mu=0.0):
    """Adam, fresh optimizer each round. Optional proximal pull (mu/2)||w - w_ref||^2 (FedProx / Ditto),
    added to the gradient directly as mu * (w - w_ref)."""
    model.train()
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, fused=(DEV.type == "cuda"))
    ref = [p.detach().clone() for p in prox_ref.parameters()] if (prox_ref is not None and mu > 0) else None
    n = y.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            loss = F.cross_entropy(model(to_input(x[idx])), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ref is not None:
                with torch.no_grad():
                    torch._foreach_add_([p.grad for p in params], torch._foreach_sub(params, ref), alpha=mu)
            opt.step()


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = torch.zeros((), device=DEV)
    for s in range(0, y.size(0), EVAL_BATCH):
        correct += (model(to_input(x[s:s + EVAL_BATCH])).argmax(1) == y[s:s + EVAL_BATCH]).sum()
    return correct.item() / y.size(0)


# =============================================================================
# METHODS -- each implements run_round(rnd, env) and client_model(cid)
# client_model(cid) is the model that client would actually deploy; that is what gets evaluated.
# =============================================================================
class Method:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work = new_model()
        self.events = []      # dicts, written to events.csv
        self.sig_log = []     # dicts, written to signatures.csv

    def local(self, start_state, env, cid, rnd, prox_ref=None, mu=0.0):
        x, y = env.data("train", cid, rnd)
        self.work.load_state_dict(start_state)
        train_local(self.work, x, y, self.cfg.local_epochs, self.cfg.lr, prox_ref, mu)
        return clone_state(self.work)


class FedAvg(Method):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.glob = new_model()
        self.mu = 0.0

    def run_round(self, rnd, env):
        g = clone_state(self.glob)
        states = [self.local(g, env, cid, rnd, self.glob, self.mu) for cid in range(NUM_CLIENTS)]
        self.glob.load_state_dict(avg_states(states, g.keys()))

    def client_model(self, cid):
        return self.glob


class FedProx(FedAvg):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mu = cfg.mu


class FedBN(Method):
    """Shared weights averaged; every BatchNorm parameter and running statistic stays local to its client."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.glob = new_model()
        self.bn = bn_keys(self.glob)
        init = clone_state(self.glob)
        self.local_bn = [{k: init[k].clone() for k in self.bn} for _ in range(NUM_CLIENTS)]
        self.eval_model = new_model()

    def run_round(self, rnd, env):
        g = clone_state(self.glob)
        states = []
        for cid in range(NUM_CLIENTS):
            st = self.local({**g, **self.local_bn[cid]}, env, cid, rnd)
            self.local_bn[cid] = {k: st[k] for k in self.bn}
            states.append(st)
        shared = [k for k in g if k not in self.bn]
        self.glob.load_state_dict({**g, **avg_states(states, shared)})

    def client_model(self, cid):
        self.eval_model.load_state_dict({**self.glob.state_dict(), **self.local_bn[cid]})
        return self.eval_model


class Ditto(Method):
    """Global model trained FedAvg-style; each client also keeps a personalized model pulled toward the
    global one with (lambda/2)||v - w||^2. Evaluated with the personalized models (Ditto's deployed model)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.glob = new_model()
        g = clone_state(self.glob)
        self.pers = []
        for _ in range(NUM_CLIENTS):
            m = new_model()
            m.load_state_dict(g)
            self.pers.append(m)

    def run_round(self, rnd, env):
        g = clone_state(self.glob)
        states = []
        for cid in range(NUM_CLIENTS):
            states.append(self.local(g, env, cid, rnd))
            x, y = env.data("train", cid, rnd)
            train_local(self.pers[cid], x, y, self.cfg.local_epochs, self.cfg.lr,
                        prox_ref=self.glob, mu=self.cfg.ditto_lambda)
        self.glob.load_state_dict(avg_states(states, g.keys()))

    def client_model(self, cid):
        return self.pers[cid]


class CFL(Method):
    """Clustered FL (after Sattler et al.), simplified: after a warm-up, each cluster's client updates are
    bipartitioned by cosine similarity (complete linkage); the cluster splits if the largest cross-group
    similarity falls below gamma. The stationarity pre-condition of the original is omitted because Adam
    with local epochs never reaches it. The max cross similarity is logged every round so gamma can be
    calibrated on pre-shift rounds -- report the value ."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.clusters = [list(range(NUM_CLIENTS))]
        self.models = [new_model()]
        self.param_keys = [k for k, _ in self.models[0].named_parameters()]

    def run_round(self, rnd, env):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        new_clusters, new_models = [], []
        for members, model in zip(self.clusters, self.models):
            start = clone_state(model)
            states, updates = [], []
            for cid in members:
                st = self.local(start, env, cid, rnd)
                states.append(st)
                updates.append(torch.cat([(st[k] - start[k]).flatten() for k in self.param_keys]))
            model.load_state_dict(avg_states(states, start.keys()))

            split = None
            if len(members) > 1 and rnd > self.cfg.cfl_warmup:
                U = F.normalize(torch.stack(updates), dim=1)
                S = (U @ U.T).clamp(-1, 1).cpu().numpy()
                D = np.clip(1 - S, 0, None)
                np.fill_diagonal(D, 0)
                labels = fcluster(linkage(squareform(D, checks=False), "complete"), 2, "maxclust")
                g1 = [i for i, l in enumerate(labels) if l == 1]
                g2 = [i for i, l in enumerate(labels) if l != 1]
                cross = float(S[np.ix_(g1, g2)].max()) if g1 and g2 else 1.0
                self.sig_log.append({"round": rnd, "client": -1, "score": cross, "flag": int(cross < self.cfg.cfl_gamma),
                                     "note": f"cfl_cross_max cluster={members}"})
                if g1 and g2 and cross < self.cfg.cfl_gamma:
                    split = ([members[i] for i in g1], [members[i] for i in g2])
            if split:
                for part in split:
                    m = new_model()
                    m.load_state_dict(model.state_dict())
                    new_clusters.append(part)
                    new_models.append(m)
                self.events.append({"round": rnd, "client": -1, "event": "cfl_split",
                                    "detail": f"{members} -> {split[0]} | {split[1]}"})
                print(f"   >>> [CFL round {rnd}] split {members} -> {split[0]} | {split[1]}")
            else:
                new_clusters.append(members)
                new_models.append(model)
        self.clusters, self.models = new_clusters, new_models

    def client_model(self, cid):
        for members, model in zip(self.clusters, self.models):
            if cid in members:
                return model


class FedAdapt(Method):
    """Fed-ADAPT: track each client's final-layer signature, isolate a client into its own cluster after
    `persistence` consecutive-ish flags (counter +1 on flag, -1 otherwise), aggregate FedAvg within clusters.

    detector = "self"   :  original rule. signature = final-layer weights after local training;
                          flag if ||sig_t - sig_{t-1}|| > threshold.
    detector = "cohort" : signature = final-layer UPDATE (local - cluster model). Within the base cluster,
                          distance of each client's update to the median update, robust z-score
                          (median / MAD); flag if z > adapt_z. No absolute threshold to hand-tune.
    Every client's score is logged every round (signatures.csv)  can see whether Client 0 separates."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.models = {0: new_model()}
        self.assign = {i: 0 for i in range(NUM_CLIENTS)}
        self.prev_sig = {i: None for i in range(NUM_CLIENTS)}
        self.counter = {i: 0 for i in range(NUM_CLIENTS)}
        self.hist = {i: [] for i in range(NUM_CLIENTS)}  # temporal variant: each client's own recent distances
        self.hist_cos = {i: [] for i in range(NUM_CLIENTS)}  # harm variant: each client's own recent alignment
        self._pkeys = None

    temporal = False  # Fed-ADAPT-T sets this to True
    harm = False      # Fed-ADAPT-H sets this to True
    v2 = False        # Fed-ADAPT-H2 sets this to True
    v3 = False        # Fed-ADAPT-HG sets this to True
    v4 = False        # Fed-ADAPT-HG2 sets this to True

    @staticmethod
    def _final_layer(state):
        return torch.cat([state["fc.2.weight"].flatten(), state["fc.2.bias"].flatten()])

    def run_round(self, rnd, env):
        cfg = self.cfg
        start = {c: clone_state(m) for c, m in self.models.items()}
        local = {cid: self.local(start[self.assign[cid]], env, cid, rnd) for cid in range(NUM_CLIENTS)}

        scores, flags, notes = {}, {}, {}
        if cfg.adapt_detector == "self" and not (self.temporal or self.harm):
            for cid in range(NUM_CLIENTS):
                sig = self._final_layer(local[cid])
                if self.prev_sig[cid] is not None:
                    d = (sig - self.prev_sig[cid]).norm().item()
                    scores[cid], flags[cid] = d, d > cfg.adapt_threshold
                self.prev_sig[cid] = sig
        else:
            base = [cid for cid in range(NUM_CLIENTS) if self.assign[cid] == 0]
            if len(base) >= 4:
                U = torch.stack([self._final_layer(local[c]) - self._final_layer(start[0]) for c in base])
                d = (U - U.median(dim=0).values).norm(dim=1)
                md = d.median()
                mad = (d - md).abs().median() * 1.4826 + 1e-12
                z = ((d - md) / mad).tolist()
                notes = {}
                if self.harm:
                    # Update conflict: cosine between each client's FULL update and the mean update of the OTHER
                    # main-group clients. A client whose update points against the others would pull the shared
                    # model away from them when averaged, i.e. cause negative transfer.
                    if self._pkeys is None:
                        self._pkeys = [k for k, _ in self.models[0].named_parameters()]
                    G = torch.stack([torch.cat([(local[c][k] - start[0][k]).flatten().float() for k in self._pkeys])
                                     for c in base])
                    S = G.sum(0)
                    cosv = [F.cosine_similarity(G[i], (S - G[i]) / (len(base) - 1), dim=0).item()
                            for i in range(len(base))]
                for i, (c, zc, dc) in enumerate(zip(base, z, d.tolist())):
                    flag = zc > cfg.adapt_z
                    if self.harm:
                        # Harmful change = the client suddenly differs from ITS OWN history (distance jump) AND its
                        # update conflicts with the others (alignment drops below its own history, or turns negative).
                        h, hc, cs = self.hist[c], self.hist_cos[c], cosv[i]
                        if len(h) >= cfg.adapt_window and len(hc) >= cfg.adapt_window:
                            hm = float(np.median(h))
                            hmad = max(1.4826 * float(np.median(np.abs(np.array(h) - hm))), 0.1 * hm, 1e-12)
                            sz = (dc - hm) / hmad
                            cm = float(np.median(hc))
                            cmad = max(1.4826 * float(np.median(np.abs(np.array(hc) - cm))), 0.05)
                            cz = (cm - cs) / cmad
                            if self.v2:
                                # v2 (fed_adapt_h2): raw cosine is near zero or negative for most clients in normal
                                # rounds, so "cs < 0" is not evidence of conflict; use only the drop vs own history.
                                conflict = cz > cfg.h2_conflict_z
                                if self.v3:
                                    # HG: hospitals that change TOGETHER agree with each other, so their alignment
                                    # can rise; they still stand far from the group median (cohort z).
                                    conflict = conflict or zc > cfg.hg_cohort_z
                                self_thr = cfg.hg2_self_z if self.v4 else cfg.h2_self_z
                                flag = sz > self_thr and conflict
                            else:
                                conflict = cz > cfg.adapt_conflict_z or cs < 0
                                flag = sz > cfg.adapt_self_z and conflict
                        else:
                            sz, cz, flag = float("nan"), float("nan"), False
                        # HG2: a suspicious round (own-history z above hg2_freeze_z) is also kept out of the history,
                        # so a real change is not slowly absorbed into "normal" before it is confirmed.
                        frozen = self.v4 and sz == sz and sz > cfg.hg2_freeze_z
                        if not flag and not frozen:
                            h.append(dc)
                            del h[:-cfg.adapt_window]
                            hc.append(cs)
                            del hc[:-cfg.adapt_window]
                        notes[c] = f"harm self_z={sz:.2f};cos={cs:.3f};cos_z={cz:.2f};cohort_z={zc:.2f}"
                        scores[c], flags[c] = zc, flag
                        continue
                    if self.temporal:
                        # Temporal check: is this client also unusual compared with ITS OWN recent history?
                        # A naturally different (non-IID) client is far from the group every round, so its own
                        # history is also far and it is not flagged; a client whose data changes jumps.
                        h = self.hist[c]
                        if len(h) >= cfg.adapt_window:
                            hm = float(np.median(h))
                            hmad = max(1.4826 * float(np.median(np.abs(np.array(h) - hm))), 0.1 * hm, 1e-12)
                            sz = (dc - hm) / hmad
                            flag = flag and sz > cfg.adapt_self_z
                        else:
                            sz = float("nan")
                            flag = False  # not enough history yet (also avoids early-training false alarms)
                        if not flag:  # only normal rounds enter the history, so a change keeps standing out
                            h.append(dc)
                            del h[:-cfg.adapt_window]
                        notes[c] = f"temporal self_z={sz:.2f}"
                    scores[c], flags[c] = zc, flag

        for cid in range(NUM_CLIENTS):
            if cid not in scores:
                continue
            self.sig_log.append({"round": rnd, "client": cid, "score": scores[cid], "flag": int(flags[cid]),
                                 "note": notes.get(cid, cfg.adapt_detector) if (self.temporal or self.harm)
                                 else cfg.adapt_detector})
            if self.assign[cid] != 0:
                continue
            self.counter[cid] = self.counter[cid] + 1 if flags[cid] else max(0, self.counter[cid] - 1)
            if self.counter[cid] >= cfg.adapt_persistence:
                new_id = max(self.models) + 1
                m = new_model()
                m.load_state_dict(local[cid])
                self.models[new_id] = m
                self.assign[cid] = new_id
                self.counter[cid] = 0
                kind = "true_detection" if (cid in env.shifted and rnd >= env.shift_round and env.severity > 0) else "FALSE_ALARM"
                self.events.append({"round": rnd, "client": cid, "event": f"isolate_{kind}",
                                    "detail": f"cluster {new_id}"})
                name = "Fed-ADAPT-H" if self.harm else ("Fed-ADAPT-T" if self.temporal else "Fed-ADAPT")
                print(f"   >>> [{name} round {rnd}] isolated client {cid} "
                      f"-> cluster {new_id} ({kind})")

        for c, model in self.models.items():
            members = [cid for cid in range(NUM_CLIENTS) if self.assign[cid] == c]
            if members:
                model.load_state_dict(avg_states([local[i] for i in members], local[members[0]].keys()))

    def client_model(self, cid):
        return self.models[self.assign[cid]]


class FedAdaptT(FedAdapt):
    """Fed-ADAPT-T: Fed-ADAPT with a temporal check. A client is flagged only if it is unusual compared with the
    other clients (robust z > adapt_z) AND compared with its own last `adapt_window` normal rounds
    (robust z of its distance > adapt_self_z). Everything else (persistence, isolation, aggregation) is unchanged."""
    temporal = True


class FedAdaptH(FedAdapt):
    """Fed-ADAPT-H (harm-aware): flag a client only if (1) its distance to the group's median update jumps compared
    with its own last `adapt_window` normal rounds (robust z > adapt_self_z) AND (2) its full update conflicts with the
    other clients' mean update (cosine drops below its own history by robust z > adapt_conflict_z, or is negative).
    Persistence, isolation and aggregation are unchanged. All signals are logged in signatures.csv (note column)."""
    harm = True


class FedAdaptH2(FedAdaptH):
    """Fed-ADAPT-H2: same as Fed-ADAPT-H, but conflict = alignment with the others drops below the client's own
    history (robust z > h2_conflict_z, default 0.5); the "cosine < 0" rule is removed because normal non-IID updates
    already have cosine around zero. Own-history jump threshold h2_self_z (default 2.5)."""
    v2 = True


class FedAdaptHG(FedAdaptH2):
    """Fed-ADAPT-HG: like H2, but a client that jumps from its own history is also flagged when it is far from the
    group median (cohort robust z > hg_cohort_z, default 5). This catches several hospitals that change in the same
    way at once: they agree with each other, so their alignment with the others does not drop."""
    v3 = True


class FedAdaptHG2(FedAdaptHG):
    """Fed-ADAPT-HG2 (final): HG with own-history threshold hg2_self_z (default 2.0) and a history freeze: rounds
    whose own-history z exceeds hg2_freeze_z (default 1.5) do not enter the history, so a change keeps standing
    out until it is confirmed (in non-IID runs the change was otherwise absorbed within a few rounds)."""
    v4 = True


def _modularity_clusters(W, max_iter=100):
    """Greedy modularity local moving (FedCM, Sec. 3.2): every node starts alone and moves to the neighbouring
    community with the largest positive modularity gain, until no node moves. W: symmetric, non-negative."""
    n = W.shape[0]
    k = W.sum(1)
    m2 = W.sum()
    if m2 <= 0:
        return [[i] for i in range(n)]
    lab = list(range(n))
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            own = lab[i]
            lab[i] = -1
            tot = {}
            for j in range(n):
                if lab[j] >= 0:
                    tot[lab[j]] = tot.get(lab[j], 0.0) + k[j]
            best, best_gain = None, 0.0
            for c in set(l for l in lab if l >= 0):
                k_in = sum(W[i, j] for j in range(n) if lab[j] == c)
                gain = k_in - k[i] * tot[c] / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            new = best if best is not None else (own if own not in [l for l in lab if l >= 0] else max(lab) + 1 + n)
            lab[i] = new
            moved |= new != own
        if not moved:
            break
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(l, []).append(i)
    return sorted(groups.values())


class FedCM(Method):
    """FedCM (Wang et al., IJCAI 2025), reimplemented from the paper (no public code).
    - Personalized layer = the classification layer (fc.2), as in the paper.
    - Warm-up (fedcm_warmup rounds, FedAvg): record each client's classifier-update path; cosine similarity of the
      full paths (Eq. 4) -> graph -> greedy modularity clustering (Eq. 5), one shot.
    - Afterwards, FedAvg within clusters. Each round, a client's update direction (unit classifier update, Eq. 6) is
      compared with its cluster's norm-weighted mean direction (Eq. 7-8); if the cosine < tau the client migrates to
      the cluster with the most similar direction (Eq. 9).
    Not specified in the paper (our choices, reported): tau (default 0.0), warm-up length (10), negative similarities
    clipped to 0 for the graph, migration only to EXISTING clusters (fedcm). Variant fedcm_new also lets an outlier
    that matches no cluster (best cosine < tau) start a new cluster from its own local model."""
    allow_new = False
    tuned = False

    def _tau(self):
        return self.cfg.fedcm_tau_tuned if self.tuned else self.cfg.fedcm_tau

    def __init__(self, cfg):
        super().__init__(cfg)
        self.glob = new_model()
        self.paths = {c: [] for c in range(NUM_CLIENTS)}
        self.clusters = None
        self.models = []

    @staticmethod
    def _cls(state):
        return torch.cat([state["fc.2.weight"].flatten(), state["fc.2.bias"].flatten()]).float()

    def _kind(self, cid, rnd, env):
        return "true_detection" if (cid in env.shifted and rnd >= env.shift_round and env.severity > 0) else "FALSE_ALARM"

    def run_round(self, rnd, env):
        cfg = self.cfg
        if self.clusters is None:
            g = clone_state(self.glob)
            states = [self.local(g, env, c, rnd) for c in range(NUM_CLIENTS)]
            for c in range(NUM_CLIENTS):
                self.paths[c].append(self._cls(states[c]) - self._cls(g))
            self.glob.load_state_dict(avg_states(states, g.keys()))
            if rnd >= cfg.fedcm_warmup:
                P = torch.stack([torch.cat(self.paths[c]) for c in range(NUM_CLIENTS)])
                P = F.normalize(P, dim=1)
                S = (P @ P.T).clamp(-1, 1).cpu().numpy()
                W = np.clip(S, 0, None)
                np.fill_diagonal(W, 0)
                self.clusters = _modularity_clusters(W)
                self.models = []
                for _ in self.clusters:
                    m = new_model()
                    m.load_state_dict(self.glob.state_dict())
                    self.models.append(m)
                self.events.append({"round": rnd, "client": -1, "event": "fedcm_clusters",
                                    "detail": str(self.clusters)})
                print(f"   >>> [FedCM round {rnd}] initial clusters {self.clusters}")
            return
        # clustered training
        local_states, dirs, deltas = {}, {}, {}
        for k, members in enumerate(self.clusters):
            start = clone_state(self.models[k])
            sts = []
            for c in members:
                st = self.local(start, env, c, rnd)
                local_states[c] = st
                d = self._cls(st) - self._cls(start)
                deltas[c] = d
                dirs[c] = F.normalize(d, dim=0)
                sts.append(st)
            self.models[k].load_state_dict(avg_states(sts, start.keys()))
        # cluster directions: norm-weighted average of member unit directions = normalized sum of deltas (Eq. 7)
        cdir = [F.normalize(torch.stack([deltas[c] for c in mem]).sum(0), dim=0) for mem in self.clusters]
        moves = []
        for k, members in enumerate(self.clusters):
            for c in members:
                s_own = float(dirs[c] @ cdir[k])
                sims = [float(dirs[c] @ cd) for cd in cdir]
                flag = s_own < self._tau()
                self.sig_log.append({"round": rnd, "client": c, "score": s_own, "flag": int(flag),
                                     "note": f"fedcm cos_own={s_own:.3f};best_other={max([s for j, s in enumerate(sims) if j != k], default=float('nan')):.3f}"})
                if flag:
                    others = [(s, j) for j, s in enumerate(sims) if j != k]
                    best = max(others) if others else (-2.0, -1)
                    if best[1] >= 0 and (best[0] >= self._tau() or not self.allow_new):
                        moves.append((c, k, best[1]))
                    elif self.allow_new:
                        moves.append((c, k, "new"))
        for c, src, dst in moves:
            self.clusters[src].remove(c)
            if dst == "new":
                m = new_model()
                m.load_state_dict(local_states[c])
                self.models.append(m)
                self.clusters.append([c])
                dst_id = len(self.clusters) - 1
            else:
                self.clusters[dst].append(c)
                dst_id = dst
            kind = self._kind(c, rnd, env)
            self.events.append({"round": rnd, "client": c, "event": f"migrate_{kind}",
                                "detail": f"cluster {src} -> {dst_id}"})
            print(f"   >>> [FedCM round {rnd}] client {c} migrated {src} -> {dst_id} ({kind})")
        keep = [i for i, mem in enumerate(self.clusters) if mem]
        self.clusters = [self.clusters[i] for i in keep]
        self.models = [self.models[i] for i in keep]

    def client_model(self, cid):
        if self.clusters is None:
            return self.glob
        for mem, m in zip(self.clusters, self.models):
            if cid in mem:
                return m


class FedCMNew(FedCM):
    """FedCM variant: an outlier that matches no existing cluster starts a new cluster (our extension)."""
    allow_new = True


class FedCMTuned(FedCMNew):
    """FedCM (+new clusters) with tau tuned using knowledge of the change (best case, like our CFL threshold):
    tau = fedcm_tau_tuned, chosen from logged cosines so that the changed client falls below it."""
    tuned = True


class FedDrift(Method):
    """FedDrift (Jothimurugesan et al., AISTATS 2023), adapted from the official code (microsoft/FedDrift,
    'softcluster' + 'H_A_F_1_06_0'), with each communication round treated as one time step:
    - Start: every client has its own model (variant F, all from the same initialisation).
    - Each round, before training, every model in use is evaluated on every client's current local data
      (fd_eval training samples). A client moves to its best model; if its accuracy dropped by more than fd_delta
      (0.06) since the last round, drift is declared: it gets a NEW model (copy of its best model) and is kept
      isolated (no merging) for fd_w rounds (1).
    - Models are merged by hierarchical clustering (complete linkage) on the cluster distance
      D(i,j) = max(A_ii - A_ij, A_jj - A_ji, 0), where A_ij = accuracy of model i on cluster j's data; merge if
      D < fd_deltap (= fd_delta). Merged parameters are averaged weighted by cumulative client-rounds.
    - Then FedAvg within each model's clients. Unlike our method, this needs every client to evaluate every model
      every round (extra client computation, reported as overhead)."""

    tuned = False

    def _delta(self):
        return self.cfg.fd_delta_tuned if self.tuned else self.cfg.fd_delta

    def __init__(self, cfg):
        super().__init__(cfg)
        base = new_model()
        self.models = {}
        for c in range(NUM_CLIENTS):
            m = new_model()
            m.load_state_dict(base.state_dict())
            self.models[c] = m
        self.assign = {c: c for c in range(NUM_CLIENTS)}
        self.usage = {c: 0.0 for c in range(NUM_CLIENTS)}
        self.marked = {}           # client -> (model id, round to unmark)
        self.prev_acc = {}
        self.next_id = NUM_CLIENTS
        self.eval_idx = {}

    def _eval_data(self, env, c, rnd):
        x, y = env.data("train", c, rnd)
        if c not in self.eval_idx:
            g = torch.Generator().manual_seed(1000 + c)
            self.eval_idx[c] = torch.randperm(y.size(0), generator=g)[:self.cfg.fd_eval].to(y.device)
        idx = self.eval_idx[c]
        return x[idx], y[idx]

    def run_round(self, rnd, env):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        cfg = self.cfg
        for c in [c for c, (m, t) in self.marked.items() if t <= rnd]:
            del self.marked[c]
        data = {c: self._eval_data(env, c, rnd) for c in range(NUM_CLIENTS)}
        in_use = sorted(set(self.assign.values()))
        acc = {(m, c): accuracy(self.models[m], *data[c]) for m in in_use for c in range(NUM_CLIENTS)}
        # 1) assignment and drift detection
        for c in range(NUM_CLIENTS):
            if c in self.marked:
                continue
            best = max(in_use, key=lambda m: acc[(m, c)])
            newest = acc[(best, c)]
            drop = self.prev_acc[c] - newest if c in self.prev_acc else float("nan")
            drift = c in self.prev_acc and drop > self._delta()
            self.sig_log.append({"round": rnd, "client": c, "score": drop, "flag": int(drift),
                                 "note": f"feddrift best={best};acc={newest:.3f}"})
            if drift:
                nid = self.next_id
                self.next_id += 1
                m = new_model()
                m.load_state_dict(self.models[best].state_dict())
                self.models[nid] = m
                self.usage[nid] = 0.0
                self.assign[c] = nid
                self.marked[c] = (nid, rnd + cfg.fd_w)
                kind = "true_detection" if (c in env.shifted and rnd >= env.shift_round and env.severity > 0) else "FALSE_ALARM"
                self.events.append({"round": rnd, "client": c, "event": f"isolate_{kind}", "detail": f"new model {nid}"})
                print(f"   >>> [FedDrift round {rnd}] client {c} drift -> new model {nid} ({kind})")
            else:
                self.assign[c] = best
            self.prev_acc[c] = newest
        # 2) hierarchical merging among models not held by an isolated client
        marked_models = {m for (m, t) in self.marked.values()}
        cand = sorted(set(self.assign.values()) - marked_models)
        if len(cand) > 1:
            cdata = {}
            for m in cand:
                mem = [c for c in range(NUM_CLIENTS) if self.assign[c] == m]
                cdata[m] = (torch.cat([data[c][0] for c in mem]), torch.cat([data[c][1] for c in mem]))
            A = np.array([[accuracy(self.models[i], *cdata[j]) for j in cand] for i in cand])
            n = len(cand)
            D = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    D[i, j] = max(A[i, i] - A[i, j], A[j, j] - A[j, i], 0.0)
            D = np.maximum(D, D.T)
            np.fill_diagonal(D, 0)
            labels = fcluster(linkage(squareform(D, checks=False), "complete"), t=cfg.fd_delta, criterion="distance")
            groups = {}
            for i, l in enumerate(labels):
                groups.setdefault(l, []).append(cand[i])
            for grp in groups.values():
                if len(grp) < 2:
                    continue
                w = np.array([max(self.usage[m], 1.0) for m in grp])
                w = w / w.sum()
                sts = [clone_state(self.models[m]) for m in grp]
                merged = {k: sum(float(w[i]) * sts[i][k].float() for i in range(len(grp))).to(sts[0][k].dtype)
                          for k in sts[0]}
                base = grp[0]
                self.models[base].load_state_dict(merged)
                self.usage[base] = sum(self.usage[m] for m in grp)
                for m in grp[1:]:
                    for c in range(NUM_CLIENTS):
                        if self.assign[c] == m:
                            self.assign[c] = base
                    del self.models[m]
                    del self.usage[m]
                if rnd > 1:
                    self.events.append({"round": rnd, "client": -1, "event": "feddrift_merge", "detail": str(grp)})
        # 3) FedAvg within each model's clients
        for m in sorted(set(self.assign.values())):
            mem = [c for c in range(NUM_CLIENTS) if self.assign[c] == m]
            start = clone_state(self.models[m])
            sts = [self.local(start, env, c, rnd) for c in mem]
            self.models[m].load_state_dict(avg_states(sts, start.keys()))
            self.usage[m] += len(mem)
        # drop unused models
        used = set(self.assign.values())
        for m in [m for m in self.models if m not in used]:
            del self.models[m]
            self.usage.pop(m, None)

    def client_model(self, cid):
        return self.models[self.assign[cid]]


class FedDriftTuned(FedDrift):
    """FedDrift with the drift threshold tuned from our logs (fd_delta_tuned, default 0.3) so that normal
    round-to-round accuracy noise does not trigger new models (fairness check, like fedcm_tuned).
    The merge distance stays at fd_delta (0.06), as in the original."""
    tuned = True


class LocalOnly(Method):
    """Local-only baseline: every client trains its own model on its own data; nothing is shared."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.models = [new_model() for _ in range(NUM_CLIENTS)]

    def run_round(self, rnd, env):
        for cid in range(NUM_CLIENTS):
            self.models[cid].load_state_dict(self.local(clone_state(self.models[cid]), env, cid, rnd))

    def client_model(self, cid):
        return self.models[cid]


METHODS = {"fedavg": FedAvg, "fedprox": FedProx, "fedbn": FedBN, "ditto": Ditto, "cfl": CFL, "fed_adapt": FedAdapt,
           "fed_adapt_t": FedAdaptT, "fed_adapt_h": FedAdaptH, "fed_adapt_h2": FedAdaptH2, "fed_adapt_hg": FedAdaptHG, "fed_adapt_hg2": FedAdaptHG2, "fedcm": FedCM, "fedcm_new": FedCMNew, "fedcm_tuned": FedCMTuned, "feddrift": FedDrift, "feddrift_tuned": FedDriftTuned,
           "local": LocalOnly}


# =============================================================================
# RUNNER
# =============================================================================
def append_csv(path, rows, fields):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def completed_runs(out):
    path = os.path.join(out, "per_round.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["split"] == "test":
                    done.add((r["method"], float(r["severity"]), int(r["seed"])))
    return done


def run_one(method, severity, seed, cfg):
    seed_everything(seed)
    env = Env(severity, cfg.shift_round, cfg.shifted_set)
    stable_ids = [c for c in range(NUM_CLIENTS) if c not in cfg.shifted_set]
    shift_ids = list(cfg.shifted_set)
    m = METHODS[method](cfg)
    tag = f"[{method} sev={severity} seed={seed}]"
    rows = []
    t_start = time.time()
    for rnd in range(1, cfg.rounds + 1):
        t0 = time.time()
        m.run_round(rnd, env)
        accs = []
        for cid in range(NUM_CLIENTS):
            acc = accuracy(m.client_model(cid), *env.data("val", cid, rnd))
            accs.append(acc)
            rows.append({"seed": seed, "method": method, "severity": severity, "split": "val",
                         "round": rnd, "client": cid, "acc": acc})
        flag = " [SHIFT]" if rnd >= cfg.shift_round else ""
        print(f"{tag} round {rnd:02d}/{cfg.rounds}{flag} | stable {100 * np.mean([accs[c] for c in stable_ids]):.2f}% "
              f"| shifted {100 * np.mean([accs[c] for c in shift_ids]):.2f}% | {time.time() - t0:.1f}s")

    test_accs = []
    for cid in range(NUM_CLIENTS):
        acc = accuracy(m.client_model(cid), *env.data("test", cid, cfg.rounds))
        test_accs.append(acc)
        rows.append({"seed": seed, "method": method, "severity": severity, "split": "test",
                     "round": cfg.rounds, "client": cid, "acc": acc})
    print(f"{tag} TEST | stable {100 * np.mean([test_accs[c] for c in stable_ids]):.2f}% "
          f"| shifted {100 * np.mean([test_accs[c] for c in shift_ids]):.2f}% "
          f"| total {(time.time() - t_start) / 60:.1f} min")

    base = {"seed": seed, "method": method, "severity": severity}
    append_csv(os.path.join(cfg.out, "events.csv"), [{**base, **e} for e in m.events],
               ["seed", "method", "severity", "round", "client", "event", "detail"])
    append_csv(os.path.join(cfg.out, "signatures.csv"), [{**base, **s} for s in m.sig_log],
               ["seed", "method", "severity", "round", "client", "score", "flag", "note"])
    # per_round.csv written last: a run only counts as complete once its test rows exist
    append_csv(os.path.join(cfg.out, "per_round.csv"), rows,
               ["seed", "method", "severity", "split", "round", "client", "acc"])


def check_config(cfg):
    """Each output folder holds ONE experimental condition. The settings that define the condition are saved in
    <out>/config.json on first use; a later run with different settings is refused so results never get mixed."""
    import json
    cond = {"shifted_clients": list(cfg.shifted_set), "shift_round": cfg.shift_round, "rounds": cfg.rounds,
            "local_epochs": cfg.local_epochs, "lr": cfg.lr}
    if cfg.norm != "bn":  # only recorded when used, so older (BatchNorm) result folders stay compatible
        cond["norm"] = cfg.norm
    if cfg.partition != "iid":  # only recorded when used, so older (IID) result folders stay compatible
        cond["partition"], cond["alpha"] = cfg.partition, cfg.alpha
    path = os.path.join(cfg.out, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in cond.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"Output folder {cfg.out} was used with different settings {{setting: (saved, now)}}: "
                             f"{diff}. Use a separate --out folder for each condition.")
    else:
        with open(path, "w") as f:
            json.dump(cond, f, indent=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--severities", default="0.5,1.0,1.5", help="shift severities for client 0 (0 = no shift)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--shift-round", type=int, default=40)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=0.01, help="FedProx proximal weight")
    p.add_argument("--ditto-lambda", type=float, default=0.1)
    p.add_argument("--cfl-gamma", type=float, default=0.2, help="CFL split if max cross-group cosine < gamma")
    p.add_argument("--cfl-warmup", type=int, default=10)
    p.add_argument("--adapt-detector", choices=["cohort", "self"], default="cohort")
    p.add_argument("--adapt-threshold", type=float, default=0.8, help="'self' detector threshold (original)")
    p.add_argument("--adapt-z", type=float, default=3.5, help="'cohort' detector robust z-score threshold")
    p.add_argument("--adapt-persistence", type=int, default=2)
    p.add_argument("--adapt-window", type=int, default=10, help="fed_adapt_t: rounds of own history kept")
    p.add_argument("--adapt-self-z", type=float, default=3.5, help="fed_adapt_t/h: robust z threshold vs own history")
    p.add_argument("--adapt-conflict-z", type=float, default=3.5,
                   help="fed_adapt_h: robust z for a drop in alignment with the other clients vs own history")
    p.add_argument("--h2-self-z", type=float, default=2.5, help="fed_adapt_h2: own-history jump threshold")
    p.add_argument("--h2-conflict-z", type=float, default=0.5, help="fed_adapt_h2: alignment-drop threshold")
    p.add_argument("--hg-cohort-z", type=float, default=5.0, help="fed_adapt_hg: distance-from-group threshold")
    p.add_argument("--hg2-self-z", type=float, default=2.0, help="fed_adapt_hg2: own-history jump threshold")
    p.add_argument("--fedcm-warmup", type=int, default=10, help="fedcm: warm-up rounds before one-shot clustering")
    p.add_argument("--fedcm-tau", type=float, default=0.0, help="fedcm: migrate if cosine with own cluster < tau")
    p.add_argument("--fedcm-tau-tuned", type=float, default=0.3, help="fedcm_tuned: tuned migration threshold")
    p.add_argument("--fd-delta", type=float, default=0.06, help="feddrift: accuracy-drop threshold (and merge distance)")
    p.add_argument("--fd-delta-tuned", type=float, default=0.3, help="feddrift_tuned: tuned accuracy-drop threshold")
    p.add_argument("--fd-w", type=int, default=1, help="feddrift: rounds a drifted client stays isolated")
    p.add_argument("--fd-eval", type=int, default=1000, help="feddrift: local training samples used to evaluate models")
    p.add_argument("--hg2-freeze-z", type=float, default=1.5, help="fed_adapt_hg2: rounds above this stay out of history")
    p.add_argument("--shifted-clients", default="0",
                   help="comma-separated clients that shift at --shift-round (default: client 0 only)")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid",
                   help="iid = similar data at every client (original); dirichlet = non-IID label-skewed clients")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for --partition dirichlet "
                   "(smaller = more different clients)")
    p.add_argument("--norm", choices=["bn", "gn"], default="bn",
                   help="bn = BatchNorm (original); gn = GroupNorm (8 groups) instead of every BatchNorm")
    p.add_argument("--out", default="results")
    p.add_argument("--smoke", action="store_true", help="tiny run of every method to check everything works")
    p.add_argument("--rerun", action="store_true", help="do not skip runs already in per_round.csv")
    cfg = p.parse_args(argv)
    cfg.shifted_set = tuple(sorted(int(c) for c in cfg.shifted_clients.split(",") if c.strip()))
    if not cfg.shifted_set or any(c < 0 or c >= NUM_CLIENTS for c in cfg.shifted_set):
        raise SystemExit(f"--shifted-clients must list clients between 0 and {NUM_CLIENTS - 1}")
    if cfg.smoke:
        cfg.rounds, cfg.shift_round, cfg.local_epochs = 4, 3, 1
        cfg.seeds, cfg.severities, cfg.cfl_warmup = "0", "1.0", 1
        cfg.fedcm_warmup = 2
        cfg.out = cfg.out if cfg.out != "results" else "results_smoke"
        cfg.rerun = True
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    global PARTITION, DIR_ALPHA, NORM
    PARTITION, DIR_ALPHA, NORM = cfg.partition, cfg.alpha, cfg.norm
    os.makedirs(cfg.out, exist_ok=True)
    check_config(cfg)
    methods = [m.strip() for m in cfg.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"Unknown method(s) {bad}; choose from {sorted(METHODS)}")
    severities = [float(s) for s in cfg.severities.split(",")]
    seeds = [int(s) for s in cfg.seeds.split(",")]
    done = set() if cfg.rerun else completed_runs(cfg.out)

    print(f"Using device: {DEV}")
    if DEV.type != "cuda":
        print("WARNING: no GPU -- this will be very slow. In Colab: Runtime > Change runtime type > GPU.")
    jobs = [(m, s, sd) for s in severities for sd in seeds for m in methods]
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already complete, {len(todo)} to go. Output: {cfg.out}/")
    for i, (m, s, sd) in enumerate(todo, 1):
        print(f"\n===== run {i}/{len(todo)}: {m} | severity {s} | seed {sd} =====")
        run_one(m, s, sd, cfg)
    print("\nAll runs finished.")


if __name__ == "__main__":
    main()
