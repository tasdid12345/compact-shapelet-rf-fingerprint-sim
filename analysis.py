"""Simulated I/Q device fingerprinting under domain shift: random fixed-window shapelets + MLP
vs multi-length k-means shapelets + phase-invariant stats + CV-tuned class-balanced LightGBM."""
import json, os, sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, SEEDS, ND, NS, K = 42, (42, 43, 44), 10, 256, 40  # devices, frame length, shapelet budget
N_TRAIN_DOM, N_SHIFT = 300, 100  # frames per device (training domain / shifted test domain)

# ---------------- simulation ----------------
def device_params(rng):
    return [dict(g=rng.uniform(.85, 1.15), phi=rng.uniform(-.15, .15), dc=rng.uniform(.03, .12) * np.exp(2j * np.pi * rng.random()),
                 cfo=rng.uniform(-.01, .01), pa=rng.uniform(.02, .3)) for _ in range(ND)]

def gen_domain(devs, n, rng, shifted):
    """n frames per device. Hardware impairments -> channel -> noise."""
    X, y = [], []
    cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
    for d, p in enumerate(devs):
        s = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, (n, NS // 4))))
        x = lfilter(np.hanning(6) / np.hanning(6).sum(), 1, np.repeat(s, 4, axis=1), axis=1)
        x /= np.sqrt((abs(x) ** 2).mean(1, keepdims=True))
        x = x * (1 - p["pa"] * abs(x) ** 2)                                   # PA nonlinearity
        x = p["g"] * x.real + 1j * (x.imag * np.cos(p["phi"]) + x.real * np.sin(p["phi"]))  # IQ imbalance
        x = x + p["dc"]                                                         # DC offset
        x = x * np.exp(2j * np.pi * p["cfo"] * np.arange(NS))                   # carrier-frequency offset
        if shifted:   # new random multipath, uniform phase, low SNR
            h = cn(n, 3) * np.array([0., .4, .25]) + np.array([1., 0, 0]); h = h * np.exp(2j * np.pi * rng.random((n, 1)))
            snr = rng.uniform(8, 14, (n, 1))
        else:         # mild channel, small phase, high SNR
            h = np.stack([np.ones(n), .03 * cn(n), np.zeros(n)], 1) * np.exp(1j * rng.uniform(-.3, .3, (n, 1)))
            snr = 25.0
        y_ = sum(h[:, k:k + 1] * np.roll(x, dl, axis=1) for k, dl in enumerate((0, 2, 5)))
        y_ = y_ + cn(n, NS) * 10 ** (-snr / 20)
        X.append(np.stack([y_.real, y_.imag], 1).astype(np.float32)); y += [d] * n
    return np.concatenate(X), np.array(y)

# ---------------- shared feature code ----------------
def sample_windows(X, L, size, rng):
    fi, off = rng.integers(len(X), size=size), rng.integers(0, NS - L + 1, size=size)
    W = X[fi[:, None, None], np.arange(2)[None, :, None], off[:, None, None] + np.arange(L)[None, None, :]]
    return W.reshape(size, 2 * L)

def min_dist_feats(X, shapelets):
    """RMS min-distance of each (2,L) shapelet over all window positions (same for both models)."""
    out = []
    for L, S in shapelets:
        cols = []
        for i in range(0, len(X), 400):
            W = sliding_window_view(X[i:i + 400], L, axis=2).transpose(0, 2, 1, 3).reshape(-1, 2 * L)  # (n*nw, 2L)
            d2 = (W ** 2).sum(1, keepdims=True) - 2 * W @ S.T + (S ** 2).sum(1)
            cols.append(np.sqrt(np.maximum(d2.reshape(min(400, len(X) - i), -1, len(S)).min(1), 0) / L))
        out.append(np.concatenate(cols))
    return np.hstack(out)

def phase_invariant_stats(X):
    x = X[:, 0].astype(np.float64) + 1j * X[:, 1]
    a, pw = abs(x), (abs(x) ** 2).mean(1)
    r1 = (x[:, 1:] * np.conj(x[:, :-1])).mean(1)
    return np.c_[abs(x.mean(1)), abs((x ** 2).mean(1)) / pw, a.mean(1), a.std(1), kurtosis(a, axis=1),
                 a.max(1) / a.mean(1), abs(r1) / pw, np.angle(r1)]

def make_shapelets(Xtr, improved, seed):
    rng = np.random.default_rng(seed)
    if not improved:  # fixed-length random windows cut from training frames
        return [(32, sample_windows(Xtr, 32, K, rng).astype(np.float32))]
    res = []          # variable-length: k-means centroids at several scales, same total budget
    for L, k in zip((16, 32, 64), (13, 13, 14)):
        km = KMeans(k, n_init=3, random_state=seed).fit(sample_windows(Xtr, L, 12000, rng))
        res.append((L, km.cluster_centers_.astype(np.float32)))
    return res

def feats(X, shp, improved):
    F = min_dist_feats(X, shp)
    return np.hstack([F, phase_invariant_stats(X)]) if improved else F

def evaluate(clf, F, y):
    p = clf.predict(F)
    return accuracy_score(y, p), f1_score(y, p, average="macro")

# ---------------- main ----------------
rng = np.random.default_rng(SEED)
devs = device_params(rng)
Xd, yd = gen_domain(devs, N_TRAIN_DOM, rng, shifted=False)
Xs, ys = gen_domain(devs, N_SHIFT, rng, shifted=True)
Xtr, Xid, ytr, yid = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=SEED)  # single split
print(f"train {len(Xtr)}, in-domain test {len(Xid)}, shifted test {len(Xs)}", flush=True)

res = {"baseline": [], "improved": []}
for seed in SEEDS:
    for name in res:
        imp = name == "improved"
        shp = make_shapelets(Xtr, imp, seed)
        Ftr, Fid, Fs = (feats(X, shp, imp) for X in (Xtr, Xid, Xs))
        if imp:  # small CV grid on the training set only
            clf = GridSearchCV(LGBMClassifier(class_weight="balanced", colsample_bytree=.7, learning_rate=.1,
                                              random_state=seed, n_jobs=2, verbose=-1),
                               {"num_leaves": [7, 15], "n_estimators": [100, 250]}, cv=StratifiedKFold(3, shuffle=True, random_state=seed))
        else:
            clf = make_pipeline(StandardScaler(), MLPClassifier((128,), max_iter=300, early_stopping=True, random_state=seed))
        clf.fit(Ftr, ytr)
        (a, f), (ai, _) = evaluate(clf, Fs, ys), evaluate(clf, Fid, yid)  # test sets used once, after fitting
        res[name].append((a, f, ai))
        print(f"seed {seed} {name}: shifted acc {a:.3f} f1 {f:.3f} in-domain acc {ai:.3f}", flush=True)

m = {}
for name in res:
    r = np.array(res[name])
    m.update({f"{name}_score": r[:, 0].mean(), f"{name}_score_std": r[:, 0].std(), f"{name}_f1": r[:, 1].mean(),
              f"{name}_f1_std": r[:, 1].std(), f"{name}_in_domain_acc": r[:, 2].mean()})
m["improvement"] = m["improved_score"] - m["baseline_score"]
m = {k: float(v) for k, v in m.items()}
if not all(np.isfinite(list(m.values()))) or max(m["baseline_in_domain_acc"], m["improved_in_domain_acc"]) < 0.2:
    sys.exit("No meaningful metrics produced")
print(json.dumps(m, indent=2))

os.makedirs("figures", exist_ok=True)
plt.bar(["baseline", "improved"], [m["baseline_score"], m["improved_score"]],
        yerr=[m["baseline_score_std"], m["improved_score_std"]], color=["gray", "tab:blue"])
plt.ylabel("Shifted-domain accuracy"); plt.title("Device ID under domain shift (mean±std, 3 seeds)")
plt.savefig("figures/accuracy.png", dpi=120, bbox_inches="tight")

out = {
    "question": "Can a compact, data-adaptive multi-length I/Q shapelet model with gradient boosting, and no LLM, beat a fixed-window random-shapelet baseline at identifying simulated wireless devices under a domain shift?",
    "method": "Simulated QPSK I/Q frames (256 samples) from 10 devices with IQ imbalance, DC offset, CFO and PA nonlinearity. Train: high SNR, mild channel. Shifted test: lower SNR, new random multipath and phase offsets. Both models use the same min-distance shapelet code and shapelet budget; hyper-parameters tuned by CV on training data only.",
    "dataset": {"name": "Simulated I/Q RF-fingerprinting frames (seeded numpy)", "source_url": "self-generated (no download)",
                "n_rows": int(len(Xd) + len(Xs)), "n_train": int(len(Xtr)), "n_test": int(len(Xs)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "accuracy on shifted test domain", "higher_is_better": True,
    "baseline": f"{K} random fixed-length (32) 2D I/Q shapelets cut from training frames, min-distance features, StandardScaler + sklearn MLP.",
    "improved": f"{K} variable-length (16/32/64) shapelets from k-means on training windows, plus 8 phase-invariant statistics, class-balanced LightGBM with a small CV-tuned grid.",
    "metrics": m,
    "notes": "Simulation is a stand-in for the paper's datasets; the paper's frozen-LLM model is not reproduced. Improvements bundle several changes (k-means shapelets, multi-length, statistics, GBM) without ablation. Device impairments and data are generated once with seed 42; seeds 42-44 vary shapelet sampling, k-means and classifiers only. CV is in-domain so cannot see the shift. Results depend on the chosen simulation parameters.",
}
with open("results.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
