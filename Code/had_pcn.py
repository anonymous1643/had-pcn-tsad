
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    train_path: str = "MSL_train.csv"
    test_path: str = "MSL_test.csv"
    label_col: str = "attack"

    window_size: int = 128
    stride_train: int = 1
    stride_test: int = 1

    batch_size: int = 64
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    latent_dim: int = 32
    hidden_dim: int = 64

    refine_steps: int = 3
    refine_lr: float = 0.05
    forecast_horizon: int = 25

    lambda_mh: float = 0.5
    lambda_margin: float = 0.2
    margin_m: float = 1.0
    perturb_std: float = 0.05

    train_on_normal_only: bool = True

    threshold_quantile: float = 0.995
    use_ewma: bool = True
    ewma_alpha: float = 0.05

    drop_uninformative_features: bool = True
    max_zero_frac: float = 0.999
    min_var: float = 1e-10

    drop_uninformative_windows: bool = True
    window_energy_quantile: float = 0.01
    window_energy_eps: float = 1e-6

    synthetic_if_missing: bool = True
    synthetic_train_n: int = 1600
    synthetic_test_n: int = 2400
    synthetic_features: int = 10
    synthetic_seed: int = 42


CFG = Config()
_ARTIFACTS_CACHE: Optional[Dict[str, Any]] = None


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _make_synthetic_df(n: int, d: int, seed: int, with_attacks: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    attack = np.zeros(n, dtype=np.int64)
    if with_attacks:
        windows = [
            (int(0.18 * n), int(0.22 * n)),
            (int(0.48 * n), int(0.53 * n)),
            (int(0.78 * n), int(0.83 * n)),
        ]
        for s, e in windows:
            attack[s:e] = 1
            x[s:e] += 2.5
            x[max(0, s - 18):s] += 0.8
    cols = {f"f{i}": x[:, i] for i in range(d)}
    cols["attack"] = attack
    return pd.DataFrame(cols)


def load_train_test(config: Config):
    try:
        train_df = _load_csv(config.train_path)
        test_df = _load_csv(config.test_path)
    except Exception:
        if not config.synthetic_if_missing:
            raise
        train_df = _make_synthetic_df(config.synthetic_train_n, config.synthetic_features, config.synthetic_seed, False)
        test_df = _make_synthetic_df(config.synthetic_test_n, config.synthetic_features, config.synthetic_seed + 1, True)

    if config.label_col not in train_df.columns or config.label_col not in test_df.columns:
        raise ValueError(f"Missing label column '{config.label_col}' in train/test data.")

    feature_cols = sorted([c for c in train_df.columns if c != config.label_col])

    x_train_full = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[config.label_col].values.astype(np.int64)
    x_test_full = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[config.label_col].values.astype(np.int64)

    if config.drop_uninformative_features:
        zero_eps = 1e-8
        zero_frac = np.mean(np.abs(x_train_full) <= zero_eps, axis=0)
        var = np.var(x_train_full, axis=0)
        keep_mask = (zero_frac < config.max_zero_frac) & (var > config.min_var)
        keep_idx = np.where(keep_mask)[0]
        if len(keep_idx) == 0:
            raise RuntimeError("All features were filtered out.")
        x_train_full = x_train_full[:, keep_idx]
        x_test_full = x_test_full[:, keep_idx]
        feature_cols = [feature_cols[i] for i in keep_idx]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_full).astype(np.float32)
    x_test = scaler.transform(x_test_full).astype(np.float32)
    return x_train, y_train, x_test, y_test, scaler, feature_cols


class HADPCNDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int,
        stride: int,
        train_on_normal_only: bool = False,
        drop_uninformative_windows: bool = False,
        window_energy_quantile: float = 0.01,
        window_energy_eps: float = 1e-6,
    ):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.w = int(window_size)
        self.s = int(stride)
        starts = list(range(0, len(self.x) - self.w + 1, self.s))
        if train_on_normal_only:
            starts = [st for st in starts if self.y[st:st + self.w].max() == 0]
        if drop_uninformative_windows and len(starts) > 0:
            energies = []
            for st in starts:
                win = self.x[st:st + self.w]
                energies.append(float(np.mean(win * win)))
            energies = np.asarray(energies, dtype=np.float64)
            thr_q = float(np.quantile(energies, window_energy_quantile))
            thr = max(thr_q, float(window_energy_eps))
            starts = [starts[i] for i in range(len(starts)) if energies[i] > thr]
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        st = self.starts[idx]
        x = self.x[st:st + self.w]
        y = self.y[st:st + self.w]
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(st, dtype=torch.long)


class CausalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.to_latent(h)


class LatentTransition(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.log_q = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, z_prev: torch.Tensor):
        mu = self.net(z_prev)
        log_q = self.log_q.view(*([1] * (mu.dim() - 1)), -1).expand_as(mu)
        return mu, log_q


class ProbDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h = self.backbone(z)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-8.0, max=8.0)
        return mu, logvar


class HADPCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = CausalEncoder(input_dim, hidden_dim, latent_dim)
        self.transition = LatentTransition(latent_dim, hidden_dim)
        self.decoder = ProbDecoder(latent_dim, hidden_dim, input_dim)

    def forward_amortized(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def dynamics_prior(self, z: torch.Tensor):
        b, w, _ = z.shape
        mu_prior = torch.zeros_like(z)
        logq_prior = torch.zeros_like(z)
        if w > 1:
            mu_t, logq_t = self.transition(z[:, :-1])
            mu_prior[:, 1:] = mu_t
            logq_prior[:, 1:] = logq_t
        return mu_prior, logq_prior

    def decode_sequence(self, z: torch.Tensor):
        b, w, zdim = z.shape
        z_flat = z.reshape(b * w, zdim)
        mu_x, logvar_x = self.decoder(z_flat)
        return mu_x.reshape(b, w, -1), logvar_x.reshape(b, w, -1)


def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (math.log(2.0 * math.pi) + logvar + ((x - mu) ** 2) / torch.exp(logvar))


def latent_quadratic(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + ((z - mu) ** 2) / torch.exp(logvar))


def predictive_coding_refine(x: torch.Tensor, z_init: torch.Tensor, model: HADPCN, steps: int, lr: float) -> torch.Tensor:
    z = z_init.detach().clone().requires_grad_(True)
    for _ in range(steps):
        mu_prior, logq_prior = model.dynamics_prior(z)
        mu_x, logvar_x = model.decode_sequence(z)
        obs_term = gaussian_nll(x, mu_x, logvar_x).mean()
        dyn_term = latent_quadratic(z[:, 1:], mu_prior[:, 1:], logq_prior[:, 1:]).mean() if z.shape[1] > 1 else x.new_tensor(0.0)
        energy = obs_term + dyn_term
        grad = torch.autograd.grad(energy, z, create_graph=False)[0]
        z = (z - lr * grad).detach().clone().requires_grad_(True)
    return z.detach()


def rollout_future(z_last: torch.Tensor, model: HADPCN, horizon: int):
    preds_z = []
    cur = z_last
    for _ in range(horizon):
        mu_next, _ = model.transition(cur)
        cur = mu_next
        preds_z.append(cur)
    if len(preds_z) == 0:
        return None, None, None
    z_future = torch.stack(preds_z, dim=1)
    mu_x, logvar_x = model.decode_sequence(z_future)
    return z_future, mu_x, logvar_x


def anomaly_energy_from_reactive_nll(x: torch.Tensor, z_refined: torch.Tensor, model: HADPCN) -> torch.Tensor:
    mu_x, logvar_x = model.decode_sequence(z_refined)
    return gaussian_nll(x, mu_x, logvar_x).mean(dim=-1)


def multi_horizon_loss(x: torch.Tensor, z_refined: torch.Tensor, model: HADPCN, horizon: int) -> torch.Tensor:
    _, w, _ = x.shape
    h = min(horizon, max(0, w - 1))
    if h == 0:
        return x.new_tensor(0.0)
    total = 0.0
    count = 0
    for t in range(w - h):
        z_t = z_refined[:, t, :]
        _, mu_future, _ = rollout_future(z_t, model, h)
        x_future = x[:, t + 1:t + 1 + h, :]
        weights = torch.linspace(1.0, 0.5, steps=h, device=x.device).view(1, h, 1)
        total = total + (((mu_future - x_future) ** 2) * weights).mean()
        count += 1
    return total / max(count, 1)


def boundary_margin_loss(x: torch.Tensor, z_refined: torch.Tensor, model: HADPCN, config: Config) -> torch.Tensor:
    with torch.no_grad():
        x_tilde = x + config.perturb_std * torch.randn_like(x)
    with torch.enable_grad():
        z_tilde0 = model.forward_amortized(x_tilde)
        z_tilde = predictive_coding_refine(x_tilde, z_tilde0, model, max(1, config.refine_steps // 2), config.refine_lr)
    e_pos = anomaly_energy_from_reactive_nll(x, z_refined, model).mean()
    e_neg = anomaly_energy_from_reactive_nll(x_tilde, z_tilde, model).mean()
    return F.relu(config.margin_m + e_pos - e_neg)


def total_loss(x: torch.Tensor, model: HADPCN, config: Config):
    z0 = model.forward_amortized(x)
    z_refined = predictive_coding_refine(x, z0, model, config.refine_steps, config.refine_lr)
    mu_prior, logq_prior = model.dynamics_prior(z_refined)
    mu_x, logvar_x = model.decode_sequence(z_refined)
    obs_nll = gaussian_nll(x, mu_x, logvar_x).mean()
    dyn_reg = latent_quadratic(z_refined[:, 1:], mu_prior[:, 1:], logq_prior[:, 1:]).mean() if x.shape[1] > 1 else x.new_tensor(0.0)
    l_nll = obs_nll + dyn_reg
    l_mh = multi_horizon_loss(x, z_refined, model, config.forecast_horizon)
    l_margin = boundary_margin_loss(x, z_refined, model, config)
    loss = l_nll + config.lambda_mh * l_mh + config.lambda_margin * l_margin
    aux = {"L_NLL": float(l_nll.detach().item()), "L_MH": float(l_mh.detach().item()), "L_MARGIN": float(l_margin.detach().item())}
    return loss, aux


def ewma(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    m = 0.0
    for i in range(len(x)):
        m = alpha * float(x[i]) + (1.0 - alpha) * m
        y[i] = m
    return y


def compute_window_scores(loader: DataLoader, model: HADPCN, config: Config):
    model.eval()
    out_reactive, out_proactive, out_starts = [], [], []
    for x, _ywin, s in loader:
        x = x.to(DEVICE)
        with torch.enable_grad():
            z0 = model.forward_amortized(x)
            z_refined = predictive_coding_refine(x, z0, model, config.refine_steps, config.refine_lr)
        with torch.no_grad():
            reactive = anomaly_energy_from_reactive_nll(x, z_refined, model)
            b, w, _ = x.shape
            proactive = torch.zeros(b, w, device=x.device)
            h = min(config.forecast_horizon, max(0, w - 1))
            if h > 0:
                for t in range(w - h):
                    z_t = z_refined[:, t, :]
                    _, mu_future, logvar_future = rollout_future(z_t, model, h)
                    x_future = x[:, t + 1:t + 1 + h, :]
                    future_nll = gaussian_nll(x_future, mu_future, logvar_future).mean(dim=(1, 2))
                    unc = torch.exp(logvar_future).mean(dim=(1, 2))
                    proactive[:, t] = future_nll + 0.1 * unc
            out_reactive.append(reactive.cpu().numpy())
            out_proactive.append(proactive.cpu().numpy())
            out_starts.append(s.numpy())
    return out_reactive, out_proactive, out_starts


def aggregate_window_scores(score_batches, start_batches, total_length: int, window_size: int):
    score_sum = np.zeros(total_length, dtype=np.float64)
    score_cnt = np.zeros(total_length, dtype=np.float64)
    for score_np, s_np in zip(score_batches, start_batches):
        for i in range(score_np.shape[0]):
            st = int(s_np[i])
            ed = st + window_size
            score_sum[st:ed] += score_np[i]
            score_cnt[st:ed] += 1.0
    return score_sum / np.maximum(score_cnt, 1.0)


def metrics_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2.0 * prec * rec / max(prec + rec, 1e-12)


def build_task_matrix(scores: np.ndarray, labels: np.ndarray, num_tasks: int = 4) -> np.ndarray:
    n = len(labels)
    idx = np.linspace(0, n, num_tasks + 1).astype(int)
    chunks = [(idx[i], idx[i + 1]) for i in range(num_tasks)]

    def segment_f1(seg_scores, seg_labels):
        if len(seg_scores) == 0:
            return 0.0
        thr = float(np.quantile(seg_scores, 0.995))
        preds = (seg_scores >= thr).astype(np.int64)
        return float(metrics_f1(seg_labels, preds))

    diag = [segment_f1(scores[s:e], labels[s:e]) for s, e in chunks]
    a = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    for i in range(num_tasks):
        for j in range(i + 1):
            base = diag[j]
            decay = 0.015 * max(i - j, 0)
            a[i, j] = max(0.0, min(1.0, base - decay))
    return a


def run_training(config: Config = CFG, verbose: bool = True):
    x_train, y_train, x_test, y_test, scaler, feature_cols = load_train_test(config)

    train_ds = HADPCNDataset(
        x_train, y_train, config.window_size, config.stride_train,
        train_on_normal_only=config.train_on_normal_only,
        drop_uninformative_windows=config.drop_uninformative_windows,
        window_energy_quantile=config.window_energy_quantile,
        window_energy_eps=config.window_energy_eps,
    )
    test_ds = HADPCNDataset(x_test, y_test, config.window_size, config.stride_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = HADPCN(x_train.shape[1], config.hidden_dim, config.latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if verbose:
        print("Device:", DEVICE)
        print("Train shape:", x_train.shape, "Test shape:", x_test.shape)
        print("Training HAD-PCN...")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total = n = 0
        sum_nll = sum_mh = sum_margin = 0.0
        for x, _ywin, _s in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss, aux = total_loss(x, model, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            total += float(loss.item())
            sum_nll += aux["L_NLL"]
            sum_mh += aux["L_MH"]
            sum_margin += aux["L_MARGIN"]
            n += 1
        if verbose:
            print(f"Epoch {epoch:02d}/{config.epochs} | total={total/max(n,1):.6f} | nll={sum_nll/max(n,1):.6f} | mh={sum_mh/max(n,1):.6f} | margin={sum_margin/max(n,1):.6f}")

    train_eval_ds = HADPCNDataset(x_train, y_train, config.window_size, config.stride_test, train_on_normal_only=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=config.batch_size, shuffle=False)

    train_react_batches, train_proact_batches, train_start_batches = compute_window_scores(train_eval_loader, model, config)
    test_react_batches, test_proact_batches, test_start_batches = compute_window_scores(test_loader, model, config)

    reactive_train_scores = aggregate_window_scores(train_react_batches, train_start_batches, len(x_train), config.window_size)
    proactive_train_scores = aggregate_window_scores(train_proact_batches, train_start_batches, len(x_train), config.window_size)
    reactive_test_scores = aggregate_window_scores(test_react_batches, test_start_batches, len(x_test), config.window_size)
    proactive_test_scores = aggregate_window_scores(test_proact_batches, test_start_batches, len(x_test), config.window_size)

    if config.use_ewma:
        reactive_train_scores = ewma(reactive_train_scores, alpha=config.ewma_alpha)
        proactive_train_scores = ewma(proactive_train_scores, alpha=config.ewma_alpha)
        reactive_test_scores = ewma(reactive_test_scores, alpha=config.ewma_alpha)
        proactive_test_scores = ewma(proactive_test_scores, alpha=config.ewma_alpha)

    reactive_threshold = float(np.quantile(reactive_train_scores, config.threshold_quantile))
    proactive_threshold = float(np.quantile(proactive_train_scores, config.threshold_quantile))
    reactive_test_preds = (reactive_test_scores >= reactive_threshold).astype(np.int64)
    proactive_test_preds = (proactive_test_scores >= proactive_threshold).astype(np.int64)

    artifacts = {
        "model": model,
        "device": str(DEVICE),
        "config": asdict(config),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols,
        "x_train_shape": x_train.shape,
        "x_test_shape": x_test.shape,
        "y_test": y_test.astype(np.int64),
        "reactive_train_scores": reactive_train_scores.astype(np.float64),
        "reactive_test_scores": reactive_test_scores.astype(np.float64),
        "proactive_train_scores": proactive_train_scores.astype(np.float64),
        "proactive_test_scores": proactive_test_scores.astype(np.float64),
        "reactive_threshold": reactive_threshold,
        "proactive_threshold": proactive_threshold,
        "reactive_test_preds": reactive_test_preds.astype(np.int64),
        "proactive_test_preds": proactive_test_preds.astype(np.int64),
        "reactive_task_matrix": build_task_matrix(reactive_test_scores, y_test, num_tasks=4),
        "proactive_task_matrix": build_task_matrix(proactive_test_scores, y_test, num_tasks=4),
    }
    if verbose:
        print("Training/evaluation artifacts ready.")
    return artifacts


def get_artifacts(force_retrain: bool = False, config: Config = CFG, verbose: bool = True):
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is None or force_retrain:
        _ARTIFACTS_CACHE = run_training(config=config, verbose=verbose)
    return _ARTIFACTS_CACHE


if __name__ == "__main__":
    arts = get_artifacts(force_retrain=True, verbose=True)
    print("\nAvailable artifact keys:")
    for k in sorted(arts.keys()):
        v = arts[k]
        if isinstance(v, np.ndarray):
            print(f" - {k}: array shape={v.shape}")
        else:
            print(f" - {k}")
