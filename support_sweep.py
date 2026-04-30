"""Support Devices calibration sweep from cached logits."""
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from pathlib import Path

from config import Config
from dataset import _drop_junk_cols, _extract_pid


def main():
    cfg = Config.from_yaml("configs/hpc_densenet_v1.yaml")
    df = pd.read_csv(cfg.labels_csv)
    df = _drop_junk_cols(df)
    df["pid"] = _extract_pid(df["Path"])
    df = df[df["pid"].notna()].reset_index(drop=True)
    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(pids)
    n_val = int(round(len(pids) * cfg.val_frac))
    val_pids = set(pids[:n_val].tolist())
    df_val = df[df["pid"].isin(val_pids)].reset_index(drop=True)
    y_val_raw = df_val[cfg.label_names].to_numpy(dtype=np.float32)

    SD_IDX = 8
    yt = y_val_raw[:, SD_IDX]
    mask = ~np.isnan(yt)
    yt_m = yt[mask]
    train_mean = float(np.mean(yt_m))
    var = float(np.var(yt_m))
    print(f"Support Devices: {mask.sum()} valid, train_mean={train_mean:.4f}, var={var:.4f}")
    print()

    def raw_nmse(pred):
        return np.mean((yt_m - pred[mask]) ** 2) / var

    cache_dir = Path("/resnick/groups/CS156b/from_central/2026/scalm_akumarap/runs")
    families = {}
    for name in ["calib_cache", "raw_mse_calib_cache", "raw3_calib_cache", "raw320_calib_cache",
                 "rawwt_calib_cache", "raw_wt5_calib_cache", "raw_augstrong_calib_cache",
                 "raw_rmsprop_calib_cache", "raw_umask_calib_cache", "3class_calib_cache"]:
        vf = cache_dir / name / "val_ensemble_cache.npz"
        if vf.exists():
            d = np.load(str(vf))
            families[name] = d["logits"][:, SD_IDX]

    print(f"Loaded {len(families)} families")
    print()

    def fit_affine(x):
        xm = x[mask]
        def obj(p):
            return np.mean((yt_m - np.clip(p[0] * xm + p[1], -1, 1)) ** 2)
        best, best_mse = None, float("inf")
        for a0 in [-2, -1, 0, 1, 2, 4]:
            for b0 in [-1, -0.5, 0, 0.5, 1]:
                r = minimize(obj, [a0, b0], method="Nelder-Mead",
                             options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
                if r.fun < best_mse:
                    best_mse, best = r.fun, r
        return best.x[0], best.x[1], best_mse / var

    print(f"{'Method':45s}  {'Val NMSE':>10s}")
    print("-" * 60)

    # 1. 3-class identity
    tc = families["3class_calib_cache"]
    nmse_id = raw_nmse(np.clip(tc, -1, 1))
    print(f"{'3class identity':45s}  {nmse_id:10.4f}")

    # 2. 3-class affine
    a, b, nmse_aff = fit_affine(tc)
    print(f"{'3class affine (a={:.2f} b={:.2f})'.format(a, b):45s}  {nmse_aff:10.4f}")

    # 3. Constant mean
    nmse_const = 1.0  # by definition
    print(f"{'constant mean ({:.3f})'.format(train_mean):45s}  {nmse_const:10.4f}")

    # 4. All families affine
    print()
    print("All families (affine calibrated):")
    results = []
    for name, logits in sorted(families.items()):
        a2, b2, n2 = fit_affine(logits)
        results.append((n2, name, a2, b2))
    results.sort()
    for n2, name, a2, b2 in results[:8]:
        marker = " <-- BEST" if n2 == results[0][0] else ""
        print(f"  {name:40s}  {n2:8.4f}  a={a2:.3f} b={b2:.3f}{marker}")

    # 5. Convex blend: lam * best_affine + (1-lam) * train_mean
    best_family_name = results[0][1]
    best_family_logits = families[best_family_name]
    a_best, b_best = results[0][2], results[0][3]
    pred_best = np.clip(a_best * best_family_logits + b_best, -1, 1)

    print()
    print(f"Convex blend ({best_family_name} affine + train_mean):")
    best_lam, best_blend_nmse = 1.0, results[0][0]
    for lam in np.arange(0.0, 1.05, 0.05):
        blend = lam * pred_best + (1 - lam) * train_mean
        blend = np.clip(blend, -1, 1)
        n = raw_nmse(blend)
        if n < best_blend_nmse:
            best_blend_nmse = n
            best_lam = lam
    print(f"  best lambda={best_lam:.2f}: {best_blend_nmse:.4f} (pure affine was {results[0][0]:.4f})")

    # 6. Power sharpening
    print()
    print("Power sharpening (push predictions toward +1):")
    pred_01 = (pred_best + 1) / 2
    best_gamma_nmse = results[0][0]
    best_gamma_params = None
    for gamma in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for c in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            sharpened = 1 - c * (1 - pred_01) ** gamma
            sharpened_raw = np.clip(sharpened * 2 - 1, -1, 1)
            n = raw_nmse(sharpened_raw)
            if n < best_gamma_nmse:
                best_gamma_nmse = n
                best_gamma_params = (gamma, c)
    if best_gamma_params:
        print(f"  IMPROVED: gamma={best_gamma_params[0]}, c={best_gamma_params[1]}: {best_gamma_nmse:.4f} (was {results[0][0]:.4f})")
    else:
        print(f"  no improvement over affine ({results[0][0]:.4f})")

    # 7. Multi-family ensemble for Support Devices
    print()
    print("Multi-family average (top 3 families, affine each):")
    top3_preds = []
    for n2, name, a2, b2 in results[:3]:
        top3_preds.append(np.clip(a2 * families[name] + b2, -1, 1))
    avg3 = np.mean(top3_preds, axis=0)
    print(f"  top-3 ensemble: {raw_nmse(avg3):.4f}")

    print()
    print("=" * 60)
    print(f"SUMMARY: best Support Devices val NMSE found = {min(best_blend_nmse, best_gamma_nmse, raw_nmse(avg3)):.4f}")
    print(f"Current mega hybrid uses: {nmse_aff:.4f}")


if __name__ == "__main__":
    main()
