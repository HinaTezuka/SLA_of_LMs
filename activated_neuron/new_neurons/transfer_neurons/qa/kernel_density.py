# -*- coding: utf-8 -*-
# PCA(sklearn) -> 2D + KDE(2D,1D) with marginals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------- 2D KDE ----------
def kde2d_grid(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, bandwidth: float = 0.5) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (G,2)
    n = len(pts)
    if n == 0:
        return np.zeros((len(grid_y), len(grid_x)))
    diff = grid[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)  # (G,n)
    h2 = bandwidth ** 2
    coef = 1.0 / (2.0 * np.pi * h2)
    K = coef * np.exp(-0.5 * dist2 / h2)
    dens = K.sum(axis=1) / n
    return dens.reshape(len(grid_y), len(grid_x))

# ---------- 1D KDE ----------
def kde1d_grid(x: np.ndarray, grid: np.ndarray, bandwidth: float = 0.4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros_like(grid, dtype=float)
    diff = grid[:, None] - x[None, :]
    h2 = bandwidth ** 2
    coef = 1.0 / np.sqrt(2.0 * np.pi * h2)
    K = coef * np.exp(-0.5 * (diff * diff) / h2)
    return K.mean(axis=1)

# ---------- メイン関数 ----------
def plot_pca_kde_with_marginals(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth2d: float = 0.6,
    bandwidth1d: float = 0.4,
    gridsize: int = 220,
    figsize=(8, 8)
):
    # PCA (sklearn)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)  # (n,2)
    Z0, Z1 = Z[y==0], Z[y==1]

    # グリッド範囲
    all_min = Z.min(axis=0)
    all_max = Z.max(axis=0)
    pad = 0.05 * (all_max - all_min + 1e-9)
    x_min, y_min = (all_min - pad)
    x_max, y_max = (all_max + pad)
    gx = np.linspace(x_min, x_max, gridsize)
    gy = np.linspace(y_min, y_max, gridsize)

    dens0 = kde2d_grid(Z0, gx, gy, bandwidth=bandwidth2d) if len(Z0) else np.zeros((gridsize, gridsize))
    dens1 = kde2d_grid(Z1, gx, gy, bandwidth=bandwidth2d) if len(Z1) else np.zeros((gridsize, gridsize))

    # Figureレイアウト
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top  = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right= fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 中央 2D KDE + 散布
    ax_main.contourf(gx, gy, dens0, levels=12, alpha=0.45)
    ax_main.contourf(gx, gy, dens1, levels=12, alpha=0.45)
    if len(Z0): ax_main.scatter(Z0[:,0], Z0[:,1], s=6, alpha=0.5, label="label=0")
    if len(Z1): ax_main.scatter(Z1[:,0], Z1[:,1], s=6, alpha=0.5, label="label=1")
    ax_main.set_xlabel("PC1")
    ax_main.set_ylabel("PC2")
    ax_main.legend(loc="upper right")
    ax_main.grid(True, ls=":", alpha=0.4)

    # 上 1D KDE (PC1)
    if len(Z0): ax_top.plot(gx, kde1d_grid(Z0[:,0], gx, bandwidth1d), label="0")
    if len(Z1): ax_top.plot(gx, kde1d_grid(Z1[:,0], gx, bandwidth1d), label="1")
    ax_top.set_ylabel("density")
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, ls=":", alpha=0.4)

    # 右 1D KDE (PC2)
    if len(Z0): ax_right.plot(kde1d_grid(Z0[:,1], gy, bandwidth1d), gy, label="0")
    if len(Z1): ax_right.plot(kde1d_grid(Z1[:,1], gy, bandwidth1d), gy, label="1")
    ax_right.set_xlabel("density")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, ls=":", alpha=0.4)

    fig.suptitle("PCA (sklearn) -> 2D KDE with marginals (label 0/1)")
    plt.tight_layout()
    plt.show()

# ---------------- 使用例 ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n0, n1 = 300, 250
    X0 = rng.normal(0, 1, (n0, 128))
    X1 = rng.normal(0.5, 1, (n1, 128))
    X_demo = np.vstack([X0, X1])
    y_demo = np.array([0]*n0 + [1]*n1)
    plot_pca_kde_with_marginals(X_demo, y_demo, bandwidth2d=0.6, bandwidth1d=0.4)