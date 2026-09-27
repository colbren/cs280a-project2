# Part 1.3: Derivative of Gaussian (DoG) Filter
#
# (a) Blur the cameraman with a Gaussian G, then apply Dx / Dy (two convolutions).
# (b) Build DoG filters G * Dx and G * Dy and apply them directly (one convolution).
# Show the gradient magnitude / edge images for both and verify they match.

import os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

SIGMA = 2.0
KSIZE = int(6 * SIGMA + 1)  # covers +-3 sigma
THRESHOLD_FD = 0.25         # threshold chosen in part 1.2 (no smoothing)
THRESHOLD = 0.05            # threshold chosen here (smoothing lowers the magnitudes)
THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1]

Dx = np.array([[1, -1]], dtype=float)
Dy = np.array([[1], [-1]], dtype=float)


def load_gray(path):
    im = plt.imread(path).astype(float)
    if im.max() > 1.0:
        im /= 255.0
    if im.ndim == 3:
        im = im[..., :3] @ np.array([0.299, 0.587, 0.114])
    return im


def crop_white_border(im, tol=0.99):
    """cameraman.png has a thin pure-white screenshot margin; trim it so it
    doesn't show up as a fake rectangular edge."""
    rows = np.where(im.min(axis=1) < tol)[0]
    cols = np.where(im.min(axis=0) < tol)[0]
    return im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def gaussian_kernel_2d(ksize, sigma):
    g = cv2.getGaussianKernel(ksize, sigma)  # (ksize, 1), sums to 1
    return g @ g.T


def conv(im, k):
    return convolve2d(im, k, mode="same", boundary="symm")


def gradient(im):
    dx, dy = conv(im, Dx), conv(im, Dy)
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def signed_display(im):
    m = np.abs(im).max() + 1e-12
    return 0.5 + 0.5 * im / m


def save(name, im):
    plt.imsave(os.path.join(OUT, name), np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)


def row_figure(panels, suptitle, fname):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5))
    for a, (p, t) in zip(axes, panels):
        a.imshow(p, cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=13)
        a.axis("off")
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=100)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    im = crop_white_border(load_gray(os.path.join(HERE, "cameraman.png")))
    G = gaussian_kernel_2d(KSIZE, SIGMA)

    # --- finite differences on the raw image (part 1.2 baseline) ---
    _, _, mag_fd = gradient(im)
    edges_fd = mag_fd > THRESHOLD_FD

    # --- (a) blur first, then finite differences: two convolutions ---
    blurred = conv(im, G)
    dx_a, dy_a, mag_a = gradient(blurred)
    edges_a = mag_a > THRESHOLD

    save("cameraman_blurred.png", blurred)
    save("blur_then_diff_dx.png", signed_display(dx_a))
    save("blur_then_diff_dy.png", signed_display(dy_a))
    save("blur_then_diff_grad_mag.png", mag_a / mag_a.max())
    save(f"blur_then_diff_edges_thresh_{THRESHOLD}.png", edges_a.astype(float))

    # --- (b) derivative of Gaussian filters: one convolution ---
    DoG_x = convolve2d(G, Dx, mode="full")  # (KSIZE, KSIZE + 1)
    DoG_y = convolve2d(G, Dy, mode="full")  # (KSIZE + 1, KSIZE)
    dx_b, dy_b = conv(im, DoG_x), conv(im, DoG_y)
    mag_b = np.sqrt(dx_b ** 2 + dy_b ** 2)
    edges_b = mag_b > THRESHOLD

    save("dog_dx.png", signed_display(dx_b))
    save("dog_dy.png", signed_display(dy_b))
    save("dog_grad_mag.png", mag_b / mag_b.max())
    save(f"dog_edges_thresh_{THRESHOLD}.png", edges_b.astype(float))

    # --- filter visualizations ---
    def upscale(k, s=20):  # nearest-neighbour so individual taps are visible
        return np.kron(k, np.ones((s, s)))
    save("filter_gaussian.png", upscale(G / G.max()))
    save("filter_dog_x.png", upscale(signed_display(DoG_x)))
    save("filter_dog_y.png", upscale(signed_display(DoG_y)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for a, (k, t) in zip(axes, [(G, f"Gaussian G  ({KSIZE}x{KSIZE}, σ={SIGMA})"),
                                (DoG_x, f"DoG_x = G * Dx  {DoG_x.shape}"),
                                (DoG_y, f"DoG_y = G * Dy  {DoG_y.shape}")]):
        m = np.abs(k).max()
        cmap, lo = ("gray", 0) if k is G else ("RdBu_r", -m)
        img = a.imshow(k, cmap=cmap, vmin=lo, vmax=m)
        fig.colorbar(img, ax=a, fraction=0.046)
        a.set_title(t, fontsize=12)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Part 1.3: Gaussian and derivative of Gaussian filters", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dog_filters_figure.png"), dpi=100)
    plt.close(fig)

    # --- verification: (a) vs (b) ---
    # Convolution is associative, so (I * G) * Dx == I * (G * Dx). We check
    # this with the symmetric boundary used above, away from the border, and
    # with plain zero padding + 'full' output (where it holds exactly).
    b = KSIZE
    diff = np.abs(mag_a - mag_b)
    diff_interior = diff[b:-b, b:-b]
    zero_a = np.sqrt(convolve2d(convolve2d(im, G), Dx) ** 2)
    zero_b = np.sqrt(convolve2d(im, DoG_x) ** 2)
    diff_zero = np.abs(zero_a - zero_b).max()
    edges_disagree = np.mean(edges_a != edges_b) * 100

    # --- labeled figures ---
    row_figure([(im, "Original"),
                (blurred, f"Blurred (G, σ={SIGMA}, {KSIZE}x{KSIZE})"),
                (signed_display(dx_a), "(I * G) * Dx"),
                (signed_display(dy_a), "(I * G) * Dy"),
                (mag_a / mag_a.max(), f"Gradient magnitude\nmax = {mag_a.max():.3f}"),
                (edges_a, f"Edges: |∇| > {THRESHOLD}")],
               "Part 1.3a: blur with a Gaussian, then finite differences (two convolutions)",
               "blur_then_diff_figure.png")
    row_figure([(signed_display(dx_b), "I * DoG_x"),
                (signed_display(dy_b), "I * DoG_y"),
                (mag_b / mag_b.max(), f"Gradient magnitude\nmax = {mag_b.max():.3f}"),
                (edges_b, f"Edges: |∇| > {THRESHOLD}"),
                (diff / max(diff.max(), 1e-12),
                 f"|mag(a) - mag(b)| (rescaled)\nmax = {diff.max():.1e}, interior max = {diff_interior.max():.1e}")],
               "Part 1.3b: derivative of Gaussian filters (single convolution)",
               "dog_figure.png")
    row_figure([(mag_fd / mag_fd.max(), "Finite difference: gradient magnitude"),
                (edges_fd, f"Finite difference: edges (|∇| > {THRESHOLD_FD})"),
                (mag_b / mag_b.max(), f"DoG σ={SIGMA}: gradient magnitude"),
                (edges_b, f"DoG σ={SIGMA}: edges (|∇| > {THRESHOLD})")],
               "Part 1.2 vs 1.3: effect of Gaussian smoothing",
               "fd_vs_dog_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 10.8))
    for a, t in zip(axes.flat, THRESHOLDS):
        a.imshow(mag_b > t, cmap="gray")
        chosen = "  (chosen)" if t == THRESHOLD else ""
        a.set_title(f"threshold = {t}{chosen}", fontsize=13, color="tab:red" if chosen else "black")
        a.axis("off")
    fig.suptitle(f"Part 1.3: DoG (σ={SIGMA}) gradient magnitude thresholds", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "threshold_sweep.png"), dpi=100)
    plt.close(fig)

    report = "\n".join([
        f"Gaussian: sigma = {SIGMA}, kernel = {KSIZE}x{KSIZE}",
        f"DoG_x shape {DoG_x.shape}, DoG_y shape {DoG_y.shape}",
        f"Finite-difference magnitude max: {mag_fd.max():.4f} (threshold {THRESHOLD_FD})",
        f"Blurred magnitude max: {mag_a.max():.4f} (threshold {THRESHOLD})",
        f"Edge pixels: finite diff {edges_fd.mean() * 100:.2f}%, blur+diff {edges_a.mean() * 100:.2f}%, "
        f"DoG {edges_b.mean() * 100:.2f}%",
        "",
        "Verification (blur-then-diff vs single DoG convolution):",
        f"  max |mag_a - mag_b| over whole image (symm boundary): {diff.max():.2e}",
        f"  max |mag_a - mag_b| ignoring a {b}px border:          {diff_interior.max():.2e}",
        f"  max diff with zero padding + 'full' convolution:       {diff_zero:.2e}",
        f"  edge pixels that disagree:                              {edges_disagree:.4f}%",
    ])
    print(report)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
