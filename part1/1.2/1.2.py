# Part 1.2: Finite Difference Operator
#
# Partial derivatives of the cameraman image with Dx and Dy, the gradient
# magnitude, and a binarized edge image from thresholding the magnitude.

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

Dx = np.array([[1, -1]], dtype=float)
Dy = np.array([[1], [-1]], dtype=float)

# Thresholds tried (image intensities are in [0, 1]); THRESHOLD is the one
# picked by eye as the best tradeoff between noise and real edges.
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
THRESHOLD = 0.25


def load_gray(path):
    im = plt.imread(path).astype(float)
    if im.max() > 1.0:
        im /= 255.0
    if im.ndim == 3:
        im = im[..., :3] @ np.array([0.299, 0.587, 0.114])
    return im


def crop_white_border(im, tol=0.99):
    """The provided cameraman.png is a screenshot with a thin pure-white
    margin around it, which would show up as a fake rectangular edge.
    Trim any all-white rows/columns from the outside."""
    rows = np.where(im.min(axis=1) < tol)[0]
    cols = np.where(im.min(axis=0) < tol)[0]
    return im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def conv(im, k):
    # symmetric boundary so the image border doesn't produce fake edges
    return convolve2d(im, k, mode="same", boundary="symm")


def signed_display(im):
    m = np.abs(im).max() + 1e-12
    return 0.5 + 0.5 * im / m


def save(name, im):
    plt.imsave(os.path.join(OUT, name), np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    im = crop_white_border(load_gray(os.path.join(HERE, "cameraman.png")))

    dx = conv(im, Dx)
    dy = conv(im, Dy)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    edges = mag > THRESHOLD

    save("cameraman_gray.png", im)
    save("cameraman_dx.png", signed_display(dx))
    save("cameraman_dy.png", signed_display(dy))
    save("cameraman_grad_mag.png", mag / mag.max())
    save(f"cameraman_edges_thresh_{THRESHOLD}.png", edges.astype(float))

    # --- labeled summary figure ---
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    panels = [(im, "Original"),
              (signed_display(dx), "∂I/∂x  (I * Dx)"),
              (signed_display(dy), "∂I/∂y  (I * Dy)"),
              (mag / mag.max(), f"Gradient magnitude\nmax = {mag.max():.3f}"),
              (edges, f"Binarized edges\n|∇I| > {THRESHOLD}")]
    for a, (p, t) in zip(axes, panels):
        a.imshow(p, cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=13)
        a.axis("off")
    fig.suptitle("Part 1.2: finite difference operators  Dx = [1, -1],  Dy = [1, -1]ᵀ", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "summary_figure.png"), dpi=100)
    plt.close(fig)

    # --- threshold sweep ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 10.5))
    for a, t in zip(axes.flat, THRESHOLDS):
        a.imshow(mag > t, cmap="gray")
        chosen = "  (chosen)" if t == THRESHOLD else ""
        a.set_title(f"threshold = {t}{chosen}", fontsize=13,
                    color="tab:red" if chosen else "black")
        a.axis("off")
    fig.suptitle("Part 1.2: gradient magnitude thresholds (finite differences, no smoothing)", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "threshold_sweep.png"), dpi=100)
    plt.close(fig)

    report = (f"Gradient magnitude range: [{mag.min():.4f}, {mag.max():.4f}]\n"
              f"Thresholds tried: {THRESHOLDS}\n"
              f"Chosen threshold: {THRESHOLD}  ({edges.mean() * 100:.2f}% of pixels marked as edges)\n")
    print(report)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
