# Part 1.3 Bells & Whistles: gradient orientation visualized in HSV.
#
# Using the DoG gradients from 1.3:
#   hue        = gradient orientation  theta = atan2(dy, dx), mapped to [0, 1)
#   saturation = 1
#   value      = gradient magnitude (normalized), so flat regions are black
# Hue is cyclic, which matches orientation being an angle (-pi and pi are the
# same direction), so there's no artificial jump in color.

import os
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_bells_and_whistles")

# reuse the helpers / parameters from 1.3.py (file name isn't importable directly)
_spec = importlib.util.spec_from_file_location("part1_3", os.path.join(HERE, "1.3.py"))
p13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p13)

# magnitude at which the brightness saturates (as a percentile of the magnitude)
VALUE_PERCENTILE = 99


def orientation_hsv(dx, dy, mag, mask=None):
    theta = np.arctan2(dy, dx)                  # in (-pi, pi]
    hue = (theta + np.pi) / (2 * np.pi)         # in (0, 1]
    value = np.clip(mag / np.percentile(mag, VALUE_PERCENTILE), 0, 1)
    if mask is not None:
        value = mask.astype(float)
    hsv = np.stack([hue % 1.0, np.ones_like(hue), value], axis=-1)
    return hsv_to_rgb(hsv), theta


def color_wheel(size=256):
    """Legend: hue at each point = the orientation that color encodes."""
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    # y increases downward (image rows), matching the gradient convention:
    # with Dy = [1, -1]^T, dy > 0 when the image gets brighter going down,
    # and with Dx = [1, -1], dx > 0 when it gets brighter going right.
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    hsv = np.stack([((theta + np.pi) / (2 * np.pi)) % 1.0,
                    np.ones_like(r),
                    np.clip(r, 0, 1) * (r <= 1)], axis=-1)
    return hsv_to_rgb(hsv)


def main():
    os.makedirs(OUT, exist_ok=True)
    im = p13.crop_white_border(p13.load_gray(os.path.join(HERE, "cameraman.png")))
    G = p13.gaussian_kernel_2d(p13.KSIZE, p13.SIGMA)
    DoG_x = p13.convolve2d(G, p13.Dx, mode="full")
    DoG_y = p13.convolve2d(G, p13.Dy, mode="full")

    dx, dy = p13.conv(im, DoG_x), p13.conv(im, DoG_y)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    edges = mag > p13.THRESHOLD

    rgb_mag, theta = orientation_hsv(dx, dy, mag)
    rgb_edges, _ = orientation_hsv(dx, dy, mag, mask=edges)
    rgb_full = hsv_to_rgb(np.stack([(theta + np.pi) / (2 * np.pi) % 1.0,
                                    np.ones_like(theta), np.ones_like(theta)], -1))
    wheel = color_wheel()

    plt.imsave(os.path.join(OUT, "orientation_hsv_magnitude.png"), rgb_mag)
    plt.imsave(os.path.join(OUT, "orientation_hsv_edges.png"), rgb_edges)
    plt.imsave(os.path.join(OUT, "orientation_hue_only.png"), rgb_full)
    plt.imsave(os.path.join(OUT, "orientation_color_wheel.png"), wheel)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.6]})
    panels = [(im, "Original", "gray"),
              (rgb_full, "Hue = orientation\n(V = 1 everywhere, noisy in flat areas)", None),
              (rgb_mag, f"Hue = orientation, Value = |∇I|\n(DoG σ={p13.SIGMA})", None),
              (rgb_edges, f"Hue = orientation on edge pixels\n(|∇I| > {p13.THRESHOLD})", None),
              (wheel, "Legend: hue ↔ gradient direction\n(→ = brighter to the right)", None)]
    for a, (p, t, cmap) in zip(axes, panels):
        a.imshow(p, cmap=cmap)
        a.set_title(t, fontsize=12)
        a.axis("off")
    fig.suptitle("Part 1.3 Bells & Whistles: gradient orientation in HSV color space", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "orientation_figure.png"), dpi=100)
    plt.close(fig)

    # Same thing with a matplotlib cyclic colormap (twilight) on edge pixels.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    shown = np.ma.masked_where(~edges, theta)
    axes[0].imshow(np.zeros_like(im), cmap="gray")
    img = axes[0].imshow(shown, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Edge orientation, 'twilight' cyclic colormap", fontsize=12)
    axes[0].axis("off")
    cb = fig.colorbar(img, ax=axes[0], fraction=0.046, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    axes[1].hist(theta[edges], bins=72, range=(-np.pi, np.pi), color="gray")
    axes[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[1].set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    axes[1].set_xlabel("orientation θ = atan2(dy, dx)")
    axes[1].set_ylabel("# edge pixels")
    axes[1].set_title("Histogram of edge orientations", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "orientation_twilight_and_histogram.png"), dpi=100)
    plt.close(fig)

    print(f"Saved orientation visualizations to {OUT}")


if __name__ == "__main__":
    main()
