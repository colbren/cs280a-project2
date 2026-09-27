# Part 2.2: Hybrid Images (Oliva, Torralba & Schyns 2006), grayscale.
#
#   hybrid = (im1 - im1 * G_sigma1) + (im2 * G_sigma2)
#            high frequencies of im1   low frequencies of im2
#
# Up close the high-frequency image dominates; from far away only the
# low-frequency image is visible.

import os
import argparse

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from align_image_code import align_images

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

# Alignment points are (x, y) pixel coordinates of two corresponding features
# (eyes for faces, wheel hubs for bikes), listed in the same order in both
# images. They were picked by hand; run with --interactive to click new ones.
PAIRS = [
    dict(name="nutmeg_derek",
         high="nutmeg.jpg", low="DerekPicture.jpg",
         pts_high=((604, 292), (752, 372)), pts_low=((302, 338), (440, 330)),
         sigma_high=5, sigma_low=10,
         desc="Nutmeg up close, Derek from far away"),
    dict(name="happy_mad_baby",
         high="happy_baby_hybrid.jpg", low="mad_baby_hybrid.jpg",
         pts_high=((372, 226), (455, 218)), pts_low=((490, 333), (665, 330)),
         sigma_high=3, sigma_low=6,
         desc="Change of expression: laughing baby up close, grumpy baby from far away"),
    dict(name="bicycle_motorcycle",
         high="bicycle_hybrid.png", low="motorcycle_hybrid.png",
         pts_high=((199, 678), (873, 683)), pts_low=((725, 1318), (2100, 1315)),
         sigma_high=4, sigma_low=10,
         desc="Morph between objects: bicycle up close, motorcycle from far away"),
]

# Pair used for the full frequency-analysis figure.
FAVORITE = "happy_mad_baby"


def imread(path):
    im = plt.imread(os.path.join(HERE, path)).astype(float)
    if im.max() > 1.0:
        im /= 255.0
    return im[..., :3]


def to_gray(im):
    return im @ np.array([0.299, 0.587, 0.114]) if im.ndim == 3 else im


def gaussian_blur(im, sigma):
    """Separable Gaussian (same result as the 2D outer-product kernel, much
    faster for large sigma). Works on grayscale or per channel on color."""
    ksize = int(6 * sigma + 1) | 1
    g = cv2.getGaussianKernel(ksize, sigma)
    if im.ndim == 3:
        return np.dstack([gaussian_blur(im[..., c], sigma) for c in range(im.shape[2])])
    tmp = convolve2d(im, g.T, mode="same", boundary="symm")
    return convolve2d(tmp, g, mode="same", boundary="symm")


def low_pass(im, sigma):
    return gaussian_blur(im, sigma)


def high_pass(im, sigma):
    # impulse minus Gaussian
    return im - gaussian_blur(im, sigma)


def hybrid_image(im1, im2, sigma1, sigma2):
    """High frequencies of im1 (cutoff sigma1) + low frequencies of im2 (cutoff sigma2)."""
    return high_pass(im1, sigma1) + low_pass(im2, sigma2)


def largest_valid_rectangle(mask):
    """Largest axis-aligned all-True rectangle in a boolean mask
    (histogram / stack method). Returns (r0, r1, c0, c1)."""
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    best = (0, 0, h, 0, w)
    for r in range(h):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []  # (start column, height)
        for c in range(w + 1):
            cur = heights[c] if c < w else 0
            start = c
            while stack and stack[-1][1] >= cur:
                s, hgt = stack.pop()
                if hgt * (c - s) > best[0]:
                    best = (hgt * (c - s), r - hgt + 1, r + 1, s, c)
                start = s
            stack.append((start, cur))
    return best[1:]


def load_aligned(pair, interactive=False):
    """Align the pair, then crop away the black padding / rotation corners
    that alignment introduces (they would otherwise show up as strong fake
    edges in the high-pass image). A constant 4th channel is carried
    through the alignment to track which pixels are real."""
    im_high, im_low = imread(pair["high"]), imread(pair["low"])
    with_valid = lambda im: np.dstack([im, np.ones(im.shape[:2])])
    pts = None if interactive else (*pair["pts_high"], *pair["pts_low"])
    a_high, a_low = align_images(with_valid(im_high), with_valid(im_low), pts)
    valid = (a_high[..., 3] > 0.99) & (a_low[..., 3] > 0.99)
    r0, r1, c0, c1 = largest_valid_rectangle(valid)
    a_high, a_low = a_high[r0:r1, c0:c1, :3], a_low[r0:r1, c0:c1, :3]
    if interactive:
        print(f"{pair['name']}: aligned + cropped shape {a_high.shape}")
    return np.clip(a_high, 0, 1), np.clip(a_low, 0, 1)


def log_fft(gray):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def show_high(im):
    """High-pass images are zero-mean; shift to gray for display."""
    return np.clip(im + 0.5, 0, 1)


def downscale(im, f):
    h, w = im.shape[:2]
    return cv2.resize(im, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)


def save(out_dir, name, im):
    im = np.clip(im, 0, 1)
    plt.imsave(os.path.join(out_dir, name), im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)


def distance_figure(hybrid, title, path):
    """Show the hybrid at shrinking sizes to simulate viewing it from farther away."""
    scales = [1, 2, 4, 8]
    widths = [1 / s for s in scales]
    fig, axes = plt.subplots(1, len(scales), figsize=(16, 5.2),
                             gridspec_kw={"width_ratios": widths})
    for a, s in zip(axes, scales):
        a.imshow(np.clip(hybrid, 0, 1), cmap="gray" if hybrid.ndim == 2 else None, vmin=0, vmax=1)
        a.set_title(f"1/{s} size" if s > 1 else "full size", fontsize=11)
        a.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def frequency_figure(gray_high, gray_low, hp, lp, hybrid, pair, path):
    ffts = [log_fft(x) for x in (gray_high, gray_low, hp, lp, hybrid)]
    lo, hi = min(f.min() for f in ffts[:2]), max(f.max() for f in ffts[:2])
    lo = max(lo, hi - 14)  # keep a common, readable dynamic range
    titles = [f"Input 1 (high): {pair['high']}", f"Input 2 (low): {pair['low']}",
              f"High-pass of input 1\n(im − im * G, σ={pair['sigma_high']})",
              f"Low-pass of input 2\n(im * G, σ={pair['sigma_low']})", "Hybrid image"]
    images = [gray_high, gray_low, show_high(hp), lp, hybrid]
    fig, axes = plt.subplots(2, 5, figsize=(24, 10.5))
    for j in range(5):
        axes[0, j].imshow(np.clip(images[j], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(titles[j], fontsize=12)
        axes[1, j].imshow(ffts[j], cmap="viridis", vmin=lo, vmax=hi)
        axes[1, j].set_title("log |FFT|", fontsize=12)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"Part 2.2 frequency analysis: {pair['desc']}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                        help="click alignment points instead of using the saved ones")
    args = parser.parse_args()
    if not args.interactive:
        matplotlib.use("Agg")
    os.makedirs(OUT, exist_ok=True)
    lines = []

    for pair in PAIRS:
        name = pair["name"]
        a_high, a_low = load_aligned(pair, args.interactive)
        g_high, g_low = to_gray(a_high), to_gray(a_low)

        hp = high_pass(g_high, pair["sigma_high"])
        lp = low_pass(g_low, pair["sigma_low"])
        hybrid = np.clip(hp + lp, 0, 1)

        save(OUT, f"{name}_input_high.png", g_high)
        save(OUT, f"{name}_input_low.png", g_low)
        save(OUT, f"{name}_high_pass.png", show_high(hp))
        save(OUT, f"{name}_low_pass.png", lp)
        save(OUT, f"{name}_hybrid.png", hybrid)
        lines.append(f"{name}: {pair['desc']}\n"
                     f"  high-frequency image: {pair['high']} (sigma_high = {pair['sigma_high']})\n"
                     f"  low-frequency image:  {pair['low']} (sigma_low = {pair['sigma_low']})\n"
                     f"  aligned size: {hybrid.shape[1]}x{hybrid.shape[0]}")

        # inputs + result
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for a, (p, t) in zip(axes, [(g_high, f"High-frequency input\n{pair['high']}  (σ={pair['sigma_high']})"),
                                    (g_low, f"Low-frequency input\n{pair['low']}  (σ={pair['sigma_low']})"),
                                    (hybrid, "Hybrid")]):
            a.imshow(p, cmap="gray", vmin=0, vmax=1)
            a.set_title(t, fontsize=12)
            a.axis("off")
        fig.suptitle(f"Part 2.2: {pair['desc']}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(OUT, f"{name}_figure.png"), dpi=100)
        plt.close(fig)

        distance_figure(hybrid, f"{pair['desc']} (shrink = step back)",
                        os.path.join(OUT, f"{name}_distance.png"))
        frequency_figure(g_high, g_low, hp, lp, hybrid, pair,
                         os.path.join(OUT, f"{name}_frequency_analysis.png"))

    lines.append(f"\nFavorite (for the frequency analysis on the website): {FAVORITE}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
