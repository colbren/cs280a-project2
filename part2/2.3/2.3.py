# Part 2.3: Gaussian and Laplacian Stacks (implemented from scratch).
#
# Gaussian stack:  G_0 = image,  G_{i+1} = G_i * g(sigma_i)     (no downsampling)
# Laplacian stack: L_i = G_i - G_{i+1}  for i < N,  L_N = G_N   (so sum(L) = image)
#
# sigma_i doubles every level (sigma_i = SIGMA0 * 2^i), so the effective blur
# roughly doubles per level, mimicking a pyramid that halves resolution each level.
#
# Then recreate Figure 3.42 of Szeliski (2nd ed.): Laplacian levels 0, 2, 4 of the
# masked apple, the masked orange, their sum, and the fully collapsed stacks.

import os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

N_LEVELS = 6      # Laplacian levels 0..5 (level 5 is the low-pass residual)
SIGMA0 = 2.0


def imread(path):
    im = plt.imread(path).astype(float)
    if im.max() > 1.0:
        im /= 255.0
    return im[..., :3] if im.ndim == 3 else im


def gaussian_blur(im, sigma):
    """Separable Gaussian blur, applied per channel for color images."""
    ksize = int(6 * sigma + 1) | 1
    g = cv2.getGaussianKernel(ksize, sigma)
    if im.ndim == 3:
        return np.dstack([gaussian_blur(im[..., c], sigma) for c in range(im.shape[2])])
    tmp = convolve2d(im, g.T, mode="same", boundary="symm")
    return convolve2d(tmp, g, mode="same", boundary="symm")


def gaussian_stack(im, n_levels=N_LEVELS, sigma0=SIGMA0):
    """Returns n_levels images; level 0 is the input."""
    stack = [im]
    for i in range(n_levels - 1):
        stack.append(gaussian_blur(stack[-1], sigma0 * 2 ** i))
    return stack


def laplacian_stack(im, n_levels=N_LEVELS, sigma0=SIGMA0):
    g = gaussian_stack(im, n_levels, sigma0)
    return [g[i] - g[i + 1] for i in range(n_levels - 1)] + [g[-1]]


def collapse(lap_stack):
    return np.sum(lap_stack, axis=0)


def multires_blend(im1, im2, mask, n_levels=N_LEVELS, sigma0=SIGMA0):
    """Burt & Adelson blending with stacks: at every level, combine the two
    Laplacian bands using that level of the mask's Gaussian stack.
    mask = 1 -> im1, mask = 0 -> im2. Returns the blend plus the per-level
    masked bands (used for the visualizations)."""
    if im1.ndim == 3 and mask.ndim == 2:
        mask = np.repeat(mask[..., None], im1.shape[2], axis=2)
    l1 = laplacian_stack(im1, n_levels, sigma0)
    l2 = laplacian_stack(im2, n_levels, sigma0)
    gm = gaussian_stack(mask.astype(float), n_levels, sigma0)
    masked1 = [m * a for m, a in zip(gm, l1)]
    masked2 = [(1 - m) * b for m, b in zip(gm, l2)]
    blended = [a + b for a, b in zip(masked1, masked2)]
    return collapse(blended), dict(l1=l1, l2=l2, gm=gm, masked1=masked1,
                                   masked2=masked2, blended=blended)


def normalize(im):
    """Contrast stretch for displaying Laplacian bands (which are ~zero-mean).
    Uses the 0.5 / 99.5 percentiles instead of min / max so a few extreme
    pixels (e.g. stars) don't wash out the rest of the band."""
    lo, hi = np.percentile(im, 0.5), np.percentile(im, 99.5)
    return (im - lo) / (hi - lo + 1e-12)


def show(a, im, title=None, fs=11):
    im = np.clip(im, 0, 1)
    a.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
    if title:
        a.set_title(title, fontsize=fs)
    a.axis("off")


def save(out_dir, name, im):
    im = np.clip(im, 0, 1)
    plt.imsave(os.path.join(out_dir, name), im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)


def stacks_figure(im, name, path, n_levels=N_LEVELS, sigma0=SIGMA0):
    g = gaussian_stack(im, n_levels, sigma0)
    l = laplacian_stack(im, n_levels, sigma0)
    fig, axes = plt.subplots(2, n_levels, figsize=(3.2 * n_levels, 7.8))
    for i in range(n_levels):
        eff = np.sqrt(sum((sigma0 * 2 ** k) ** 2 for k in range(i)))
        show(axes[0, i], g[i], f"Gaussian level {i}\n(effective σ ≈ {eff:.1f})")
        last = i == n_levels - 1
        show(axes[1, i], l[i] if last else normalize(l[i]),
             f"Laplacian level {i}" + (" (= G_last)" if last else "\n(normalized)"))
    fig.suptitle(f"Part 2.3: Gaussian and Laplacian stacks of {name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return g, l


def figure_342(parts, title, path, levels=(0, 2, 4), labels=("left image", "right image")):
    """Recreates Szeliski Fig 3.42 / Burt & Adelson Fig 10: rows = Laplacian
    levels (masked), last row = collapsed; columns = masked image 1,
    masked image 2, blend."""
    rows = [(f"Level {i}", parts["masked1"][i], parts["masked2"][i], parts["blended"][i], True)
            for i in levels]
    rows.append(("Collapsed", collapse(parts["masked1"]), collapse(parts["masked2"]),
                 collapse(parts["blended"]), False))
    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.6 * len(rows)))
    for r, (rname, a, b, c, is_band) in enumerate(rows):
        for j, (im, col) in enumerate(zip((a, b, c), (*labels, "blend"))):
            show(axes[r, j], normalize(im) if is_band else im, f"{rname}: {col}")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    apple = imread(os.path.join(HERE, "apple.jpeg"))
    orange = imread(os.path.join(HERE, "orange.jpeg"))

    for im, name in [(apple, "apple"), (orange, "orange")]:
        g, l = stacks_figure(im, f"{name}.jpeg", os.path.join(OUT, f"{name}_stacks.png"))
        for i in range(N_LEVELS):
            save(OUT, f"{name}_gaussian_level{i}.png", g[i])
            save(OUT, f"{name}_laplacian_level{i}.png",
                 l[i] if i == N_LEVELS - 1 else normalize(l[i]))
        err = np.abs(collapse(l) - im).max()
        print(f"{name}: max |collapse(laplacian stack) - image| = {err:.2e}")

    # vertical seam: apple on the left, orange on the right
    h, w = apple.shape[:2]
    mask = np.zeros((h, w))
    mask[:, : w // 2] = 1.0
    blend, parts = multires_blend(apple, orange, mask)

    for i in range(N_LEVELS):
        band = i < N_LEVELS - 1
        f = normalize if band else (lambda x: x)
        save(OUT, f"oraple_level{i}_masked_apple.png", f(parts["masked1"][i]))
        save(OUT, f"oraple_level{i}_masked_orange.png", f(parts["masked2"][i]))
        save(OUT, f"oraple_level{i}_blend.png", f(parts["blended"][i]))
        save(OUT, f"mask_gaussian_level{i}.png", parts["gm"][i])
    save(OUT, "oraple_collapsed_masked_apple.png", collapse(parts["masked1"]))
    save(OUT, "oraple_collapsed_masked_orange.png", collapse(parts["masked2"]))
    save(OUT, "oraple.png", blend)

    figure_342(parts, "Part 2.3: recreating Szeliski Fig. 3.42 (Laplacian levels 0, 2, 4 + collapsed)",
               os.path.join(OUT, "figure_3_42_recreation.png"), labels=("apple", "orange"))

    # all levels of the mask's Gaussian stack
    fig, axes = plt.subplots(1, N_LEVELS, figsize=(3 * N_LEVELS, 3.5))
    for i, a in enumerate(axes):
        show(a, parts["gm"][i][..., 0], f"mask level {i}")
    fig.suptitle("Gaussian stack of the step-function mask", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(OUT, "mask_stack.png"), dpi=90)
    plt.close(fig)

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"Stacks: {N_LEVELS} levels, sigma_i = {SIGMA0} * 2^i applied between levels\n"
                f"Laplacian level {N_LEVELS - 1} is the final Gaussian level (low-pass residual)\n"
                "Laplacian bands are contrast-stretched (0.5-99.5 percentile) for display\n")


if __name__ == "__main__":
    main()
