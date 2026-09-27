# Part 2.1: Image "Sharpening" with the unsharp mask filter.
#
#   blurred   = f * G
#   details   = f - f * G                        (high frequencies)
#   sharpened = f + alpha * (f - f * G)
#             = f * ((1 + alpha) * e - alpha * G)   <- single convolution
# where e is the unit impulse. The kernel in parentheses is the unsharp mask filter.

import os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

ALPHAS = [0.5, 1, 2, 4]

# image -> (sigma, alpha used for the headline result)
IMAGES = {
    "taj.jpg": (2.0, 2),
    "colosseum.jpg": (2.0, 1),
    "space_needle.jpg": (2.0, 1),
}

# Evaluation: take a sharp image, blur it, then try to sharpen it back.
EVAL_IMAGE = "space_needle.jpg"
EVAL_BLUR_SIGMA = 2.0
EVAL_SHARPEN_SIGMA = 2.0
EVAL_ALPHAS = [1, 2, 4]


def imread(path):
    im = plt.imread(path).astype(float)
    if im.max() > 1.0:
        im /= 255.0
    return im[..., :3] if im.ndim == 3 else im


def gaussian_kernel_2d(sigma):
    ksize = int(6 * sigma + 1) | 1  # odd, covers +-3 sigma
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T


def filter_color(im, k):
    """Convolve each color channel with k."""
    if im.ndim == 2:
        return convolve2d(im, k, mode="same", boundary="symm")
    return np.dstack([convolve2d(im[..., c], k, mode="same", boundary="symm")
                      for c in range(im.shape[2])])


def unsharp_mask_kernel(sigma, alpha):
    G = gaussian_kernel_2d(sigma)
    impulse = np.zeros_like(G)
    impulse[G.shape[0] // 2, G.shape[1] // 2] = 1.0
    return (1 + alpha) * impulse - alpha * G


def sharpen(im, sigma, alpha):
    return np.clip(filter_color(im, unsharp_mask_kernel(sigma, alpha)), 0, 1)


def show_details(d):
    """High-frequency image is signed; center it on gray and boost for visibility."""
    m = np.percentile(np.abs(d), 99.5) + 1e-12
    return np.clip(0.5 + 0.5 * d / m, 0, 1)


def save(name, im):
    plt.imsave(os.path.join(OUT, name), np.clip(im, 0, 1))


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    for fname, (sigma, alpha_main) in IMAGES.items():
        name = os.path.splitext(fname)[0]
        im = imread(os.path.join(HERE, fname))
        G = gaussian_kernel_2d(sigma)
        blurred = filter_color(im, G)
        details = im - blurred

        # two-step result vs single unsharp-mask convolution
        two_step = np.clip(im + alpha_main * details, 0, 1)
        one_step = sharpen(im, sigma, alpha_main)
        lines.append(f"{fname}: sigma={sigma}, alpha={alpha_main}, "
                     f"max |two-step - single convolution| = {np.abs(two_step - one_step).max():.2e}")

        save(f"{name}_original.png", im)
        save(f"{name}_blurred.png", blurred)
        save(f"{name}_high_freq.png", show_details(details))
        save(f"{name}_sharpened_alpha{alpha_main}.png", one_step)

        # process figure: original, low-pass, high-pass, sharpened
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
        for a, (p, t) in zip(axes, [(im, "Original"),
                                    (blurred, f"Low frequencies: f * G  (σ={sigma})"),
                                    (show_details(details), "High frequencies: f − f * G\n(gray = 0, contrast boosted)"),
                                    (one_step, f"Sharpened: f + α(f − f * G),  α={alpha_main}")]):
            a.imshow(p)
            a.set_title(t, fontsize=12)
            a.axis("off")
        fig.suptitle(f"Part 2.1: unsharp masking on {fname}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(OUT, f"{name}_process_figure.png"), dpi=100)
        plt.close(fig)

        # alpha sweep
        fig, axes = plt.subplots(1, len(ALPHAS) + 1, figsize=(4.4 * (len(ALPHAS) + 1), 5.4))
        axes[0].imshow(im)
        axes[0].set_title("Original (α = 0)", fontsize=12)
        for a, alpha in zip(axes[1:], ALPHAS):
            s = sharpen(im, sigma, alpha)
            save(f"{name}_sharpened_alpha{alpha}.png", s)
            a.imshow(s)
            a.set_title(f"α = {alpha}", fontsize=12)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"Part 2.1: sharpening strength on {fname}  (σ = {sigma})", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(OUT, f"{name}_alpha_sweep.png"), dpi=100)
        plt.close(fig)

    # the unsharp mask filter itself
    k = unsharp_mask_kernel(2.0, 2)
    fig, a = plt.subplots(figsize=(5.5, 4.5))
    m = np.abs(k).max()
    img = a.imshow(k, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    fig.colorbar(img, ax=a)
    a.set_title(f"Unsharp mask filter (1+α)e − αG\nσ=2, α=2 (center tap = {k[k.shape[0] // 2, k.shape[1] // 2]:.2f}, "
                "color clipped)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "unsharp_mask_filter.png"), dpi=100)
    plt.close(fig)

    # --- evaluation: blur a sharp image, then sharpen it ---
    name = os.path.splitext(EVAL_IMAGE)[0]
    orig = imread(os.path.join(HERE, EVAL_IMAGE))
    blurred = filter_color(orig, gaussian_kernel_2d(EVAL_BLUR_SIGMA))
    resharpened = {a: sharpen(blurred, EVAL_SHARPEN_SIGMA, a) for a in EVAL_ALPHAS}

    save(f"eval_{name}_original.png", orig)
    save(f"eval_{name}_blurred.png", blurred)
    lines += ["", f"Evaluation on {EVAL_IMAGE}: blur sigma={EVAL_BLUR_SIGMA}, "
                  f"sharpen sigma={EVAL_SHARPEN_SIGMA}",
              f"  PSNR(blurred, original)       = {psnr(blurred, orig):.2f} dB"]
    for a, r in resharpened.items():
        save(f"eval_{name}_resharpened_alpha{a}.png", r)
        lines.append(f"  PSNR(resharpened α={a}, original) = {psnr(r, orig):.2f} dB")

    # zoomed crops make the difference easier to see
    h, w = orig.shape[:2]
    crop = (slice(int(0.25 * h), int(0.55 * h)), slice(int(0.25 * w), int(0.75 * w)))
    cols = [(orig, "Original (sharp)"), (blurred, f"Blurred (σ={EVAL_BLUR_SIGMA})")] + \
           [(r, f"Re-sharpened α={a}") for a, r in resharpened.items()]
    fig, axes = plt.subplots(2, len(cols), figsize=(4.4 * len(cols), 8.6))
    for j, (p, t) in enumerate(cols):
        axes[0, j].imshow(p)
        axes[0, j].set_title(t, fontsize=12)
        axes[1, j].imshow(p[crop])
        axes[1, j].set_title("zoomed crop", fontsize=10)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"Part 2.1 evaluation: blur a sharp image, then sharpen it again ({EVAL_IMAGE})", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"eval_{name}_figure.png"), dpi=100)
    plt.close(fig)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
