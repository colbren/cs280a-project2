# Part 1.1: Convolutions from Scratch
#
# Convolution implemented with numpy only (four loops, then two loops),
# with zero padding ("same" or "full"), compared against
# scipy.signal.convolve2d. Then a picture of me is convolved with a 9x9
# box filter and the finite difference operators Dx and Dy.

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d  # only used for the comparison

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

# Downsample factor for the portrait. The original is 1737x3088, and a
# pure-python four-loop convolution over 5 million pixels would take
# hours, so we work on a 1/6-size version.
DOWNSAMPLE = 6


# ---------------------------------------------------------------------------
# Convolution from scratch
# ---------------------------------------------------------------------------

def zero_pad(im, kh, kw, mode="same"):
    """Pad `im` with zeros so that sliding a kh x kw kernel over it gives
    an output of the requested size.
      same: output is the same size as `im`
      full: output is (H + kh - 1) x (W + kw - 1)
    For even-sized kernels "same" puts the extra row/col of padding on the
    top/left, which matches scipy.signal.convolve2d."""
    if mode == "same":
        top, left = kh // 2, kw // 2
        bottom, right = kh - 1 - top, kw - 1 - left
    elif mode == "full":
        top = bottom = kh - 1
        left = right = kw - 1
    else:
        raise ValueError(f"unknown mode {mode!r}")
    H, W = im.shape
    padded = np.zeros((H + top + bottom, W + left + right), dtype=float)
    padded[top:top + H, left:left + W] = im
    return padded


def conv_four_loops(im, kernel, mode="same"):
    kernel = kernel[::-1, ::-1]  # flip the filter -> convolution, not correlation
    kh, kw = kernel.shape
    padded = zero_pad(im, kh, kw, mode)
    out_h, out_w = padded.shape[0] - kh + 1, padded.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += padded[i + u, j + v] * kernel[u, v]
            out[i, j] = total
    return out


def conv_two_loops(im, kernel, mode="same"):
    kernel = kernel[::-1, ::-1]
    kh, kw = kernel.shape
    padded = zero_pad(im, kh, kw, mode)
    out_h, out_w = padded.shape[0] - kh + 1, padded.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gray(path):
    im = plt.imread(path).astype(float)
    if im.max() > 1.0:
        im /= 255.0
    rgb = im[..., :3]
    return rgb @ np.array([0.299, 0.587, 0.114])


def downsample(im, f):
    """Block-average downsample by an integer factor (numpy only)."""
    H, W = (im.shape[0] // f) * f, (im.shape[1] // f) * f
    return im[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def to_display(im, signed=False):
    """Map a filter response to [0, 1]. Signed responses (derivatives)
    are centered on gray so negative = dark, positive = bright. We scale
    by the 99.5th percentile rather than the max so the strong artificial
    edge created by zero padding at the border doesn't wash everything out."""
    if signed:
        m = np.percentile(np.abs(im), 99.5) + 1e-12
        return np.clip(0.5 + 0.5 * im / m, 0, 1)
    return np.clip(im, 0, 1)


def save(name, im, signed=False):
    plt.imsave(os.path.join(OUT, name), to_display(im, signed), cmap="gray", vmin=0, vmax=1)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)

    box = np.ones((9, 9)) / 81.0
    Dx = np.array([[1, -1]], dtype=float)
    Dy = np.array([[1], [-1]], dtype=float)
    filters = {"box": box, "Dx": Dx, "Dy": Dy}

    # --- correctness check on a small random image, both padding modes ---
    rng = np.random.default_rng(0)
    test = rng.random((25, 31))
    lines = ["Correctness check vs scipy.signal.convolve2d (random 25x31 image)",
             "max |ours - scipy|:"]
    for mode in ("same", "full"):
        for fname, k in filters.items():
            ref = convolve2d(test, k, mode=mode, boundary="fill", fillvalue=0)
            d4 = np.abs(conv_four_loops(test, k, mode) - ref).max()
            d2 = np.abs(conv_two_loops(test, k, mode) - ref).max()
            lines.append(f"  mode={mode:4s} filter={fname:3s}  four-loop: {d4:.2e}  two-loop: {d2:.2e}")

    # --- portrait ---
    portrait = downsample(load_gray(os.path.join(HERE, "portrait.png")), DOWNSAMPLE)
    save("portrait_gray.png", portrait)
    lines += ["", f"Portrait (grayscale, downsampled {DOWNSAMPLE}x) size: "
                  f"{portrait.shape[1]}x{portrait.shape[0]}", "",
              "Timings on portrait, mode='same' (seconds) and max |diff| vs scipy:",
              f"  {'filter':6s} {'four-loop':>10s} {'two-loop':>10s} {'scipy':>10s} {'max diff':>10s}"]

    results = {}
    for fname, k in filters.items():
        r4, t4 = timed(conv_four_loops, portrait, k, "same")
        r2, t2 = timed(conv_two_loops, portrait, k, "same")
        rs, ts = timed(convolve2d, portrait, k, mode="same", boundary="fill", fillvalue=0)
        diff = max(np.abs(r4 - rs).max(), np.abs(r2 - rs).max())
        lines.append(f"  {fname:6s} {t4:10.4f} {t2:10.4f} {ts:10.4f} {diff:10.2e}")
        results[fname] = (r4, r2, rs, (t4, t2, ts))
        signed = fname != "box"
        save(f"portrait_{fname}_four_loops.png", r4, signed)
        save(f"portrait_{fname}_two_loops.png", r2, signed)
        save(f"portrait_{fname}_scipy.png", rs, signed)

    # "full" padding example so the size difference is visible
    full_box = conv_two_loops(portrait, box, "full")
    save("portrait_box_full_padding.png", full_box)
    lines.append("")
    lines.append(f"'full' padding with 9x9 box: output {full_box.shape[1]}x{full_box.shape[0]} "
                 f"(input {portrait.shape[1]}x{portrait.shape[0]}; +8 in each dimension)")

    # --- labeled comparison figure ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    method_names = ["Four loops", "Two loops", "scipy convolve2d"]
    for row, (fname, (r4, r2, rs, times)) in enumerate(results.items()):
        signed = fname != "box"
        axes[row, 0].imshow(portrait, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Original (grayscale)")
        for col, (r, t, m) in enumerate(zip((r4, r2, rs), times, method_names), start=1):
            axes[row, col].imshow(to_display(r, signed), cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{fname}: {m}\n{t:.4f} s")
    for a in axes.flat:
        a.axis("off")
    fig.suptitle("Part 1.1: convolution from scratch vs scipy (zero 'same' padding)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "comparison_figure.png"), dpi=100)
    plt.close(fig)

    # --- filters as images ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for a, (fname, k) in zip(axes, filters.items()):
        a.imshow(k, cmap="gray")
        a.set_title(f"{fname} {k.shape[0]}x{k.shape[1]}")
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "filters.png"), dpi=100)
    plt.close(fig)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
