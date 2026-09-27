# Part 2.4: Multiresolution Blending (Burt & Adelson 1983), grayscale.
#
# Uses the Gaussian / Laplacian stacks from part 2.3. For every level i:
#     blended_i = GM_i * LA_i + (1 - GM_i) * LB_i
# where LA, LB are the Laplacian stacks of the two images and GM is the
# Gaussian stack of the mask. Collapsing (summing) the blended stack gives
# the result. Blurring the mask more at coarser levels makes low frequencies
# transition over a wide region and high frequencies over a narrow one.
#
# The color version lives in 2.4_bells_and_whistles.py (run(color=True)).

import os
import importlib.util

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

# stack code from part 2.3 (file name isn't importable directly)
_spec = importlib.util.spec_from_file_location("part2_3", os.path.join(HERE, "..", "2.3", "2.3.py"))
p23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p23)

FAVORITE = "clock_baseball"


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def imread(path):
    return p23.imread(os.path.join(HERE, path))


def to_gray(im):
    return im @ np.array([0.299, 0.587, 0.114])


def resize(im, w, h):
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


def crop_fruit(im, sat_thresh, margin=0.03):
    """Crop to the (saturated) fruit, ignoring the white background, gray
    shadow, and a dark border line in the lime image. The lemon's shadow is
    slightly yellow, so it needs a higher saturation threshold than the
    (darker, less saturated) lime."""
    sat = im.max(axis=2) - im.min(axis=2)
    fruit = ndi.binary_opening(sat > sat_thresh, iterations=3)
    lab, n = ndi.label(fruit)
    fruit = lab == (np.argmax(ndi.sum(fruit, lab, range(1, n + 1))) + 1)
    rows, cols = np.where(fruit.any(axis=1))[0], np.where(fruit.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    mr, mc = int(margin * (r1 - r0)), int(margin * (c1 - c0))
    H, W = im.shape[:2]
    return im[max(r0 - mr, 0):min(r1 + mr, H), max(c0 - mc, 0):min(c1 + mc, W)]


def fit_ball_ellipse(ball, cx=227, cy=181, grow=5):
    """Fit an ellipse to the ball's silhouette. The red stitching runs all
    the way around the ball's outline, so we take the small red blobs
    (stitch dashes; the big red blobs are fingernails) 75-100 px from the
    ball's approximate center (cx, cy) and fit an ellipse to them. Stitches
    sit just inside the silhouette, so the ellipse is grown by `grow` px.
    Returns ((center x, center y), (axis 1, axis 2), angle) as in OpenCV."""
    R, G, B = ball[..., 0], ball[..., 1], ball[..., 2]
    stitch = (R - G > 0.3) & (R - B > 0.25)
    lab, n = ndi.label(stitch)
    sizes = ndi.sum(stitch, lab, range(1, n + 1))
    dashes = np.isin(lab, np.where(sizes <= 25)[0] + 1)
    H, W = stitch.shape
    yy, xx = np.mgrid[:H, :W]
    rad = np.hypot(xx - cx, yy - cy)
    pts = np.argwhere(dashes & (rad > 75) & (rad < 100))[:, ::-1].astype(np.float32)
    center, (a1, a2), angle = cv2.fitEllipse(pts)
    return center, (a1 + 2 * grow, a2 + 2 * grow), angle


def baseball_mask(ball, ellipse):
    """Visible part of the ball: the fitted ellipse minus the two fingertips
    that wrap over its top edge, so they stay in front of the clock. The
    ball's rim picks up a pinkish reflection from the hand, so skin is only
    searched for in the top part of the ball (where the fingertips are);
    elsewhere the smooth ellipse edge is kept. Skin is red-orange with
    G > B, which separates it from the red stitches (where G ~ B)."""
    H, W = ball.shape[:2]
    e = np.zeros((H, W), np.uint8)
    cv2.ellipse(e, ellipse, 1, -1)
    (ex, ey), (a1, a2), _ = ellipse
    yy = np.mgrid[:H, :W][0]
    R, G, B = ball[..., 0], ball[..., 1], ball[..., 2]
    skin = (R - B > 0.2) & (G - B > 0.05) & (yy < ey - max(a1, a2) / 4)
    skin = ndi.binary_opening(skin, iterations=1)
    skin = ndi.binary_fill_holes(ndi.binary_closing(skin, iterations=3))
    lab, n = ndi.label(skin)
    sizes = ndi.sum(skin, lab, range(1, n + 1))
    fingers = np.isin(lab, np.where(sizes >= 100)[0] + 1)
    return (e.astype(bool) & ~fingers).astype(float)


def clock_onto_ball(clock, ball_shape, ellipse, ccx=600, ccy=600, cr=415):
    """Affine-warp the clock so its circular face (center (ccx, ccy), radius
    cr) lands exactly on the ball's ellipse, which covers the whole mask.
    Everything outside the face (the dark wooden frame, the white
    background) is first painted with the face's own cream color: the
    coarse Laplacian levels blend past the mask edge, and a dark frame
    there would bleed a dark halo around the ball."""
    Y, X = np.mgrid[:clock.shape[0], :clock.shape[1]]
    inside = (X - ccx) ** 2 + (Y - ccy) ** 2 <= cr ** 2
    face = np.median(clock[inside & (clock.mean(axis=2) > 0.8)], axis=0)
    clock = clock.copy()
    clock[~inside] = face
    # map the face's center and two perpendicular radii onto the ellipse's
    # center and semi-axes
    (ex, ey), (a1, a2), angle = ellipse
    t = np.deg2rad(angle)
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-np.sin(t), np.cos(t)])
    src = np.float32([[ccx, ccy], [ccx + cr, ccy], [ccx, ccy + cr]])
    dst = np.float32([[ex, ey], [ex, ey] + u * a1 / 2, [ex, ey] + v * a2 / 2])
    M = cv2.getAffineTransform(src, dst)
    H, W = ball_shape[:2]
    return cv2.warpAffine(clock, M, (W, H), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in face))


def prepare_pairs():
    """Returns a list of blend specs with RGB images in [0, 1].
    mask = 1 selects im1, mask = 0 selects im2."""
    pairs = []

    apple, orange = imread("apple.jpeg"), imread("orange.jpeg")
    h, w = apple.shape[:2]
    mask = np.zeros((h, w))
    mask[:, : w // 2] = 1
    pairs.append(dict(name="oraple", im1=apple, im2=orange, mask=mask,
                      labels=("apple", "orange"), levels=6, sigma0=2.0,
                      desc="The oraple: apple (left) + orange (right), vertical seam"))

    lemon = resize(crop_fruit(imread("lemon_blend.png"), 0.5), 500, 400)
    lime = resize(crop_fruit(imread("lime_blend.png"), 0.2), 500, 400)
    mask = np.zeros((400, 500))
    mask[:200] = 1
    pairs.append(dict(name="lemon_lime", im1=lemon, im2=lime, mask=mask,
                      labels=("lemon", "lime"), levels=6, sigma0=2.0,
                      desc="Lemon (top) + lime (bottom), horizontal seam"))

    ball = imread("baseball_blend.png")
    ellipse = fit_ball_ellipse(ball)
    clock = clock_onto_ball(imread("clock_blend.jpg"), ball.shape, ellipse)
    pairs.append(dict(name="clock_baseball", im1=clock, im2=ball, mask=baseball_mask(ball, ellipse),
                      labels=("clock", "baseball"), levels=5, sigma0=1.0,
                      desc="Clock blended into a baseball (irregular mask)"))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(color=False, out=OUT, tag="grayscale"):
    os.makedirs(out, exist_ok=True)
    lines = []

    for pair in prepare_pairs():
        name = pair["name"]
        im1, im2, mask = pair["im1"], pair["im2"], pair["mask"]
        if not color:
            im1, im2 = to_gray(im1), to_gray(im2)
        m = mask if im1.ndim == 2 else mask[..., None]

        blend, parts = p23.multires_blend(im1, im2, mask, pair["levels"], pair["sigma0"])
        naive = m * im1 + (1 - m) * im2
        blend = np.clip(blend, 0, 1)

        p23.save(out, f"{name}_input1.png", im1)
        p23.save(out, f"{name}_input2.png", im2)
        p23.save(out, f"{name}_mask.png", mask)
        p23.save(out, f"{name}_masked_input1.png", m * im1)
        p23.save(out, f"{name}_masked_input2.png", (1 - m) * im2)
        p23.save(out, f"{name}_naive_paste.png", naive)
        p23.save(out, f"{name}_blend.png", blend)
        lines.append(f"{name}: {pair['desc']}  (levels={pair['levels']}, sigma0={pair['sigma0']}, "
                     f"size {blend.shape[1]}x{blend.shape[0]})")

        # inputs, mask, naive paste, blend
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        for a, (p, t) in zip(axes, [(im1, pair["labels"][0]), (im2, pair["labels"][1]),
                                    (mask, "mask (1 = first image)"),
                                    (naive, "naive cut & paste"),
                                    (blend, f"multiresolution blend\n({pair['levels']} levels)")]):
            p23.show(a, p, t, fs=12)
        fig.suptitle(f"Part 2.4 ({tag}): {pair['desc']}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(out, f"{name}_figure.png"), dpi=90)
        plt.close(fig)

        # Laplacian stack illustration (Burt & Adelson Fig. 10 / Szeliski Fig. 3.42)
        n = pair["levels"]
        levels = sorted({0, n // 3, (2 * n) // 3, n - 2})
        p23.figure_342(parts, f"Part 2.4 ({tag}): Laplacian stack of {name}\n"
                              f"masked {pair['labels'][0]}, masked {pair['labels'][1]}, blend",
                       os.path.join(out, f"{name}_laplacian_figure.png"),
                       levels=levels, labels=pair["labels"])

        # every level of the blended stack + mask stack
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.8))
        for i in range(n):
            band = parts["blended"][i]
            p23.show(axes[0, i], band if i == n - 1 else p23.normalize(band),
                     f"blended level {i}" + ("" if i == n - 1 else " (norm.)"), fs=10)
            gm = parts["gm"][i]
            p23.show(axes[1, i], gm if gm.ndim == 2 else gm[..., 0], f"mask level {i}", fs=10)
        fig.suptitle(f"Part 2.4 ({tag}): {name} blended Laplacian stack and mask Gaussian stack", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(out, f"{name}_all_levels.png"), dpi=90)
        plt.close(fig)

    lines.append(f"\nFavorite (for the Laplacian stack illustration on the website): {FAVORITE}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    run(color=False, out=OUT, tag="grayscale")
