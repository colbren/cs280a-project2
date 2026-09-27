# Part 2.2 Bells & Whistles: using color to enhance hybrid images.
#
# For each pair, build four versions of the hybrid:
#   1. gray high + gray low         (the baseline from 2.2.py)
#   2. color high + gray low
#   3. gray high + color low
#   4. color high + color low
# Filtering is done per RGB channel; a "gray" component is the grayscale
# image replicated to three channels.

import os
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_bells_and_whistles")

# reuse everything from 2.2.py (file name isn't importable directly)
_spec = importlib.util.spec_from_file_location("part2_2", os.path.join(HERE, "2.2.py"))
p22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p22)


def gray3(im):
    return np.repeat(p22.to_gray(im)[..., None], 3, axis=2)


def main():
    os.makedirs(OUT, exist_ok=True)

    for pair in p22.PAIRS:
        name = pair["name"]
        a_high, a_low = p22.load_aligned(pair)
        s_hi, s_lo = pair["sigma_high"], pair["sigma_low"]

        hp_color = p22.high_pass(a_high, s_hi)
        hp_gray = p22.high_pass(gray3(a_high), s_hi)
        lp_color = p22.low_pass(a_low, s_lo)
        lp_gray = p22.low_pass(gray3(a_low), s_lo)

        variants = [
            ("gray_high_gray_low", "Gray high + gray low", hp_gray + lp_gray),
            ("color_high_gray_low", "Color high + gray low", hp_color + lp_gray),
            ("gray_high_color_low", "Gray high + color low", hp_gray + lp_color),
            ("color_high_color_low", "Color high + color low", hp_color + lp_color),
        ]

        p22.save(OUT, f"{name}_input_high_color.png", a_high)
        p22.save(OUT, f"{name}_input_low_color.png", a_low)
        p22.save(OUT, f"{name}_high_pass_color.png", p22.show_high(hp_color))
        p22.save(OUT, f"{name}_low_pass_color.png", lp_color)
        for key, _, h in variants:
            p22.save(OUT, f"{name}_{key}.png", h)

        # comparison grid: full size (close up) on top, 1/6 size (far away) below
        fig, axes = plt.subplots(2, 4, figsize=(20, 9.5),
                                 gridspec_kw={"height_ratios": [1, 0.45]})
        for j, (_, title, h) in enumerate(variants):
            h = np.clip(h, 0, 1)
            axes[0, j].imshow(h)
            axes[0, j].set_title(f"{title}\n(close up)", fontsize=12)
            small = p22.downscale(h, 6)
            axes[1, j].imshow(small)
            axes[1, j].set_title("far away (1/6 size)", fontsize=10)
        for a in axes.flat:
            a.axis("off")
        fig.suptitle(f"Part 2.2 Bells & Whistles: color in the hybrid ({pair['desc']};"
                     f" σ_high={s_hi}, σ_low={s_lo})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(OUT, f"{name}_color_comparison.png"), dpi=90)
        plt.close(fig)

        # inputs + best color result, like the base figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for a, (p, t) in zip(axes, [(a_high, f"High-frequency input\n{pair['high']}"),
                                    (a_low, f"Low-frequency input\n{pair['low']}"),
                                    (np.clip(hp_color + lp_color, 0, 1), "Hybrid (color high + color low)")]):
            a.imshow(p)
            a.set_title(t, fontsize=12)
            a.axis("off")
        fig.suptitle(f"Part 2.2 Bells & Whistles: {pair['desc']}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(OUT, f"{name}_color_figure.png"), dpi=100)
        plt.close(fig)

        p22.distance_figure(np.clip(hp_color + lp_color, 0, 1),
                            f"{pair['desc']}, color high + color low (shrink = step back)",
                            os.path.join(OUT, f"{name}_color_distance.png"))

    print(f"Saved color hybrid comparisons to {OUT}")


if __name__ == "__main__":
    main()
