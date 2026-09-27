# Part 2.4 Bells & Whistles: multiresolution blending in color.
#
# Same pipeline as 2.4.py, but each RGB channel gets its own Laplacian stack
# (the mask's Gaussian stack is shared across channels), so the blend keeps
# the colors of both images instead of working on luminance only.

import os
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_bells_and_whistles")

_spec = importlib.util.spec_from_file_location("part2_4", os.path.join(HERE, "2.4.py"))
p24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p24)


if __name__ == "__main__":
    p24.run(color=True, out=OUT, tag="color")
