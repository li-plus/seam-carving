import timeit

import numpy as np
from PIL import Image

import seam_carving


def perf(energy_mode, delta, n=5):
    src = np.asarray(Image.open("../fig/castle.jpg"))
    h, w, _ = src.shape

    f = lambda: seam_carving.resize(src, (w + delta, h), energy_mode=energy_mode)
    cost = timeit.timeit(f, setup=f, number=n) / n
    print(f"energy: {energy_mode}, delta: {delta:+}px, cost: {cost:.4f}s")


def main():
    perf("backward", -200)
    perf("backward", 200)
    perf("forward", -200)
    perf("forward", 200)


if __name__ == "__main__":
    main()
