from pathlib import Path

import numpy as np
from PIL import Image

import seam_carving

ROOT = Path(__file__).resolve().parent.parent
PAD_WIDTH = 4


def main():
    # scaling up & down
    src = np.array(Image.open(ROOT / 'fig/castle.jpg'))
    h, w, c = src.shape
    scale_down = seam_carving.resize(src, (w - 200, h))
    scale_up = seam_carving.resize(src, (w + 200, h))
    padding = np.zeros((h, PAD_WIDTH, c), dtype=np.uint8)
    merged = np.hstack((src, padding, scale_down, padding, scale_up))
    Image.fromarray(merged).show()

    # forward energy vs backward energy
    src = np.array(Image.open(ROOT / 'fig/bench.jpg'))
    h, w, c = src.shape
    backward = seam_carving.resize(src, (w - 200, h))
    forward = seam_carving.resize(src, (w - 200, h), energy_mode='forward')
    padding = np.zeros((h, PAD_WIDTH, c), dtype=np.uint8)
    merged = np.hstack((src, padding, backward, padding, forward))
    Image.fromarray(merged).show()

    # object removal
    src = np.array(Image.open(ROOT / 'fig/beach.jpg'))
    h, w, c = src.shape
    mask = np.array(Image.open(ROOT / 'fig/beach_girl.png').convert('L'))
    dst = seam_carving.remove_object(src, mask)
    padding = np.zeros((h, PAD_WIDTH, c), dtype=np.uint8)
    merged = np.hstack((src, padding, dst))
    Image.fromarray(merged).show()


if __name__ == '__main__':
    main()
