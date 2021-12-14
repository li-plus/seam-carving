import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seam_carving import carve


class TestRealImage:
    ROOT = Path(__file__).resolve().parent
    FIG_DIR = ROOT / "fig"
    FIG_BEACH = FIG_DIR / "a-strap-black-dress.jpg"
    FIG_BEACH_GIRL = FIG_DIR / "beach_girl.png"
    FIG_BEACH_BIRD = FIG_DIR / "beach_bird.png"
    def setUp(self) -> None:
        self.img = np.array(Image.open(str(self.FIG_BEACH)))
        self.mask_girl = np.array(Image.open(str(self.FIG_BEACH_GIRL)).convert("L"))
        self.mask_bird = np.array(Image.open(str(self.FIG_BEACH_BIRD)).convert("L"))
    def test_remove_object(self):
        carve.remove_object(self.img, self.mask_girl)
        carve.remove_object(self.img, self.mask_bird, self.mask_girl)
        h, w, _ = self.img.shape
        with pytest.raises(ValueError):
            carve.remove_object(np.empty((1, 2, 3, 4)), self.mask_girl)
        with pytest.raises(ValueError):
            carve.remove_object(self.img, np.zeros((h, w - 1)))
        with pytest.raises(ValueError):
            carve.remove_object(self.img, self.mask_bird, np.zeros((h + 1, w)))
    def test_protect_object(self):
        h, w, _ = self.img.shape
        keep_mask = self.mask_girl | self.mask_bird
        # carve.resize(self.img, (w + 200, h), "backward", "width-first", keep_mask)
        print(f"--- Former version")
        s1 = time.time()
        carve.resize(self.img, (w + 200, h), "forward", "width-first", None)
        print(f"Took {time.time() - s1} seconds")
tri = TestRealImage()
tri.setUp()
tri.test_protect_object()
