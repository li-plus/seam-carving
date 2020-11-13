import math
import unittest
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seam_carving import carve

warnings.filterwarnings('error')


class TestCarve(unittest.TestCase):
    def setUp(self):
        self.h = np.random.randint(100, 200)
        self.w = np.random.randint(100, 200)
        self.delta = np.random.randint(10, 30)
        self.hi_h = self.h + self.delta
        self.lo_h = self.h - self.delta
        self.hi_w = self.w + self.delta
        self.lo_w = self.w - self.delta
        self.gray = np.random.randint(0, 256, (self.h, self.w), dtype=np.uint8)
        self.rgb = np.random.randint(0, 256, (self.h, self.w, 3),
                                     dtype=np.uint8)

    @staticmethod
    def _naive_remove_seam(src, seam):
        if src.ndim == 2:
            h, w = src.shape
            dst = np.zeros((h, w - 1), dtype=np.uint8)
        else:
            h, w, c = src.shape
            dst = np.zeros((h, w - 1, c), dtype=np.uint8)
        for r in range(h):
            c = seam[r]
            dst[r, :c] = src[r, :c]
            dst[r, c:] = src[r, c + 1:]
        return dst

    def test_remove_seam(self):
        for src in (self.gray, self.rgb):
            for _ in range(self.delta):
                gray = src if src.ndim == 2 else carve._rgb2gray(src)
                energy = carve._get_energy(gray)
                seam, _ = carve._get_backward_seam(energy)
                dst = carve._remove_seam(src, seam)
                ans_dst = self._naive_remove_seam(src, seam)
                assert (dst == ans_dst).all()
                src = dst

    @staticmethod
    def _naive_get_backward_seam(energy):
        h, w = energy.shape
        cost = np.zeros((h, w), dtype=np.float32)
        parent = np.zeros((h, w), dtype=np.int32)
        cost[0] = energy[0]
        for r in range(1, h):
            for c in range(w):
                min_col = c
                if c > 0 and cost[r - 1, c - 1] <= cost[r - 1, min_col]:
                    min_col = c - 1
                if c < w - 1 and cost[r - 1, c + 1] < cost[r - 1, min_col]:
                    min_col = c + 1
                cost[r, c] = cost[r - 1, min_col] + energy[r, c]
                parent[r, c] = min_col

        c = np.argmin(cost[-1])
        total_cost = cost[-1, c]
        seam = np.empty(h, dtype=np.int32)

        for r in range(h - 1, -1, -1):
            seam[r] = c
            c = parent[r, c]
        return seam, total_cost

    def test_get_backward_seam(self):
        energy = carve._get_energy(self.gray)
        seam, cost = carve._get_backward_seam(energy)
        ans_seam, ans_cost = self._naive_get_backward_seam(energy)
        assert (seam == ans_seam).all()
        assert math.isclose(cost, ans_cost)

    @staticmethod
    def _naive_get_forward_seam(gray):
        gray = gray.astype(np.float32)
        h, w = gray.shape
        dp = np.zeros((h, w), dtype=np.float32)
        parent = np.zeros((h, w), dtype=np.int32)

        for c in range(w):
            left = max(c - 1, 0)
            right = min(c + 1, w - 1)
            dp[0, c] = np.abs(gray[0, left] - gray[0, right])

        for r in range(1, h):
            for c in range(w):
                left = max(c - 1, 0)
                right = min(c + 1, w - 1)
                cost_up = np.abs(gray[r, left] - gray[r, right])
                cost_left = cost_up + np.abs(gray[r - 1, c] - gray[r, left])
                cost_right = cost_up + np.abs(gray[r - 1, c] - gray[r, right])

                dp_up = dp[r - 1, c]
                dp_left = dp[r - 1, c - 1] if c > 0 else np.inf
                dp_right = dp[r - 1, c + 1] if c < w - 1 else np.inf

                choices = [cost_left + dp_left,
                           cost_up + dp_up,
                           cost_right + dp_right]

                dp[r, c] = np.min(choices)
                parent[r, c] = np.argmin(choices) + c - 1

        c = np.argmin(dp[-1])
        total_cost = dp[-1, c]

        seam = np.empty(h, dtype=np.int32)
        for r in range(h - 1, -1, -1):
            seam[r] = c
            c = parent[r, c]

        return seam, total_cost

    def test_get_forward_seam(self):
        out_seam, out_cost = carve._get_forward_seam(self.gray, None)
        ans_seam, ans_cost = self._naive_get_forward_seam(self.gray)
        assert (out_seam == ans_seam).all()
        assert math.isclose(out_cost, ans_cost)

    @staticmethod
    def _naive_reduce_width(src, num_seams, energy_mode):
        for _ in range(num_seams):
            gray = src if src.ndim == 2 else carve._rgb2gray(src)
            if energy_mode == 'backward':
                seam, _ = carve._get_backward_seam(carve._get_energy(gray))
            else:
                seam, _ = carve._get_forward_seam(gray, None)
            src = carve._remove_seam(src, seam)
        return src

    def test_reduce_width(self):
        for src in (self.gray, self.rgb):
            out = carve._reduce_width(src, self.delta, 'backward', None)
            ans = self._naive_reduce_width(src, self.delta, 'backward')
            assert (out == ans).all()

            out = carve._reduce_width(src, self.delta, 'forward', None)
            ans = self._naive_reduce_width(src, self.delta, 'forward')
            assert (out == ans).all()

    @staticmethod
    def _naive_insert_seam(src, seam):
        if src.ndim == 2:
            h, w = src.shape
            dst_shape = (h, w + 1)
        else:
            h, w, c = src.shape
            dst_shape = (h, w + 1, c)

        dst = np.empty(dst_shape, dtype=np.uint8)

        for r, c in enumerate(seam):
            dst[r, :c] = src[r, :c]
            dst[r, c] = src[r, max(c - 1, 0):c + 1].mean(axis=0)
            dst[r, c + 1:] = src[r, c:]

        return dst

    @staticmethod
    def _naive_expand_width(src, delta_width, energy_mode):
        gray = src if src.ndim == 2 else carve._rgb2gray(src)
        h, w = gray.shape
        seams_mask = carve._get_seams(gray, delta_width, energy_mode, None)
        _, insert_cols = np.nonzero(seams_mask)
        insert_cols = insert_cols.reshape((h, delta_width))

        for seam_idx in range(delta_width):
            seam = insert_cols[:, seam_idx]
            src = TestCarve._naive_insert_seam(src, seam)
            insert_cols += 1

        return src

    def test_expand_width(self):
        out = carve._expand_width(self.gray, self.delta, 'backward', None)
        ans = self._naive_expand_width(self.gray, self.delta, 'backward')
        assert (out == ans).all()

    def test_resize(self):
        for src in (self.gray, self.rgb):
            dst = carve.resize(src, (self.w, self.h), 'backward', 'width-first')
            assert (dst == src).all()
            dst = carve.resize(src, (self.w, self.h), 'forward', 'height-first')
            assert (dst == src).all()

            dst = carve.resize(
                src, (self.lo_w, self.lo_h), 'backward', 'width-first')
            assert dst.shape[:2] == (self.lo_h, self.lo_w)
            dst = carve.resize(
                src, (self.lo_w, self.hi_h), 'forward', 'width-first')
            assert dst.shape[:2] == (self.hi_h, self.lo_w)
            dst = carve.resize(
                src, (self.hi_w, self.lo_h), 'forward', 'height-first')
            assert dst.shape[:2] == (self.lo_h, self.hi_w)
            dst = carve.resize(
                src, (self.hi_w, self.hi_h), 'backward', 'height-first')
            assert dst.shape[:2] == (self.hi_h, self.hi_w)

        with pytest.raises(ValueError):
            carve.resize(self.rgb, (0, 100))
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (100, 0))
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (self.w * 2, self.h))
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (self.w, self.h * 2))
        with pytest.raises(ValueError):
            carve.resize(np.zeros(10), (100, 100))
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (100, 100), 'oops')
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (100, 100), 'forward', 'oops')
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (100, 100), keep_mask=np.zeros((10, 10)))
        with pytest.raises(ValueError):
            carve.resize(self.rgb, (100, 100),
                         keep_mask=np.zeros((self.h, self.w, 3)))


class TestRealImage(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent
    FIG_DIR = ROOT / 'fig'
    FIG_BEACH = FIG_DIR / 'beach.jpg'
    FIG_BEACH_GIRL = FIG_DIR / 'beach_girl.png'
    FIG_BEACH_BIRD = FIG_DIR / 'beach_bird.png'

    def setUp(self) -> None:
        self.img = np.array(Image.open(str(self.FIG_BEACH)))
        self.mask_girl = np.array(
            Image.open(str(self.FIG_BEACH_GIRL)).convert('L'))
        self.mask_bird = np.array(
            Image.open(str(self.FIG_BEACH_BIRD)).convert('L'))

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
        carve.resize(self.img, (w - 200, h), 'backward', 'width-first',
                     keep_mask)
        carve.resize(self.img, (w - 200, h), 'forward', 'width-first',
                     keep_mask)
