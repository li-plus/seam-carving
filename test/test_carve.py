import unittest
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import seam_carving
from seam_carving import carve


class RefCarve:
    @staticmethod
    def _remove_seam(src, seam):
        if src.ndim == 2:
            h, w = src.shape
            dst = np.zeros((h, w - 1), dtype=src.dtype)
        else:
            h, w, c = src.shape
            dst = np.zeros((h, w - 1, c), dtype=src.dtype)
        for r in range(h):
            c = seam[r]
            dst[r, :c] = src[r, :c]
            dst[r, c:] = src[r, c + 1 :]
        return dst

    @staticmethod
    def _get_backward_seam(energy):
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
        seam = np.empty(h, dtype=np.int32)

        for r in range(h - 1, -1, -1):
            seam[r] = c
            c = parent[r, c]

        return seam

    @staticmethod
    def _get_forward_seam(gray, aux_energy):
        gray = gray.astype(np.float32)
        h, w = gray.shape
        dp = np.zeros((h, w), dtype=np.float32)
        parent = np.zeros((h, w), dtype=np.int32)

        if aux_energy is None:
            aux_energy = np.zeros((h, w), dtype=np.float32)

        for c in range(w):
            left = max(c - 1, 0)
            right = min(c + 1, w - 1)
            dp[0, c] = np.abs(gray[0, left] - gray[0, right])

        for r in range(1, h):
            for c in range(w):
                left = max(c - 1, 0)
                right = min(c + 1, w - 1)
                cost_up = np.abs(gray[r, left] - gray[r, right]) + aux_energy[r, c]
                cost_left = cost_up + np.abs(gray[r - 1, c] - gray[r, left])
                cost_right = cost_up + np.abs(gray[r - 1, c] - gray[r, right])

                dp_up = dp[r - 1, c]
                dp_left = dp[r - 1, c - 1] if c > 0 else np.inf
                dp_right = dp[r - 1, c + 1] if c < w - 1 else np.inf

                choices = [cost_left + dp_left, cost_up + dp_up, cost_right + dp_right]

                dp[r, c] = np.min(choices)
                parent[r, c] = np.argmin(choices) + c - 1

        c = np.argmin(dp[-1])
        seam = np.empty(h, dtype=np.int32)
        for r in range(h - 1, -1, -1):
            seam[r] = c
            c = parent[r, c]

        return seam

    @classmethod
    def _reduce_width(cls, src, delta_width, energy_mode, aux_energy):
        if aux_energy is None:
            aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        for _ in range(delta_width):
            gray = src if src.ndim == 2 else carve._rgb2gray(src)
            if energy_mode == "backward":
                energy = carve._get_energy(gray) + aux_energy
                seam = cls._get_backward_seam(energy)
            else:
                seam = cls._get_forward_seam(gray, aux_energy)
            src = cls._remove_seam(src, seam)
            aux_energy = cls._remove_seam(aux_energy, seam)
        return src, aux_energy

    @staticmethod
    def _insert_seam(src, seam):
        if src.ndim == 2:
            h, w = src.shape
            dst_shape = (h, w + 1)
        else:
            h, w, c = src.shape
            dst_shape = (h, w + 1, c)

        dst = np.empty(dst_shape, dtype=src.dtype)

        for r, c in enumerate(seam):
            dst[r, :c] = src[r, :c]
            dst[r, c] = src[r, max(c - 1, 0) : c + 1].mean(axis=0)
            dst[r, c + 1 :] = src[r, c:]

        return dst

    @classmethod
    def _expand_width(cls, src, delta_width, energy_mode, aux_energy, step_ratio):
        assert step_ratio == 0.5

        gray = src if src.ndim == 2 else carve._rgb2gray(src)
        h, w = gray.shape
        seams = carve._get_seams(gray, delta_width, energy_mode, aux_energy)
        _, insert_cols = np.nonzero(seams)
        insert_cols = insert_cols.reshape((h, delta_width))

        for seam_idx in range(delta_width):
            seam = insert_cols[:, seam_idx]
            src = cls._insert_seam(src, seam)
            if aux_energy is not None:
                aux_energy = cls._insert_seam(aux_energy, seam)
            insert_cols += 1

        return src, aux_energy


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
        self.rgb = np.random.randint(0, 256, (self.h, self.w, 3), dtype=np.uint8)
        self.keep_mask = np.random.randint(0, 2, (self.h, self.w)).astype(bool)
        self.drop_mask = np.random.random((self.h, self.w)) < 0.05
        self.keep_aux_energy = np.zeros((self.h, self.w), dtype=np.float32)
        self.keep_aux_energy[self.keep_mask] += carve.KEEP_MASK_ENERGY
        self.drop_aux_energy = np.zeros((self.h, self.w), dtype=np.float32)
        self.drop_aux_energy[self.drop_mask] -= carve.DROP_MASK_ENERGY

    def test_reduce_width(self):
        cases = [
            dict(
                src=src,
                delta_width=delta_width,
                energy_mode=energy_mode,
                aux_energy=aux_energy,
            )
            for src in (self.gray, self.rgb)
            for delta_width in (1, self.delta)
            for energy_mode in ("backward", "forward")
            for aux_energy in (None, self.drop_aux_energy, self.keep_aux_energy)
        ]
        for kwargs in cases:
            out, out_aux = carve._reduce_width(**kwargs)
            ref, ref_aux = RefCarve._reduce_width(**kwargs)
            assert (out == ref).all()
            if out_aux is not None:
                assert np.allclose(out_aux, ref_aux)

    def test_expand_width(self):
        cases = [
            dict(
                src=src,
                delta_width=delta_width,
                energy_mode=energy_mode,
                aux_energy=aux_energy,
                step_ratio=0.5,
            )
            for src in (self.gray, self.rgb)
            for delta_width in (1, self.delta)
            for energy_mode in ("backward", "forward")
            for aux_energy in (None, self.drop_aux_energy, self.keep_aux_energy)
        ]
        for kwargs in cases:
            out, out_aux = carve._expand_width(**kwargs)
            ref, ref_aux = RefCarve._expand_width(**kwargs)
            assert (out == ref).all()
            if out_aux is not None:
                assert np.allclose(out_aux, ref_aux)

        # multi-step expansion
        cases = [
            dict(
                src=src,
                delta_width=2 * src.shape[1],
                energy_mode=energy_mode,
                aux_energy=aux_energy,
                step_ratio=0.5,
            )
            for src in (self.gray, self.rgb)
            for energy_mode in ("backward", "forward")
            for aux_energy in (None, self.drop_aux_energy, self.keep_aux_energy)
        ]
        for kwargs in cases:
            out, _ = carve._expand_width(**kwargs)

            delta = kwargs.pop("delta_width")
            ref = kwargs.pop("src")
            aux_energy = kwargs.pop("aux_energy")
            step1 = round(ref.shape[1] * 0.5)
            ref, aux_energy = RefCarve._expand_width(
                ref, step1, aux_energy=aux_energy, **kwargs
            )
            step2 = round(ref.shape[1] * 0.5)
            ref, aux_energy = RefCarve._expand_width(
                ref, step2, aux_energy=aux_energy, **kwargs
            )
            step3 = delta - step1 - step2
            ref, aux_energy = RefCarve._expand_width(
                ref, step3, aux_energy=aux_energy, **kwargs
            )
            assert (out == ref).all()

    def test_resize(self):
        cases = (
            [
                dict(src=src, size=(w, h), energy_mode=energy_mode, order=order)
                for src in (self.gray, self.rgb)
                for w in (self.lo_w, self.w, self.hi_w)
                for h in (self.lo_h, self.h, self.hi_h)
                for energy_mode in ("backward", "forward")
                for order in ("width-first", "height-first")
            ]
            + [
                dict(src=src, size=(w, h), energy_mode=energy_mode, dtype=dtype)
                for src in (self.gray, self.rgb)
                for w in (self.lo_w, self.w, self.hi_w)
                for h in (self.lo_h, self.h, self.hi_h)
                for energy_mode in ("backward", "forward")
                for dtype in (np.uint8, np.int32, np.int64, np.float32, np.float64)
            ]
            + [
                dict(src=src, size=(w, h), keep_mask=self.keep_mask)
                for src in (self.gray, self.rgb)
                for w in (self.lo_w, self.w, self.hi_w)
                for h in (self.lo_h, self.h, self.hi_h)
            ]
        )
        for kwargs in cases:
            src = kwargs.pop("src")
            dtype = kwargs.pop("dtype", src.dtype)
            src = src.astype(dtype)
            dst = seam_carving.resize(src, **kwargs)
            w, h = kwargs["size"]
            assert dst.shape[:2] == (h, w)
            assert dst.ndim == src.ndim
            assert dst.dtype == src.dtype

        cases = [
            dict(src=src, order=order, drop_mask=self.drop_mask)
            for src in (self.gray, self.rgb)
            for order in ("width-first", "height-first")
        ]
        for kwargs in cases:
            src = kwargs.pop("src")
            src_h, src_w = src.shape[:2]
            dst = seam_carving.resize(src, **kwargs)
            dst_h, dst_w = dst.shape[:2]
            if kwargs["order"] == "width-first":
                assert dst_h == src_h
                assert dst_w <= src_w
            else:
                assert dst_h <= src_h
                assert dst_w == src_w
            assert dst.ndim == src.ndim
            assert dst.dtype == src.dtype

        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (0, 100))
        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (100, 0))
        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (self.w + 1, self.h), step_ratio=0)
        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (self.w + 1, self.h), step_ratio=1.1)
        with pytest.raises(ValueError):
            seam_carving.resize(np.zeros(10), (100, 100))
        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (100, 100), energy_mode="oops")
        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (100, 100), order="oops")
        with pytest.raises(ValueError):
            seam_carving.resize(self.rgb, (100, 100), keep_mask=np.zeros((10, 10)))
        with pytest.raises(ValueError):
            seam_carving.resize(
                self.rgb, (100, 100), keep_mask=np.zeros((self.h, self.w, 3))
            )


class TestRealImage(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent
    FIG_DIR = ROOT / "fig"
    FIG_BEACH = FIG_DIR / "beach.jpg"
    FIG_BEACH_GIRL = FIG_DIR / "beach_girl.png"
    FIG_BEACH_BIRD = FIG_DIR / "beach_bird.png"

    def setUp(self) -> None:
        self.img = np.asarray(Image.open(self.FIG_BEACH))
        self.mask_girl = np.asarray(Image.open(self.FIG_BEACH_GIRL).convert("L"))
        self.mask_bird = np.asarray(Image.open(self.FIG_BEACH_BIRD).convert("L"))

    def test_remove_object(self):
        # remove object
        out = seam_carving.resize(self.img, drop_mask=self.mask_girl)
        with pytest.warns(DeprecationWarning):
            ref = seam_carving.remove_object(self.img, self.mask_girl)
        assert (out == ref).all()

        # drop & keep
        out = seam_carving.resize(
            self.img, keep_mask=self.mask_girl, drop_mask=self.mask_bird
        )
        with pytest.warns(DeprecationWarning):
            ref = seam_carving.remove_object(self.img, self.mask_bird, self.mask_girl)
        assert (out == ref).all()

        # remove object & resize back
        out = seam_carving.resize(
            self.img, self.img.shape[:2], drop_mask=self.mask_girl
        )
        with pytest.warns(DeprecationWarning):
            ref = seam_carving.remove_object(self.img, self.mask_girl)
            ref = seam_carving.resize(ref, self.img.shape[:2])
        assert (out == ref).all()

        h, w, _ = self.img.shape
        with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
            seam_carving.remove_object(np.empty((1, 2, 3, 4)), self.mask_girl)
        with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
            seam_carving.remove_object(self.img, np.zeros((h, w - 1)))
        with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
            seam_carving.remove_object(self.img, self.mask_bird, np.zeros((h + 1, w)))

    def test_protect_object(self):
        h, w, _ = self.img.shape
        keep_mask = self.mask_girl | self.mask_bird
        for energy_mode in ("backward", "forward"):
            seam_carving.resize(
                self.img,
                (w - 200, h),
                energy_mode=energy_mode,
                order="width-first",
                keep_mask=keep_mask,
            )
