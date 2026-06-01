"""
test_cubemap.py — CubeMapConverter 단위 테스트.

실행:
    pytest tests/test_cubemap.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def converter():
    from pipeline.cubemap import CubeMapConverter

    return CubeMapConverter(face_size=256, device="cpu")


@pytest.fixture
def dummy_erp() -> np.ndarray:
    """2:1 종횡비 더미 ERP 이미지 (512x256 RGB)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)


class TestErpToCubemap:
    def test_returns_six_faces(self, converter, dummy_erp):
        faces = converter.erp_to_cubemap(dummy_erp)
        assert set(faces.keys()) == {"front", "right", "back", "left", "top", "bottom"}

    def test_face_shape(self, converter, dummy_erp):
        faces = converter.erp_to_cubemap(dummy_erp)
        for name, face in faces.items():
            assert face.shape == (256, 256, 3), f"{name} face 크기 오류: {face.shape}"

    def test_face_dtype(self, converter, dummy_erp):
        faces = converter.erp_to_cubemap(dummy_erp)
        for name, face in faces.items():
            assert face.dtype == np.uint8, f"{name} face dtype 오류: {face.dtype}"


class TestCubemapToErp:
    def test_roundtrip_shape(self, converter, dummy_erp):
        faces = converter.erp_to_cubemap(dummy_erp)
        reconstructed = converter.cubemap_to_erp(faces, erp_height=256, erp_width=512)
        assert reconstructed.shape == dummy_erp.shape

    def test_roundtrip_dtype(self, converter, dummy_erp):
        faces = converter.erp_to_cubemap(dummy_erp)
        reconstructed = converter.cubemap_to_erp(faces, erp_height=256, erp_width=512)
        assert reconstructed.dtype == np.uint8

    def test_roundtrip_content_similar(self, converter, dummy_erp):
        """라운드트립 후 PSNR이 20 dB 이상이어야 한다."""
        faces = converter.erp_to_cubemap(dummy_erp)
        reconstructed = converter.cubemap_to_erp(faces, erp_height=256, erp_width=512)

        mse = np.mean((dummy_erp.astype(float) - reconstructed.astype(float)) ** 2)
        if mse == 0:
            return  # perfect match
        psnr = 10 * np.log10(255**2 / mse)
        assert psnr >= 20.0, f"PSNR 너무 낮음: {psnr:.2f} dB"
