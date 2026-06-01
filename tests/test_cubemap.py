"""
test_cubemap.py — CubeMapConverter 단위 테스트.

실행:
    pytest tests/test_cubemap.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 미설치")
Image = pytest.importorskip("PIL.Image", reason="Pillow 미설치")


# ── 픽스처 ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def converter_256():
    from pipeline.cubemap import CubeMapConverter
    return CubeMapConverter(face_size=256, device="cpu")


@pytest.fixture(scope="module")
def converter_512():
    from pipeline.cubemap import CubeMapConverter
    return CubeMapConverter(face_size=512, device="cpu")


@pytest.fixture(scope="module")
def converter_1024():
    from pipeline.cubemap import CubeMapConverter
    return CubeMapConverter(face_size=1024, device="cpu")


@pytest.fixture
def erp_np() -> np.ndarray:
    """512x256 uint8 ndarray (2:1 ERP)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)


@pytest.fixture
def erp_tensor(erp_np) -> torch.Tensor:
    """(3, 256, 512) float32 tensor."""
    t = torch.from_numpy(erp_np).float() / 255.0
    return t.permute(2, 0, 1)


@pytest.fixture
def erp_pil(erp_np) -> Image.Image:
    return Image.fromarray(erp_np)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255**2 / mse)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float32 → (H,W,3) uint8."""
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ── erp_to_cubemap ────────────────────────────────────────────────────────

class TestErpToCubemap:
    def test_returns_six_faces(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        assert set(faces.keys()) == {"front", "right", "back", "left", "up", "down"}

    def test_face_shape(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        for name, face in faces.items():
            assert face.shape == (3, 256, 256), f"{name} shape={face.shape}"

    def test_face_dtype_float32(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        for name, face in faces.items():
            assert face.dtype == torch.float32, f"{name} dtype={face.dtype}"

    def test_face_value_range(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        for name, face in faces.items():
            assert face.min() >= 0.0 - 1e-5, f"{name} min={face.min()}"
            assert face.max() <= 1.0 + 1e-5, f"{name} max={face.max()}"

    def test_accepts_numpy(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        assert len(faces) == 6

    def test_accepts_tensor(self, converter_256, erp_tensor):
        faces = converter_256.erp_to_cubemap(erp_tensor)
        assert len(faces) == 6

    def test_accepts_pil(self, converter_256, erp_pil):
        faces = converter_256.erp_to_cubemap(erp_pil)
        assert len(faces) == 6


# ── cubemap_to_erp ────────────────────────────────────────────────────────

class TestCubemapToErp:
    def test_output_shape(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        erp_out = converter_256.cubemap_to_erp(faces, erp_height=256, erp_width=512)
        assert erp_out.shape == (3, 256, 512)

    def test_output_dtype(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        erp_out = converter_256.cubemap_to_erp(faces, erp_height=256, erp_width=512)
        assert erp_out.dtype == torch.float32

    def test_roundtrip_psnr_35db(self, converter_256, erp_np):
        """ERP → CubeMap → ERP 왕복 후 PSNR 35dB 이상."""
        faces = converter_256.erp_to_cubemap(erp_np)
        erp_out = converter_256.cubemap_to_erp(faces, erp_height=256, erp_width=512)
        out_np = _tensor_to_uint8(erp_out)
        psnr = _psnr(erp_np, out_np)
        assert psnr >= 35.0, f"PSNR={psnr:.2f} dB (기준: 35 dB)"

    def test_custom_output_resolution(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        erp_out = converter_256.cubemap_to_erp(faces, erp_height=512, erp_width=1024)
        assert erp_out.shape == (3, 512, 1024)


# ── 다양한 face_size ──────────────────────────────────────────────────────

class TestFaceSizes:
    @pytest.mark.parametrize("fs,conv_fixture", [
        (256,  "converter_256"),
        (512,  "converter_512"),
        (1024, "converter_1024"),
    ])
    def test_face_size(self, request, fs, conv_fixture, erp_np):
        conv = request.getfixturevalue(conv_fixture)
        faces = conv.erp_to_cubemap(erp_np)
        for name, face in faces.items():
            assert face.shape == (3, fs, fs), f"fs={fs}, {name}: {face.shape}"

    @pytest.mark.parametrize("fs,conv_fixture", [
        (256,  "converter_256"),
        (512,  "converter_512"),
    ])
    def test_roundtrip_each_size(self, request, fs, conv_fixture, erp_np):
        conv = request.getfixturevalue(conv_fixture)
        faces = conv.erp_to_cubemap(erp_np)
        # 원본 해상도로 복원
        erp_out = conv.cubemap_to_erp(faces, erp_height=256, erp_width=512)
        out_np = _tensor_to_uint8(erp_out)
        psnr = _psnr(erp_np, out_np)
        assert psnr >= 30.0, f"fs={fs}, PSNR={psnr:.2f} dB"


# ── 배치 처리 ─────────────────────────────────────────────────────────────

class TestBatchProcessing:
    def test_batch_length(self, converter_256, erp_np):
        imgs = [erp_np, erp_np, erp_np]
        results = converter_256.erp_to_cubemap_batch(imgs)
        assert len(results) == 3

    def test_batch_equals_single(self, converter_256, erp_np):
        """배치 결과와 단건 결과가 동일해야 한다."""
        single = converter_256.erp_to_cubemap(erp_np)
        batch = converter_256.erp_to_cubemap_batch([erp_np])[0]
        for name in ["front", "right", "back", "left", "up", "down"]:
            diff = (single[name] - batch[name]).abs().max().item()
            assert diff < 1e-5, f"{name} face 불일치: max_diff={diff}"

    def test_batch_mixed_types(self, converter_256, erp_np, erp_tensor, erp_pil):
        """배치에 numpy/tensor/PIL 혼합 가능."""
        results = converter_256.erp_to_cubemap_batch([erp_np, erp_tensor, erp_pil])
        assert len(results) == 3
        for r in results:
            assert set(r.keys()) == {"front", "right", "back", "left", "up", "down"}


# ── UV 맵 ─────────────────────────────────────────────────────────────────

class TestUVMap:
    @pytest.mark.parametrize("face", ["front", "right", "back", "left", "up", "down"])
    def test_uv_shape(self, converter_256, face):
        uv = converter_256.get_face_uv_map(face, erp_h=256, erp_w=512)
        assert uv.shape == (2, 256, 256), f"{face}: {uv.shape}"

    @pytest.mark.parametrize("face", ["front", "right", "back", "left", "up", "down"])
    def test_uv_range(self, converter_256, face):
        """UV 값이 [0, 1] 범위 내에 있어야 한다."""
        uv = converter_256.get_face_uv_map(face, erp_h=256, erp_w=512)
        assert uv.min() >= 0.0 - 1e-5, f"{face} uv min={uv.min()}"
        assert uv.max() <= 1.0 + 1e-5, f"{face} uv max={uv.max()}"

    def test_uv_dtype(self, converter_256):
        uv = converter_256.get_face_uv_map("front", erp_h=256, erp_w=512)
        assert uv.dtype == torch.float32

    def test_uv_coverage(self, converter_256):
        """6개 face UV가 ERP 전체 영역을 어느 정도 커버해야 한다."""
        u_vals, v_vals = [], []
        for face in ["front", "right", "back", "left", "up", "down"]:
            uv = converter_256.get_face_uv_map(face, erp_h=256, erp_w=512)
            u_vals.append(uv[0].flatten())
            v_vals.append(uv[1].flatten())
        all_u = torch.cat(u_vals)
        all_v = torch.cat(v_vals)
        # u 범위가 0~1 전체를 커버하는지 (여유 0.05)
        assert all_u.min() < 0.05
        assert all_u.max() > 0.95
        assert all_v.min() < 0.05
        assert all_v.max() > 0.95


# ── Seam Blending ─────────────────────────────────────────────────────────

class TestBlendSeams:
    def test_output_keys(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        blended = converter_256.blend_seams(faces)
        assert set(blended.keys()) == set(faces.keys())

    def test_output_shape_unchanged(self, converter_256, erp_np):
        faces = converter_256.erp_to_cubemap(erp_np)
        blended = converter_256.blend_seams(faces)
        for name in faces:
            assert blended[name].shape == faces[name].shape

    def test_does_not_mutate_input(self, converter_256, erp_np):
        """blend_seams는 원본 faces를 변경하지 않아야 한다."""
        faces = converter_256.erp_to_cubemap(erp_np)
        originals = {k: v.clone() for k, v in faces.items()}
        converter_256.blend_seams(faces)
        for name in faces:
            assert torch.allclose(faces[name], originals[name]), f"{name} 변경됨"

    def test_center_pixels_unchanged(self, converter_256, erp_np):
        """중앙 영역은 blend 영향 없이 동일해야 한다."""
        faces = converter_256.erp_to_cubemap(erp_np)
        blended = converter_256.blend_seams(faces)
        w = converter_256.SEAM_WIDTH
        for name in ["front", "back"]:
            center = faces[name][:, w:-w, w:-w]
            center_b = blended[name][:, w:-w, w:-w]
            assert torch.allclose(center, center_b), f"{name} 중앙 변경됨"
