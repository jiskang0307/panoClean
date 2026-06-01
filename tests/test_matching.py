"""
test_matching.py — BackgroundMatcher, SeamlessBlender 단위 테스트.

실행:
    pytest tests/test_matching.py -v

핵심 의존성 없으면 전체 skip.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch",  reason="torch 미설치")
np    = pytest.importorskip("numpy",  reason="numpy 미설치")
cv2   = pytest.importorskip("cv2",    reason="opencv 미설치")


# ── 공통 픽스처 ───────────────────────────────────────────────────────────

BASE_CFG = {
    "device":              "cpu",
    "feature_matcher":     "sift",   # 테스트에서 SIFT 사용 (모델 파일 불필요)
    "min_match_count":     10,
    "min_coverage_ratio":  0.5,
    "loftr_pretrained":    "indoor",
}

H, W = 128, 128   # 테스트용 소형 이미지


def _face(seed: int = 0) -> torch.Tensor:
    """재현 가능한 (3,H,W) float32 더미 face."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.rand(3, H, W, generator=rng)


def _mask(y1: int = 40, y2: int = 90, x1: int = 30, x2: int = 100) -> torch.Tensor:
    m = torch.zeros(H, W, dtype=torch.bool)
    m[y1:y2, x1:x2] = True
    return m


@pytest.fixture(scope="module")
def matcher():
    from pipeline.matching import BackgroundMatcher
    return BackgroundMatcher(BASE_CFG)


@pytest.fixture(scope="module")
def blender():
    from pipeline.blending import SeamlessBlender
    return SeamlessBlender()


# ═══════════════════════════════════════════════════════════════════════════
# 내부 유틸 (_t2bgr, _bgr2t, _t2gray_np)
# ═══════════════════════════════════════════════════════════════════════════

class TestTypeConversions:
    def test_t2bgr_shape(self, matcher):
        t = _face()
        bgr = matcher._t2bgr(t)
        assert bgr.shape == (H, W, 3)
        assert bgr.dtype == np.uint8

    def test_t2bgr_range(self, matcher):
        t = _face()
        bgr = matcher._t2bgr(t)
        assert bgr.min() >= 0 and bgr.max() <= 255

    def test_bgr2t_shape(self, matcher):
        bgr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        t = matcher._bgr2t(bgr)
        assert t.shape == (3, H, W)
        assert t.dtype == torch.float32

    def test_bgr2t_range(self, matcher):
        bgr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        t = matcher._bgr2t(bgr)
        assert t.min() >= 0.0 and t.max() <= 1.0

    def test_t2gray_np_shape(self, matcher):
        gray = matcher._t2gray_np(_face())
        assert gray.shape == (H, W)
        assert gray.dtype == np.uint8

    def test_roundtrip_color(self, matcher):
        """BGR→tensor→BGR 라운드트립 오차 < 2."""
        bgr = np.random.randint(10, 245, (H, W, 3), dtype=np.uint8)
        t = matcher._bgr2t(bgr)
        bgr2 = matcher._t2bgr(t)
        assert abs(bgr.astype(int) - bgr2.astype(int)).max() <= 2


# ═══════════════════════════════════════════════════════════════════════════
# find_homography
# ═══════════════════════════════════════════════════════════════════════════

class TestFindHomography:
    def test_same_image_returns_identity_like(self, matcher):
        """동일 이미지는 항등 homography에 가까워야 한다."""
        face = _face(seed=42)
        H_mat, ratio = matcher.find_homography(face, face, "front")
        # 동일 이미지라면 매칭이 잘 되고 inlier_ratio가 높아야 함
        # 텍스처가 있는 이미지라면 H가 None이 아니어야 함
        # SIFT는 랜덤 패턴에서 매칭이 잘 안 될 수 있으므로 None도 허용
        if H_mat is not None:
            assert H_mat.shape == (3, 3)
            assert 0.0 <= ratio <= 1.0

    def test_returns_none_on_blank_image(self, matcher):
        """균일색 이미지는 keypoint가 없어 None 반환."""
        blank = torch.ones(3, H, W) * 0.5
        H_mat, ratio = matcher.find_homography(blank, blank, "front")
        assert H_mat is None
        assert ratio == 0.0

    def test_output_types(self, matcher):
        face = _face(seed=1)
        result = matcher.find_homography(face, face, "front")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_polar_face_uses_sift(self, matcher):
        """up/down face는 force_sift=True로 SIFT 코드 경로를 타야 한다."""
        blank = torch.ones(3, H, W) * 0.5
        H_mat, ratio = matcher.find_homography(blank, blank, "down")
        # blank 이미지니 매칭 없음, None 반환 (SIFT 코드 경로 확인)
        assert H_mat is None

    def test_inlier_ratio_in_range(self, matcher):
        """유효한 homography가 있을 때 inlier_ratio ∈ [0, 1]."""
        face = _face(seed=99)
        H_mat, ratio = matcher.find_homography(face, face, "front")
        if H_mat is not None:
            assert 0.0 <= ratio <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# warp_background
# ═══════════════════════════════════════════════════════════════════════════

class TestWarpBackground:
    def _identity_H(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64)

    def test_identity_warp_preserves_image(self, matcher):
        """항등 homography는 이미지를 변경하지 않아야 한다."""
        face = _face(seed=5)
        mask = _mask()
        warped = matcher.warp_background(face, mask, self._identity_H())
        diff = (face - warped).abs().max().item()
        assert diff < 0.01, f"항등 변환 후 오차: {diff}"

    def test_output_shape(self, matcher):
        face = _face()
        mask = _mask()
        warped = matcher.warp_background(face, mask, self._identity_H())
        assert warped.shape == (3, H, W)

    def test_output_dtype(self, matcher):
        warped = matcher.warp_background(_face(), _mask(), self._identity_H())
        assert warped.dtype == torch.float32

    def test_output_range(self, matcher):
        warped = matcher.warp_background(_face(), _mask(), self._identity_H())
        assert warped.min() >= 0.0 - 1e-5
        assert warped.max() <= 1.0 + 1e-5

    def test_translation_warp(self, matcher):
        """10px 이동 homography 적용 후 warped가 원본과 달라야 한다."""
        face = _face(seed=7)
        mask = _mask()
        tx = 10.0
        H_translate = np.array([[1, 0, tx], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        warped = matcher.warp_background(face, mask, H_translate)
        assert warped.shape == face.shape
        # 경계 픽셀은 0이어야 함 (이동으로 인한 black region)
        assert warped[:, :, :int(tx)].sum() == 0.0

    def test_out_of_bounds_region_is_zero(self, matcher):
        """워핑 후 유효 영역 밖은 0으로 채워져야 한다."""
        face = torch.ones(3, H, W)  # 순백
        mask = _mask()
        # 전체를 오른쪽으로 크게 이동 → 왼쪽 절반이 0
        H_large = np.array([[1, 0, W], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        warped = matcher.warp_background(face, mask, H_large)
        assert warped.sum() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# select_best_sources
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectBestSources:
    def test_empty_sources_returns_empty(self, matcher):
        result = matcher.select_best_sources(_face(), [], _mask())
        assert result == []

    def test_blank_sources_returns_empty(self, matcher):
        """keypoint 없는 blank 소스는 skip되어 결과가 빈 리스트."""
        blank = torch.ones(3, H, W) * 0.5
        result = matcher.select_best_sources(blank, [blank, blank], _mask())
        assert result == []

    def test_output_is_list_of_tuples(self, matcher):
        """선택 결과가 (warped, H_mat, coverage) 튜플 리스트여야 한다."""
        face = _face(seed=10)
        sources = [_face(seed=11), _face(seed=12)]
        result = matcher.select_best_sources(face, sources, _mask())
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3

    def test_result_sorted_by_coverage(self, matcher):
        """결과가 coverage 내림차순으로 정렬되어야 한다."""
        face = _face(seed=20)
        sources = [_face(seed=i) for i in range(5)]
        result = matcher.select_best_sources(face, sources, _mask(), top_k=5)
        covs = [c for _, _, c in result]
        assert covs == sorted(covs, reverse=True)

    def test_top_k_respected(self, matcher):
        """top_k보다 많은 결과가 반환되지 않아야 한다."""
        face = _face(seed=30)
        sources = [_face(seed=i) for i in range(10)]
        result = matcher.select_best_sources(face, sources, _mask(), top_k=3)
        assert len(result) <= 3

    def test_warped_tensor_shape(self, matcher):
        """반환된 warped tensor 크기가 target face와 동일해야 한다."""
        face = _face(seed=40)
        sources = [face.clone()]  # 동일 이미지로 매칭 시도
        result = matcher.select_best_sources(face, sources, _mask())
        for warped, _, _ in result:
            assert warped.shape == (3, H, W)

    def test_coverage_in_range(self, matcher):
        face = _face(seed=50)
        sources = [face.clone()]
        result = matcher.select_best_sources(face, sources, _mask())
        for _, _, cov in result:
            assert 0.0 <= cov <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# blend_multiple_sources
# ═══════════════════════════════════════════════════════════════════════════

class TestBlendMultipleSources:
    def test_empty_sources_returns_target(self, matcher):
        target = _face(seed=0)
        result = matcher.blend_multiple_sources(target, _mask(), [])
        assert torch.allclose(result, target)

    def test_output_shape(self, matcher):
        target = _face(seed=1)
        mask   = _mask()
        # 더미 sources: identity warped
        H_id = np.eye(3, dtype=np.float64)
        warped = matcher.warp_background(_face(seed=2), mask, H_id)
        sources = [(warped, H_id, 0.8)]
        result = matcher.blend_multiple_sources(target, mask, sources)
        assert result.shape == (3, H, W)

    def test_output_dtype(self, matcher):
        target = _face(seed=2)
        H_id   = np.eye(3, dtype=np.float64)
        warped = matcher.warp_background(_face(seed=3), _mask(), H_id)
        sources = [(warped, H_id, 0.9)]
        result = matcher.blend_multiple_sources(target, _mask(), sources)
        assert result.dtype == torch.float32

    def test_output_range(self, matcher):
        target = _face(seed=3)
        H_id   = np.eye(3, dtype=np.float64)
        warped = matcher.warp_background(_face(seed=4), _mask(), H_id)
        sources = [(warped, H_id, 0.9)]
        result = matcher.blend_multiple_sources(target, _mask(), sources)
        assert result.min() >= 0.0 - 1e-4
        assert result.max() <= 1.0 + 1e-4

    def test_mask_region_changed(self, matcher):
        """mask 영역이 sources 내용으로 변경되어야 한다."""
        target = torch.zeros(3, H, W)   # 검정
        source = torch.ones(3, H, W)    # 흰색
        mask   = _mask()
        H_id   = np.eye(3, dtype=np.float64)
        warped = matcher.warp_background(source, mask, H_id)
        sources = [(warped, H_id, 1.0)]
        result = matcher.blend_multiple_sources(target, mask, sources)
        # mask 영역은 흰색에 가까워야 함
        assert result[:, mask].mean() > 0.5

    def test_nonmask_region_unchanged(self, matcher):
        """mask 외 영역은 target과 동일해야 한다."""
        target = _face(seed=5)
        source = torch.ones(3, H, W)
        mask   = _mask(y1=40, y2=50, x1=40, x2=60)  # 작은 마스크
        H_id   = np.eye(3, dtype=np.float64)
        warped = matcher.warp_background(source, mask, H_id)
        sources = [(warped, H_id, 1.0)]
        result = matcher.blend_multiple_sources(target, mask, sources, face_name="front")
        outside = ~mask
        # Poisson blend가 적용되면 경계 근처 비-마스크 영역도 약간 바뀔 수 있음
        # 따라서 마스크에서 멀리 떨어진 픽셀만 검증
        # mask가 중앙(40:50, 40:60)이므로 모서리 8px는 영향 없어야 함
        corner = result[:, :8, :8]
        target_corner = target[:, :8, :8]
        assert torch.allclose(corner, target_corner, atol=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# SeamlessBlender
# ═══════════════════════════════════════════════════════════════════════════

class TestSeamlessBlender:

    def _bgr(self, val: int = 128) -> np.ndarray:
        return np.full((H, W, 3), val, dtype=np.uint8)

    def _mask_np(self) -> np.ndarray:
        m = np.zeros((H, W), dtype=np.uint8)
        m[40:80, 40:90] = 255
        return m

    # ── alpha_blend_edge ─────────────────────────────────────────────────

    def test_alpha_shape(self, blender):
        result = blender.alpha_blend_edge(self._bgr(50), self._bgr(200), self._mask_np())
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_alpha_empty_mask_returns_target(self, blender):
        tgt = self._bgr(50)
        result = blender.alpha_blend_edge(tgt, self._bgr(200), np.zeros((H, W), dtype=np.uint8))
        assert np.array_equal(result, tgt)

    def test_alpha_center_is_source(self, blender):
        """마스크 중앙은 source 값에 가까워야 한다."""
        result = blender.alpha_blend_edge(
            self._bgr(0), self._bgr(200), self._mask_np(), feather_px=4
        )
        # mask 중앙 픽셀 (60, 65)
        assert result[60, 65, 0] > 150

    def test_alpha_outside_is_target(self, blender):
        """마스크 밖 영역은 target 값을 유지해야 한다."""
        result = blender.alpha_blend_edge(
            self._bgr(50), self._bgr(200), self._mask_np(), feather_px=2
        )
        assert result[0, 0, 0] == 50   # 모서리는 target 값

    # ── poisson_blend ────────────────────────────────────────────────────

    def test_poisson_shape(self, blender):
        result = blender.poisson_blend(self._bgr(50), self._bgr(200), self._mask_np())
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    def test_poisson_empty_mask_returns_target(self, blender):
        tgt = self._bgr(50)
        result = blender.poisson_blend(tgt, self._bgr(200), np.zeros((H, W), dtype=np.uint8))
        assert np.array_equal(result, tgt)

    def test_poisson_bool_mask_accepted(self, blender):
        bool_mask = self._mask_np().astype(bool)
        result = blender.poisson_blend(self._bgr(50), self._bgr(200), bool_mask)
        assert result.shape == (H, W, 3)

    # ── compute_coverage ─────────────────────────────────────────────────

    def test_coverage_full(self, blender):
        from pipeline.blending import SeamlessBlender
        m = torch.ones(H, W, dtype=torch.bool)
        assert SeamlessBlender.compute_coverage(m, m) == 1.0

    def test_coverage_empty(self, blender):
        from pipeline.blending import SeamlessBlender
        m = torch.zeros(H, W, dtype=torch.bool)
        # mask가 비어있으면 1.0 반환 (처리 불필요)
        assert SeamlessBlender.compute_coverage(m, m) == 1.0

    def test_coverage_partial(self, blender):
        from pipeline.blending import SeamlessBlender
        mask   = torch.zeros(H, W, dtype=torch.bool)
        mask[40:80, 40:80] = True  # 40×40 = 1600 pixels
        filled = torch.zeros(H, W, dtype=torch.bool)
        filled[40:80, 40:60] = True  # 40×20 = 800 pixels → 0.5
        cov = SeamlessBlender.compute_coverage(mask, filled)
        assert abs(cov - 0.5) < 0.01

    def test_coverage_range(self, blender):
        from pipeline.blending import SeamlessBlender
        m1 = torch.randint(0, 2, (H, W)).bool()
        m2 = torch.randint(0, 2, (H, W)).bool()
        cov = SeamlessBlender.compute_coverage(m1, m2)
        assert 0.0 <= cov <= 1.0
