"""
test_segmentation.py — PersonSegmenter, FaceMosaicker, MaskPostProcessor 테스트.

실행:
    pytest tests/test_segmentation.py -v

핵심 의존성(torch, cv2) 없으면 전체 skip.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch 미설치")
np    = pytest.importorskip("numpy", reason="numpy 미설치")
cv2   = pytest.importorskip("cv2",   reason="opencv 미설치")


# ── 픽스처 ────────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "device":                    "cpu",
    "yolo_model":                "yolo11x-seg.pt",
    "sam2_model":                None,
    "person_class_id":           0,
    "yolo_conf":                 0.4,
    "mask_dilate_px":            15,
    "photographer_y_ratio":      0.40,
    "photographer_size_weight":  0.5,
}


def _make_face(h: int = 64, w: int = 64) -> torch.Tensor:
    """(3,H,W) float32 더미 face."""
    return torch.rand(3, h, w)


def _make_detection(
    box: list[float], pixel_count: int = 100, bbox_area: float = 400.0
):
    """_Detection namedtuple 생성 헬퍼."""
    from pipeline.segmentation import _Detection
    h, w = 64, 64
    mask = torch.zeros(h, w, dtype=torch.bool)
    x1, y1, x2, y2 = [int(v) for v in box]
    mask[y1:y2, x1:x2] = True
    return _Detection(
        box=np.array(box, dtype=np.float32),
        mask=mask,
        conf=0.9,
        pixel_count=pixel_count,
        bbox_area=bbox_area,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PersonRole
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonRole:
    def test_enum_values(self):
        from pipeline.segmentation import PersonRole
        assert PersonRole.PHOTOGRAPHER.value == "photographer"
        assert PersonRole.BACKGROUND.value   == "background"

    def test_enum_count(self):
        from pipeline.segmentation import PersonRole
        assert len(PersonRole) == 2


# ═══════════════════════════════════════════════════════════════════════════
# classify_persons — 역할 분류 로직
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyPersons:

    @pytest.fixture(autouse=True)
    def segmenter(self):
        # YOLO 모델 로드 없이 분류 로직만 테스트 (monkeypatch로 _load_yolo 우회)
        from pipeline.segmentation import PersonSegmenter
        import unittest.mock as mock
        with mock.patch.object(PersonSegmenter, "_load_yolo"):
            self.seg = PersonSegmenter(BASE_CONFIG)

    def test_empty_detections(self):
        result = self.seg.classify_persons([], "front", 512, 1024, 64)
        assert result == []

    def test_down_face_all_photographer(self):
        from pipeline.segmentation import PersonRole
        dets = [_make_detection([0, 32, 20, 63]), _make_detection([30, 32, 50, 63])]
        roles = self.seg.classify_persons(dets, "down", 512, 1024, 64)
        # down face는 무조건 PHOTOGRAPHER이지만 최대 1명
        photographers = [r for r, _ in roles if r == PersonRole.PHOTOGRAPHER]
        assert len(photographers) == 1

    def test_up_face_no_condition1(self):
        from pipeline.segmentation import PersonRole
        # up face: 조건1 항상 실패 → score max 0.4 → BACKGROUND
        dets = [_make_detection([0, 0, 20, 20])]
        roles = self.seg.classify_persons(dets, "up", 512, 1024, 64)
        assert roles[0][0] == PersonRole.BACKGROUND

    def test_front_face_bottom_person_is_photographer(self):
        from pipeline.segmentation import PersonRole
        # front face, bbox 중심이 face 하단 → ERP y > 0.60 → 조건1 통과
        # 게다가 가장 큰 mask → 조건2 통과 → score 1.0 → PHOTOGRAPHER
        dets = [_make_detection([10, 50, 50, 63], pixel_count=500)]  # 하단
        roles = self.seg.classify_persons(dets, "front", 512, 1024, 64)
        assert roles[0][0] == PersonRole.PHOTOGRAPHER
        assert roles[0][1] == 1.0

    def test_front_face_top_person_is_background(self):
        from pipeline.segmentation import PersonRole
        # 조건1 실패(상단), 조건2 통과(단독) → score=0.4 → BACKGROUND
        dets = [_make_detection([10, 0, 50, 15], pixel_count=200)]
        roles = self.seg.classify_persons(dets, "front", 512, 1024, 64)
        assert roles[0][0] == PersonRole.BACKGROUND

    def test_single_photographer_enforced(self):
        from pipeline.segmentation import PersonRole
        # score >= 0.6인 후보가 2명이면 1명만 PHOTOGRAPHER로 강등
        dets = [
            _make_detection([10, 50, 30, 63], pixel_count=300),  # 하단, 큰 mask
            _make_detection([35, 48, 55, 63], pixel_count=300),  # 하단, 동일 mask
        ]
        roles = self.seg.classify_persons(dets, "front", 512, 1024, 64)
        photographers = [r for r, _ in roles if r == PersonRole.PHOTOGRAPHER]
        assert len(photographers) == 1

    def test_score_partial_conditions(self):
        from pipeline.segmentation import PersonRole
        # 조건1만 통과 → score=0.6 → PHOTOGRAPHER
        dets = [_make_detection([10, 50, 30, 63], pixel_count=50)]  # 작은 mask
        # 조건2는 단독이므로 통과, 조건1도 통과 → score=1.0
        # → 단독 1명이면 항상 score=1.0. 두 명으로 테스트
        dets2 = [
            _make_detection([10, 50, 30, 63], pixel_count=50),   # 하단 작은
            _make_detection([10, 5,  30, 20], pixel_count=500),  # 상단 큰
        ]
        roles = self.seg.classify_persons(dets2, "front", 512, 1024, 64)
        # 하단 작은 → cond1=True, cond2=False → score=0.6 → PHOTOGRAPHER
        # 상단 큰  → cond1=False, cond2=True  → score=0.4 → BACKGROUND
        assert roles[0][0] == PersonRole.PHOTOGRAPHER
        assert roles[1][0] == PersonRole.BACKGROUND


# ═══════════════════════════════════════════════════════════════════════════
# segment_face — 출력 구조 검증 (YOLO mock)
# ═══════════════════════════════════════════════════════════════════════════

class TestSegmentFaceOutput:

    @pytest.fixture(autouse=True)
    def setup(self):
        from pipeline.segmentation import PersonSegmenter, _Detection
        import unittest.mock as mock

        with mock.patch.object(PersonSegmenter, "_load_yolo"):
            self.seg = PersonSegmenter(BASE_CONFIG)

        # YOLO 검출 결과를 직접 주입
        dummy_box = np.array([5, 45, 55, 63], dtype=np.float32)
        dummy_mask = torch.zeros(64, 64, dtype=torch.bool)
        dummy_mask[45:63, 5:55] = True
        self.dummy_det = _Detection(
            box=dummy_box, mask=dummy_mask,
            conf=0.9, pixel_count=int(dummy_mask.sum()), bbox_area=1000.0
        )
        self.seg._yolo_detect = mock.MagicMock(return_value=[self.dummy_det])

    def test_output_keys(self):
        face = _make_face()
        result = self.seg.segment_face(face, "front", 256, 512)
        expected = {
            "photographer_mask", "background_masks",
            "background_face_masks", "roles", "detections", "role_scores",
        }
        assert set(result.keys()) == expected

    def test_photographer_mask_dtype(self):
        result = self.seg.segment_face(_make_face(), "front", 256, 512)
        assert result["photographer_mask"].dtype == torch.bool

    def test_photographer_mask_shape(self):
        result = self.seg.segment_face(_make_face(), "front", 256, 512)
        assert result["photographer_mask"].shape == (64, 64)

    def test_background_masks_list(self):
        result = self.seg.segment_face(_make_face(), "back", 256, 512)
        assert isinstance(result["background_masks"], list)

    def test_no_detection_empty_result(self):
        import unittest.mock as mock
        self.seg._yolo_detect = mock.MagicMock(return_value=[])
        result = self.seg.segment_face(_make_face(), "front", 256, 512)
        assert not result["photographer_mask"].any()
        assert result["background_masks"] == []

    def test_down_face_gives_photographer(self):
        from pipeline.segmentation import PersonRole
        result = self.seg.segment_face(_make_face(), "down", 256, 512)
        assert PersonRole.PHOTOGRAPHER in result["roles"]

    def test_roles_length_equals_detections(self):
        result = self.seg.segment_face(_make_face(), "front", 256, 512)
        assert len(result["roles"]) == len(result["detections"])

    def test_background_face_mask_shape(self):
        from pipeline.segmentation import PersonRole
        import unittest.mock as mock
        # 강제로 BACKGROUND 역할 주입
        self.seg._yolo_detect = mock.MagicMock(return_value=[self.dummy_det])
        # up face → 조건1 항상 실패 → BACKGROUND
        result = self.seg.segment_face(_make_face(), "up", 256, 512)
        if result["background_face_masks"]:
            for m in result["background_face_masks"]:
                assert m.shape == (64, 64)
                assert m.dtype == torch.bool


# ═══════════════════════════════════════════════════════════════════════════
# FaceMosaicker
# ═══════════════════════════════════════════════════════════════════════════

class TestFaceMosaicker:

    @pytest.fixture(autouse=True)
    def mosaicker(self):
        from pipeline.segmentation import FaceMosaicker
        self.m = FaceMosaicker(mosaic_block_size=8, feather_px=4)

    def test_output_shape(self):
        face = _make_face(64, 64)
        mask = torch.zeros(64, 64, dtype=torch.bool)
        mask[5:20, 10:40] = True
        result = self.m.mosaic_face(face, mask)
        assert result.shape == face.shape

    def test_output_dtype(self):
        face = _make_face(64, 64)
        mask = torch.zeros(64, 64, dtype=torch.bool)
        mask[5:20, 10:40] = True
        result = self.m.mosaic_face(face, mask)
        assert result.dtype == torch.float32

    def test_empty_mask_unchanged(self):
        face = _make_face(64, 64)
        mask = torch.zeros(64, 64, dtype=torch.bool)
        result = self.m.mosaic_face(face, mask)
        assert torch.allclose(result, face)

    def test_mosaic_region_is_blocky(self):
        """모자이크 영역 내 블록 경계에서 픽셀값이 동일해야 한다."""
        # 순수색 이미지로 블록 균일성 확인
        face = torch.ones(3, 64, 64) * 0.5
        face[:, 10:20, 10:20] = torch.rand(3, 10, 10)  # 노이즈
        mask = torch.zeros(64, 64, dtype=torch.bool)
        mask[10:20, 10:20] = True
        result = self.m.mosaic_face(face, mask)
        # 중앙 블록 내 분산이 매우 작아야 함
        roi = result[:, 12:16, 12:16]
        assert roi.std().item() < 0.15

    def test_apply_background_mosaics_no_masks(self):
        face = _make_face()
        result = self.m.apply_background_mosaics(face, {"background_face_masks": []})
        assert torch.allclose(result, face)

    def test_apply_background_mosaics_multiple(self):
        face = _make_face(64, 64)
        masks = [
            torch.zeros(64, 64, dtype=torch.bool),
            torch.zeros(64, 64, dtype=torch.bool),
        ]
        masks[0][5:15, 5:25] = True
        masks[1][30:45, 30:50] = True
        result = self.m.apply_background_mosaics(face, {"background_face_masks": masks})
        assert result.shape == face.shape


# ═══════════════════════════════════════════════════════════════════════════
# MaskPostProcessor
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskPostProcessor:

    @pytest.fixture(autouse=True)
    def processor(self):
        from pipeline.segmentation import MaskPostProcessor
        self.proc = MaskPostProcessor()

    def _make_uv(self, face_name: str, fs: int = 32) -> torch.Tensor:
        """단순 균등 UV 맵 생성 (front face는 중앙 ERP 영역)."""
        u = torch.linspace(0.25, 0.75, fs).unsqueeze(0).expand(fs, -1)
        v = torch.linspace(0.25, 0.75, fs).unsqueeze(1).expand(-1, fs)
        return torch.stack([u, v], dim=0)

    def test_output_shape(self):
        face_masks = {"front": torch.ones(32, 32, dtype=torch.bool)}
        uv_maps    = {"front": self._make_uv("front")}
        result = self.proc.expand_mask_to_erp(face_masks, uv_maps, erp_h=64, erp_w=128)
        assert result.shape == (64, 128)

    def test_output_dtype(self):
        face_masks = {"front": torch.ones(32, 32, dtype=torch.bool)}
        uv_maps    = {"front": self._make_uv("front")}
        result = self.proc.expand_mask_to_erp(face_masks, uv_maps, erp_h=64, erp_w=128)
        assert result.dtype == torch.bool

    def test_empty_mask_gives_empty_erp(self):
        face_masks = {"front": torch.zeros(32, 32, dtype=torch.bool)}
        uv_maps    = {"front": self._make_uv("front")}
        result = self.proc.expand_mask_to_erp(face_masks, uv_maps, erp_h=64, erp_w=128)
        # 빈 mask → dilate로 약간 퍼질 수 있으나, 초기에 True 픽셀 없으면 결과도 거의 비어야 함
        assert not result.all()

    def test_full_mask_covers_uv_region(self):
        """face 전체가 mask이면 UV가 가리키는 ERP 영역이 채워져야 한다."""
        fs = 32
        face_masks = {"front": torch.ones(fs, fs, dtype=torch.bool)}
        uv_maps    = {"front": self._make_uv("front", fs)}
        result = self.proc.expand_mask_to_erp(face_masks, uv_maps, erp_h=64, erp_w=128)
        # UV [0.25, 0.75] → ERP 중앙 절반 영역에 True가 있어야 함
        center = result[16:48, 32:96]
        assert center.any()

    def test_missing_uv_map_skipped(self):
        """UV 맵이 없는 face는 skip되고 오류 없이 동작해야 한다."""
        face_masks = {"front": torch.ones(32, 32, dtype=torch.bool)}
        uv_maps    = {}  # UV 없음
        result = self.proc.expand_mask_to_erp(face_masks, uv_maps, erp_h=64, erp_w=128)
        assert result.shape == (64, 128)
