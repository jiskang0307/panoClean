"""
matching.py — 멀티-뷰 Feature Matching 기반 배경 복원.

지원 매처:
  LoFTR     : kornia.feature.LoFTR (CUDA, dense matching)
  SuperPoint: kornia.feature.SuperPoint + match_smnn (CUDA)
  SIFT      : cv2.SIFT_create() (CPU fallback)

처리 흐름:
  find_homography → warp_background → select_best_sources → blend_multiple_sources

up/down face는 왜곡이 심해 LoFTR/SuperPoint 대신 SIFT를 자동으로 사용한다.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

MatcherType = Literal["loftr", "superpoint", "sift"]
POLAR_FACES = frozenset({"up", "down"})

# inlier_ratio 임계값 — 이 미만이면 낮은 신뢰도로 표시
_INLIER_RATIO_THRESHOLD = 0.3
# RANSAC 재투영 오차 임계값 (px)
_RANSAC_THRESH = 3.0
# 호모그래피 추정에 필요한 최소 매칭 수
_MIN_MATCHES = 10


class BackgroundMatcher:
    """
    다중 소스 이미지에서 feature matching으로 배경을 추출해 타깃 마스크를 채운다.

    모든 공개 메서드 I/O:
      이미지: (3, H, W) float32 torch.Tensor, 값 [0, 1]
      마스크: (H, W) bool torch.Tensor
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 불가 — CPU fallback")
            self.device = "cpu"

        self.matcher_type: MatcherType = config.get("feature_matcher", "loftr")
        self.min_match_count: int      = config.get("min_match_count", _MIN_MATCHES)
        self.min_coverage_ratio: float = config.get("min_coverage_ratio", 0.85)
        self.loftr_pretrained: str     = config.get("loftr_pretrained", "indoor")

        self._loftr     = None
        self._superpoint = None
        self._init_matcher(self.matcher_type)

        # SeamlessBlender를 내부 합성에 사용 (lazy import — blending.py에서)
        self._blender = None

    # ── 초기화 ────────────────────────────────────────────────────────────

    def _init_matcher(self, matcher_type: MatcherType) -> None:
        if matcher_type == "loftr":
            self._init_loftr()
        elif matcher_type == "superpoint":
            self._init_superpoint()
        else:
            logger.info("SIFT 매처 사용 (CPU)")

    def _init_loftr(self) -> None:
        try:
            from kornia.feature import LoFTR
            logger.info(f"LoFTR 로드: pretrained={self.loftr_pretrained}")
            self._loftr = LoFTR(pretrained=self.loftr_pretrained).to(self.device).eval()
        except Exception as e:
            logger.warning(f"LoFTR 로드 실패 ({e}) — SIFT fallback")
            self.matcher_type = "sift"

    def _init_superpoint(self) -> None:
        try:
            from kornia.feature import SuperPoint
            logger.info("SuperPoint 로드")
            self._superpoint = SuperPoint(pretrained=True).to(self.device).eval()
        except Exception as e:
            logger.warning(f"SuperPoint 로드 실패 ({e}) — SIFT fallback")
            self.matcher_type = "sift"

    def _get_blender(self):
        if self._blender is None:
            from pipeline.blending import SeamlessBlender
            self._blender = SeamlessBlender()
        return self._blender

    # ── 핵심 공개 메서드 ──────────────────────────────────────────────────

    def find_homography(
        self,
        src_face: torch.Tensor,
        dst_face: torch.Tensor,
        face_name: str = "front",
    ) -> tuple[np.ndarray | None, float]:
        """
        두 face 이미지 간 homography 행렬을 추정.

        Args:
            src_face : 소스 face (3,H,W) float32.
            dst_face : 타깃 face (3,H,W) float32.
            face_name: 극 영역(up/down)은 SIFT로 강제 전환.

        Returns:
            (3×3 homography ndarray 또는 None, inlier_ratio)
            inlier_ratio < 0.3이면 낮은 신뢰도.
        """
        force_sift = face_name in POLAR_FACES
        pts_src, pts_dst = self._match_keypoints(src_face, dst_face, force_sift=force_sift)

        if pts_src is None or len(pts_src) < self.min_match_count:
            n = 0 if pts_src is None else len(pts_src)
            logger.debug(f"[{face_name}] 매칭 부족: {n}개 (최소 {self.min_match_count})")
            return None, 0.0

        H, inlier_mask = cv2.findHomography(
            pts_src, pts_dst,
            cv2.RANSAC,
            ransacReprojThreshold=_RANSAC_THRESH,
        )
        if H is None:
            logger.debug(f"[{face_name}] RANSAC homography 추정 실패")
            return None, 0.0

        inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask)
        if inlier_ratio < _INLIER_RATIO_THRESHOLD:
            logger.debug(f"[{face_name}] 낮은 신뢰도 homography: inlier_ratio={inlier_ratio:.3f}")

        return H, inlier_ratio

    def warp_background(
        self,
        src_face: torch.Tensor,
        dst_mask: torch.Tensor,
        H_matrix: np.ndarray,
    ) -> torch.Tensor:
        """
        src_face를 H_matrix로 변환해 dst 좌표계로 warping.

        유효성 마스크(validity mask)를 함께 워핑해 검은 픽셀과 실제 배경을 구분한다.

        Args:
            src_face : 소스 face (3,H,W) float32.
            dst_mask : 채울 영역 (H,W) bool (크기 참조용).
            H_matrix : 3×3 homography ndarray.

        Returns:
            (3,H,W) float32 tensor — mask 밖 또는 유효하지 않은 픽셀은 0.
        """
        _, h, w = src_face.shape
        src_bgr = self._t2bgr(src_face)

        # 유효 픽셀을 추적하기 위한 validity mask 워핑
        validity = np.ones((h, w), dtype=np.float32)
        warped_bgr   = cv2.warpPerspective(src_bgr,  H_matrix, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
        warped_valid = cv2.warpPerspective(validity, H_matrix, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)

        # 유효하지 않은 픽셀 제거
        invalid = warped_valid < 0.5
        warped_bgr[invalid] = 0

        return self._bgr2t(warped_bgr)

    def select_best_sources(
        self,
        target_face: torch.Tensor,
        source_faces: list[torch.Tensor],
        mask: torch.Tensor,
        top_k: int = 3,
        face_name: str = "front",
    ) -> list[tuple[torch.Tensor, np.ndarray, float]]:
        """
        소스 후보 중 mask 커버가 좋은 상위 top_k를 반환.

        Args:
            target_face : 타깃 face (3,H,W).
            source_faces: 소스 후보 리스트.
            mask        : 채울 영역 (H,W) bool.
            top_k       : 반환할 최대 소스 수.
            face_name   : 극 영역 판별용.

        Returns:
            [(warped_bg, H_matrix, coverage_ratio), ...] — coverage 내림차순.
        """
        results: list[tuple[torch.Tensor, np.ndarray, float]] = []
        mask_total = int(mask.sum())
        if mask_total == 0:
            return results

        for i, src in enumerate(source_faces):
            try:
                H, inlier_ratio = self.find_homography(src, target_face, face_name)
            except Exception as e:
                logger.warning(f"[{face_name}] 소스 {i}: homography 예외 — {e}")
                continue

            if H is None:
                continue

            warped = self.warp_background(src, mask, H)

            # 마스크 내 유효 픽셀 비율
            valid_in_mask = mask & (warped.sum(0) > 0.01)
            coverage = float(valid_in_mask.sum()) / max(mask_total, 1)
            logger.debug(f"[{face_name}] 소스 {i}: coverage={coverage:.3f}, inlier={inlier_ratio:.3f}")

            results.append((warped, H, coverage))

        # coverage 내림차순 정렬 후 top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def blend_multiple_sources(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        sources: list[tuple[torch.Tensor, np.ndarray, float]],
        face_name: str = "front",
    ) -> torch.Tensor:
        """
        복수 소스의 warped background를 coverage 기반 weighted blend.

        Args:
            target  : 원본 타깃 face (3,H,W).
            mask    : 채울 영역 (H,W) bool.
            sources : select_best_sources 반환값.
            face_name: 극 영역 판별 (SeamlessBlend 방식 결정).

        Returns:
            (3,H,W) float32 — mask 영역이 배경으로 채워진 face.
        """
        if not sources:
            return target

        _, h, w = target.shape
        cov_arr = np.array([c for _, _, c in sources], dtype=np.float32)

        # Softmax 가중치 (coverage가 높을수록 비중 증가)
        exp_c = np.exp(cov_arr - cov_arr.max())
        weights = exp_c / exp_c.sum()

        # ── weighted blend ────────────────────────────────────────────────
        accum    = torch.zeros_like(target)    # (3, H, W)
        w_map    = torch.zeros(h, w)           # (H, W)

        for (warped_bg, _, _), weight in zip(sources, weights):
            valid = mask & (warped_bg.sum(0) > 0.01)
            if not valid.any():
                continue
            accum[:, valid] += warped_bg[:, valid] * float(weight)
            w_map[valid] += float(weight)

        covered = w_map > 0
        if not covered.any():
            logger.warning(f"[{face_name}] 유효 소스 없음 — 원본 유지")
            return target

        # 정규화: (3,N) / (1,N)
        accum[:, covered] = accum[:, covered] / w_map[covered].unsqueeze(0)

        # ── 타깃 위에 합성 ─────────────────────────────────────────────────
        result = target.clone()
        result[:, mask] = accum[:, mask]

        # ── 경계 seamless blend ────────────────────────────────────────────
        blender = self._get_blender()
        try:
            target_bgr  = self._t2bgr(target)
            patched_bgr = self._t2bgr(result)
            mask_u8     = mask.numpy().astype(np.uint8) * 255  # bool → 0/255

            # 극 영역은 Poisson 실패 확률이 높아 바로 alpha fallback
            if face_name in POLAR_FACES:
                blended_bgr = blender.alpha_blend_edge(target_bgr, patched_bgr, mask_u8)
            else:
                blended_bgr = blender.poisson_blend(target_bgr, patched_bgr, mask_u8)

            return self._bgr2t(blended_bgr)
        except Exception as e:
            logger.warning(f"[{face_name}] seamless blend 실패 ({e}) — 직접 합성 반환")
            return result

    # ── 내부: 매칭 ───────────────────────────────────────────────────────

    def _match_keypoints(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        force_sift: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(3,H,W) 두 텐서에서 대응 keypoint 쌍을 추출."""
        if force_sift or self.matcher_type == "sift":
            return self._sift_match(self._t2gray_np(src), self._t2gray_np(dst))
        elif self.matcher_type == "loftr":
            return self._loftr_match(src, dst)
        else:
            return self._superpoint_match(src, dst)

    def _loftr_match(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        gray0 = self._t2gray_tensor(src)   # (1,1,H,W)
        gray1 = self._t2gray_tensor(dst)

        with torch.inference_mode():
            batch = {"image0": gray0, "image1": gray1}
            out = self._loftr(batch)

        # kornia LoFTR: 결과가 batch dict 내에 in-place로 들어갈 수도 있고
        # 반환값으로 나올 수도 있음 (버전에 따라 다름)
        if isinstance(out, dict):
            kp0 = out.get("keypoints0", batch.get("keypoints0"))
            kp1 = out.get("keypoints1", batch.get("keypoints1"))
        else:
            kp0 = batch.get("keypoints0")
            kp1 = batch.get("keypoints1")

        if kp0 is None or len(kp0) == 0:
            return None, None

        # 배치 차원 제거
        if kp0.dim() == 3:
            kp0, kp1 = kp0[0], kp1[0]

        return kp0.cpu().numpy().astype(np.float32), kp1.cpu().numpy().astype(np.float32)

    def _superpoint_match(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            from kornia.feature import match_smnn
        except ImportError:
            return self._sift_match(self._t2gray_np(src), self._t2gray_np(dst))

        def extract(t: torch.Tensor) -> dict:
            gray = self._t2gray_tensor(t)   # (1,1,H,W)
            with torch.inference_mode():
                return self._superpoint({"image": gray})

        feat0 = extract(src)
        feat1 = extract(dst)

        # descriptor shape: (1, N, D) 또는 (1, D, N) — 버전별 차이
        desc0 = feat0["descriptors"][0]  # (N, D) or (D, N)
        desc1 = feat1["descriptors"][0]

        # match_smnn 은 (D, N) 형식 기대
        if desc0.shape[0] < desc0.shape[1]:
            pass  # already (D, N)
        else:
            desc0, desc1 = desc0.T, desc1.T  # (N,D) → (D,N)

        try:
            dists, idxs = match_smnn(desc0, desc1, th=0.85)
        except Exception as e:
            logger.debug(f"match_smnn 실패 ({e})")
            return None, None

        if len(idxs) == 0:
            return None, None

        kp0 = feat0["keypoints"][0]   # (N, 2) or (1, N, 2)
        kp1 = feat1["keypoints"][0]
        if kp0.dim() == 3:
            kp0, kp1 = kp0[0], kp1[0]

        # idxs: (M, 2) — column 0 = index in desc0, column 1 = index in desc1
        if idxs.dim() == 2 and idxs.shape[1] == 2:
            pts0 = kp0[idxs[:, 0]].cpu().numpy().astype(np.float32)
            pts1 = kp1[idxs[:, 1]].cpu().numpy().astype(np.float32)
        else:
            # older API: idxs is (M,) indexing desc1 for each desc0
            pts0 = kp0[:len(idxs)].cpu().numpy().astype(np.float32)
            pts1 = kp1[idxs].cpu().numpy().astype(np.float32)

        return pts0, pts1

    def _sift_match(
        self,
        gray0: np.ndarray,
        gray1: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """SIFT + FLANN 매칭. gray0, gray1은 uint8 단채널 이미지."""
        sift = cv2.SIFT_create()
        kp0, des0 = sift.detectAndCompute(gray0, None)
        kp1, des1 = sift.detectAndCompute(gray1, None)

        if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
            return None, None

        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 50}
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            raw = flann.knnMatch(des0, des1, k=2)
        except cv2.error:
            return None, None

        good = [m for m, n in raw if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < self.min_match_count:
            return None, None

        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
        return pts0, pts1

    # ── 내부: 타입 변환 ───────────────────────────────────────────────────

    @staticmethod
    def _t2bgr(t: torch.Tensor) -> np.ndarray:
        """(3,H,W) float32 [0,1] RGB → (H,W,3) uint8 BGR."""
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _bgr2t(self, arr: np.ndarray) -> torch.Tensor:
        """(H,W,3) uint8 BGR → (3,H,W) float32 [0,1]."""
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float() / 255.0
        return t.permute(2, 0, 1).to(self.device)

    def _t2gray_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """(3,H,W) → (1,1,H,W) float32 grayscale tensor on device."""
        # ITU-R BT.601
        r, g, b = t[0:1], t[1:2], t[2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.unsqueeze(0).to(self.device)  # (1,1,H,W)

    @staticmethod
    def _t2gray_np(t: torch.Tensor) -> np.ndarray:
        """(3,H,W) → (H,W) uint8 grayscale ndarray."""
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # ── 하위호환 래퍼 (batch_runner.py 구 API) ───────────────────────────

    def fill_mask_from_source(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        """
        [하위호환] numpy BGR 이미지 기반 단일 소스 fill.
        새 코드는 select_best_sources / blend_multiple_sources를 사용하라.
        """
        def bgr_np_to_t(img: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        t_tgt = bgr_np_to_t(target)
        t_src = bgr_np_to_t(source)
        t_mask = torch.from_numpy(mask.astype(bool))

        H, _ = self.find_homography(t_src, t_tgt)
        if H is None:
            return None, 0.0

        warped = self.warp_background(t_src, t_mask, H)
        valid_in_mask = t_mask & (warped.sum(0) > 0.01)
        coverage = float(valid_in_mask.sum()) / max(int(t_mask.sum()), 1)

        if coverage >= self.min_coverage_ratio:
            warped_bgr = self._t2bgr(warped)
            return warped_bgr, coverage
        return None, coverage

    def best_fill(
        self,
        target: np.ndarray,
        sources: list[np.ndarray],
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        """[하위호환] sources 중 coverage 최대 결과 반환 (numpy BGR)."""
        best_warp, best_cov = None, 0.0
        for src in sources:
            warp, cov = self.fill_mask_from_source(target, src, mask)
            if cov > best_cov:
                best_cov = cov
                best_warp = warp
        return best_warp, best_cov
