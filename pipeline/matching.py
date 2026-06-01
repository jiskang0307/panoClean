"""
matching.py — 멀티-뷰 feature matching 및 투시 변환 기반 배경 복원.

지원 매처:
  - LoFTR  (kornia, CUDA 가속)
  - SIFT   (OpenCV, CPU)
  - SuperPoint (kornia)

처리 흐름:
  1. 소스 face(배경 후보)와 타깃 face(사람 존재)에서 keypoint 매칭
  2. RANSAC으로 호모그래피 추정
  3. 호모그래피로 소스를 타깃에 워핑
  4. 마스크 내 유효 픽셀 비율(coverage) 계산
  5. coverage ≥ min_coverage_ratio면 warp 결과 반환, 아니면 None
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
from loguru import logger


MatcherType = Literal["loftr", "superpoint", "sift"]


class BackgroundMatcher:
    """멀티-뷰 매칭으로 소스 이미지에서 배경을 추출해 타깃의 마스크 영역을 채운다."""

    def __init__(
        self,
        matcher_type: MatcherType = "loftr",
        min_match_count: int = 10,
        min_coverage_ratio: float = 0.85,
        device: str = "cuda",
        loftr_pretrained: str = "outdoor",
    ) -> None:
        self.matcher_type = matcher_type
        self.min_match_count = min_match_count
        self.min_coverage_ratio = min_coverage_ratio
        self.device = device

        if matcher_type == "loftr":
            self._init_loftr(loftr_pretrained)
        elif matcher_type == "superpoint":
            self._init_superpoint()
        else:
            logger.info("SIFT 매처 사용 (CPU)")

    # ── 초기화 ────────────────────────────────────────────────────────────

    def _init_loftr(self, pretrained: str) -> None:
        from kornia.feature import LoFTR

        logger.info(f"LoFTR 로드: pretrained={pretrained}")
        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()

    def _init_superpoint(self) -> None:
        from kornia.feature import SuperPoint

        logger.info("SuperPoint 로드")
        self.matcher = SuperPoint(pretrained=True).to(self.device).eval()

    # ── 공개 API ──────────────────────────────────────────────────────────

    def fill_mask_from_source(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        """
        소스 이미지를 타깃에 매칭·워핑하여 마스크 영역 커버리지를 계산.

        Args:
            target: 마스크 영역을 채울 타깃 face (HxWx3 uint8 BGR).
            source: 배경 후보 소스 face (HxWx3 uint8 BGR).
            mask  : 채워야 할 영역 (HxW bool).

        Returns:
            (warp 결과 또는 None, 마스크 내 유효 픽셀 비율)
        """
        pts_src, pts_dst = self._match_keypoints(source, target)

        if pts_src is None or len(pts_src) < self.min_match_count:
            logger.debug(f"매칭 부족: {0 if pts_src is None else len(pts_src)} points")
            return None, 0.0

        H, inlier_mask = cv2.findHomography(
            pts_src, pts_dst, cv2.RANSAC, ransacReprojThreshold=4.0
        )
        if H is None:
            return None, 0.0

        h, w = target.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))

        # 마스크 내 non-black 픽셀 비율로 coverage 계산
        mask_u8 = mask.astype(np.uint8)
        filled = (warped.sum(axis=2) > 0).astype(np.uint8) & mask_u8
        coverage = float(filled.sum()) / max(mask_u8.sum(), 1)

        logger.debug(f"coverage={coverage:.3f} (threshold={self.min_coverage_ratio})")

        if coverage >= self.min_coverage_ratio:
            return warped, coverage
        return None, coverage

    def best_fill(
        self,
        target: np.ndarray,
        sources: list[np.ndarray],
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        """sources 중 coverage가 가장 높은 결과를 반환."""
        best_warp, best_cov = None, 0.0
        for src in sources:
            warp, cov = self.fill_mask_from_source(target, src, mask)
            if cov > best_cov:
                best_cov = cov
                best_warp = warp
        return best_warp, best_cov

    # ── 내부 매칭 ─────────────────────────────────────────────────────────

    def _match_keypoints(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.matcher_type == "loftr":
            return self._loftr_match(img1, img2)
        elif self.matcher_type == "superpoint":
            return self._superpoint_match(img1, img2)
        else:
            return self._sift_match(img1, img2)

    def _loftr_match(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        def to_gray_tensor(img: np.ndarray) -> torch.Tensor:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            batch = {"image0": to_gray_tensor(img1), "image1": to_gray_tensor(img2)}
            self.matcher(batch)
            kp0 = batch["keypoints0"].cpu().numpy()
            kp1 = batch["keypoints1"].cpu().numpy()

        if len(kp0) == 0:
            return None, None
        return kp0.astype(np.float32), kp1.astype(np.float32)

    def _superpoint_match(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        from kornia.feature import match_smnn

        def extract(img: np.ndarray):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                return self.matcher(t)

        f1 = extract(img1)
        f2 = extract(img2)
        dists, idxs = match_smnn(f1["descriptors"][0], f2["descriptors"][0], th=0.85)
        pts1 = f1["keypoints"][0][idxs[:, 0]].cpu().numpy()
        pts2 = f2["keypoints"][0][idxs[:, 1]].cpu().numpy()
        return pts1.astype(np.float32), pts2.astype(np.float32)

    def _sift_match(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None, None

        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        raw = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]

        if len(good) < self.min_match_count:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2
