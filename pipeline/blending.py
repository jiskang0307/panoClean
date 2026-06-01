"""
blending.py — 다중 소스 합성 및 경계 블렌딩.

사람 제거 후 복원된 배경 패치를 원본 이미지에 자연스럽게 합성한다.
Poisson 블렌딩 또는 알파 페더링 방식을 지원한다.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from loguru import logger


BlendMode = Literal["poisson", "feather", "direct"]


class PatchBlender:
    """마스크 영역에 패치를 자연스럽게 합성."""

    def __init__(
        self,
        mode: BlendMode = "poisson",
        feather_radius: int = 10,
    ) -> None:
        self.mode = mode
        self.feather_radius = feather_radius

    # ── 공개 API ──────────────────────────────────────────────────────────

    def blend(
        self,
        base: np.ndarray,
        patch: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        base 이미지의 mask 영역을 patch로 교체하면서 자연스럽게 블렌딩.

        Args:
            base : 원본 이미지 (HxWx3 uint8 BGR).
            patch: 복원된 배경 패치 (HxWx3 uint8 BGR, base와 동일 크기).
            mask : 교체 대상 영역 (HxW bool).

        Returns:
            블렌딩 결과 이미지 (HxWx3 uint8 BGR).
        """
        if self.mode == "poisson":
            return self._poisson_blend(base, patch, mask)
        elif self.mode == "feather":
            return self._feather_blend(base, patch, mask)
        else:
            return self._direct_blend(base, patch, mask)

    def compose_from_warped(
        self,
        base: np.ndarray,
        warped: np.ndarray,
        person_mask: np.ndarray,
    ) -> np.ndarray:
        """
        워핑된 소스 이미지를 base에 합성.
        마스크 밖 영역은 base, 마스크 안은 warped를 우선 사용.
        """
        result = base.copy()
        valid = person_mask.astype(bool) & (warped.sum(axis=2) > 0)
        result[valid] = warped[valid]
        return result

    # ── 블렌딩 방식 ───────────────────────────────────────────────────────

    def _direct_blend(
        self, base: np.ndarray, patch: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        result = base.copy()
        result[mask.astype(bool)] = patch[mask.astype(bool)]
        return result

    def _feather_blend(
        self, base: np.ndarray, patch: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        kernel_size = self.feather_radius * 2 + 1
        alpha = cv2.GaussianBlur(
            mask.astype(np.float32),
            (kernel_size, kernel_size),
            self.feather_radius,
        )
        alpha = np.clip(alpha, 0, 1)[..., np.newaxis]
        result = (patch.astype(np.float32) * alpha + base.astype(np.float32) * (1 - alpha))
        return result.astype(np.uint8)

    def _poisson_blend(
        self, base: np.ndarray, patch: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8) * 255

        # Poisson 블렌딩의 center는 마스크 bounding box 중심
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._direct_blend(base, patch, mask)

        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        center = (x + w // 2, y + h // 2)

        try:
            result = cv2.seamlessClone(patch, base, mask_u8, center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            logger.warning(f"Poisson 블렌딩 실패 ({e}) — feather로 대체")
            result = self._feather_blend(base, patch, mask)

        return result
