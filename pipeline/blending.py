"""
blending.py — 경계 seamless blending 및 coverage 계산.

SeamlessBlender: Poisson(MIXED_CLONE) / Gaussian feathering 방식
PatchBlender   : Phase 1 구현 유지 (하위호환)
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════
# SeamlessBlender  (Phase 4 신규)
# ═══════════════════════════════════════════════════════════════════════════

class SeamlessBlender:
    """
    마스크 경계를 자연스럽게 합성하는 블렌더.

    poisson_blend : cv2.seamlessClone(MIXED_CLONE) — 최고 품질, 극 영역 실패 가능
    alpha_blend_edge: Gaussian feathering — 항상 성공, Poisson 실패 시 fallback
    """

    def poisson_blend(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        MIXED_CLONE 방식 Poisson blending.

        source의 배경 질감을 살리면서 target의 gradient 방향을 보존한다.
        mask 경계가 넓을수록 부드럽게 전환된다.

        Args:
            target: 결과가 삽입될 기저 이미지 (H,W,3 uint8 BGR).
            source: 채울 패치 이미지 (H,W,3 uint8 BGR, target과 동일 크기).
            mask  : 삽입 영역 (H,W uint8, 0 또는 255).

        Returns:
            H,W,3 uint8 BGR — 블렌딩 결과.
        """
        mask_u8 = _ensure_u8_mask(mask)

        if mask_u8.sum() == 0:
            return target.copy()

        # seamlessClone은 mask가 이미지 경계에 닿으면 실패 → 1px 안쪽으로 침식
        mask_safe = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
        if mask_safe.sum() == 0:
            logger.debug("Poisson: mask 너무 작음 — alpha fallback")
            return self.alpha_blend_edge(target, source, mask_u8)

        contours, _ = cv2.findContours(
            mask_safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return self.alpha_blend_edge(target, source, mask_u8)

        pts = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(pts)

        # 경계를 넘어서는 center는 seamlessClone 실패 원인
        img_h, img_w = target.shape[:2]
        cx = int(np.clip(x + w // 2, w // 2 + 1, img_w - w // 2 - 1))
        cy = int(np.clip(y + h // 2, h // 2 + 1, img_h - h // 2 - 1))

        try:
            result = cv2.seamlessClone(
                source, target, mask_safe, (cx, cy), cv2.MIXED_CLONE
            )
        except cv2.error as e:
            logger.debug(f"Poisson 실패 ({e}) — alpha fallback")
            result = self.alpha_blend_edge(target, source, mask_u8)

        return result

    def alpha_blend_edge(
        self,
        target: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        feather_px: int = 20,
    ) -> np.ndarray:
        """
        Gaussian feathering으로 mask 경계를 부드럽게 전환.

        Poisson이 실패하는 극 영역이나 작은 마스크에 사용한다.

        Args:
            target    : H,W,3 uint8 BGR.
            source    : H,W,3 uint8 BGR.
            mask      : H,W uint8.
            feather_px: Gaussian 블러 반경 (클수록 전환이 부드러움).

        Returns:
            H,W,3 uint8 BGR.
        """
        mask_u8 = _ensure_u8_mask(mask)

        ksize = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(
            mask_u8.astype(np.float32) / 255.0,
            (ksize, ksize),
            feather_px,
        )
        alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]  # (H,W,1)

        tgt_f = target.astype(np.float32)
        src_f = source.astype(np.float32)
        blended = src_f * alpha + tgt_f * (1.0 - alpha)
        return blended.astype(np.uint8)

    @staticmethod
    def compute_coverage(
        mask: torch.Tensor,
        filled_mask: torch.Tensor,
    ) -> float:
        """
        원본 mask 중 배경으로 채워진 비율.

        반환값 < config["min_coverage_ratio"] 이면 LaMa 보완 필요.

        Args:
            mask       : 원본 사람 mask (H,W) bool.
            filled_mask: 배경 매칭으로 채워진 영역 (H,W) bool.

        Returns:
            [0.0, 1.0] 범위의 float.
        """
        total = int(mask.sum())
        if total == 0:
            return 1.0
        covered = int((mask & filled_mask).sum())
        return covered / total


# ═══════════════════════════════════════════════════════════════════════════
# PatchBlender  (Phase 1 유지 — 하위호환)
# ═══════════════════════════════════════════════════════════════════════════

BlendMode = Literal["poisson", "feather", "direct"]


class PatchBlender:
    """마스크 영역에 패치를 자연스럽게 합성. (Phase 1 호환)"""

    def __init__(
        self,
        mode: BlendMode = "poisson",
        feather_radius: int = 10,
    ) -> None:
        self.mode = mode
        self.feather_radius = feather_radius
        self._seamless = SeamlessBlender()

    def blend(
        self,
        base: np.ndarray,
        patch: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """mask 영역을 patch로 교체하면서 자연스럽게 블렌딩."""
        if self.mode == "poisson":
            return self._seamless.poisson_blend(base, patch, mask)
        elif self.mode == "feather":
            return self._seamless.alpha_blend_edge(
                base, patch, mask, feather_px=self.feather_radius
            )
        return self._direct_blend(base, patch, mask)

    def compose_from_warped(
        self,
        base: np.ndarray,
        warped: np.ndarray,
        person_mask: np.ndarray,
    ) -> np.ndarray:
        """워핑된 소스를 base에 합성."""
        result = base.copy()
        valid = person_mask.astype(bool) & (warped.sum(axis=2) > 0)
        result[valid] = warped[valid]
        return result

    @staticmethod
    def _direct_blend(
        base: np.ndarray,
        patch: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        result = base.copy()
        result[mask.astype(bool)] = patch[mask.astype(bool)]
        return result


# ── 내부 유틸 ─────────────────────────────────────────────────────────────

def _ensure_u8_mask(mask: np.ndarray) -> np.ndarray:
    """어떤 dtype이든 0/255 uint8 마스크로 정규화."""
    # bool, 0/1 uint8, 0/1 float 모두 → 0/255 uint8
    return (mask.astype(bool).astype(np.uint8)) * 255
