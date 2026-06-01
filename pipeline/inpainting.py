"""
inpainting.py — LaMa 기반 마스크 영역 복원.

simple-lama-inpainting 라이브러리를 래핑하여
배경 매칭으로 채우지 못한 잔여 영역을 생성형 inpainting으로 복원한다.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class LamaInpainter:
    """LaMa(Large Mask inpainting) 기반 이미지 복원기."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from simple_lama_inpainting import SimpleLama

            logger.info(f"LaMa 모델 로드 (device={self.device})")
            self._model = SimpleLama(device=self.device)
        except ImportError as e:
            raise RuntimeError(
                "simple-lama-inpainting 미설치. "
                "`pip install simple-lama-inpainting` 실행 후 재시도하세요."
            ) from e

    # ── 공개 API ──────────────────────────────────────────────────────────

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        마스크 영역을 LaMa로 복원.

        Args:
            image: HxWx3 uint8 RGB ndarray.
            mask : HxW bool or uint8 ndarray (True/1 = 복원 대상).

        Returns:
            HxWx3 uint8 RGB 복원 이미지.
        """
        self._load()

        mask_u8 = (mask.astype(np.uint8) * 255)
        if mask_u8.sum() == 0:
            logger.debug("빈 마스크 — inpainting 생략")
            return image

        from PIL import Image

        pil_img = Image.fromarray(image)
        pil_mask = Image.fromarray(mask_u8).convert("L")

        result = self._model(pil_img, pil_mask)
        return np.array(result)

    def inpaint_bgr(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """BGR 이미지를 RGB로 변환 후 inpaint하고 다시 BGR로 반환."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgb = self.inpaint(rgb, mask)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    def inpaint_residual(
        self,
        image: np.ndarray,
        full_mask: np.ndarray,
        filled_mask: np.ndarray,
    ) -> np.ndarray:
        """
        배경 매칭으로 채우지 못한 잔여 영역만 LaMa로 처리.

        Args:
            image      : 현재까지 복원된 이미지 (BGR).
            full_mask  : 원본 사람 마스크 전체.
            filled_mask: 배경 매칭으로 이미 채워진 영역.

        Returns:
            잔여 영역이 추가 복원된 BGR 이미지.
        """
        residual = full_mask.astype(bool) & ~filled_mask.astype(bool)
        if not residual.any():
            logger.debug("잔여 영역 없음 — LaMa 생략")
            return image
        logger.info(f"LaMa inpainting: 잔여 픽셀 수={residual.sum()}")
        return self.inpaint_bgr(image, residual)
