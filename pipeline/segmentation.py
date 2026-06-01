"""
segmentation.py — YOLO11-seg 및 SAM2를 이용한 사람 인스턴스 마스크 생성.

처리 흐름:
  1. YOLO11-seg로 사람 bounding box + 거친 마스크 추출
  2. SAM2로 box prompt 기반 정밀 마스크 생성 (선택적)
  3. 마스크 팽창(dilate) 후 반환
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger


class PersonSegmentor:
    """YOLO11-seg / SAM2 기반 사람 마스크 생성기."""

    def __init__(
        self,
        yolo_model_path: str | Path = "yolo11x-seg.pt",
        sam2_model_path: str | Path | None = None,
        sam2_config: str = "sam2_hiera_l.yaml",
        person_class_id: int = 0,
        mask_dilate_px: int = 15,
        device: str = "cuda",
        use_sam2: bool = False,
    ) -> None:
        self.person_class_id = person_class_id
        self.mask_dilate_px = mask_dilate_px
        self.device = device
        self.use_sam2 = use_sam2

        self._load_yolo(yolo_model_path)
        if use_sam2 and sam2_model_path:
            self._load_sam2(sam2_model_path, sam2_config)

    # ── 모델 로드 ──────────────────────────────────────────────────────────

    def _load_yolo(self, model_path: str | Path) -> None:
        from ultralytics import YOLO

        logger.info(f"YOLO 모델 로드: {model_path}")
        self.yolo = YOLO(str(model_path))
        self.yolo.to(self.device)

    def _load_sam2(self, model_path: str | Path, config: str) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"SAM2 모델 로드: {model_path}")
            sam2_model = build_sam2(config, str(model_path), device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        except ImportError:
            logger.warning("SAM2 미설치 — YOLO 마스크만 사용합니다.")
            self.use_sam2 = False

    # ── 공개 API ──────────────────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 사람을 검출하고 합산 마스크를 반환.

        Args:
            image: HxWx3 uint8 BGR ndarray.

        Returns:
            HxW bool ndarray — True 위치가 사람 영역.
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        boxes, yolo_masks = self._yolo_predict(image)
        if not boxes:
            return combined_mask.astype(bool)

        if self.use_sam2 and hasattr(self, "sam2_predictor"):
            masks = self._sam2_refine(image, boxes)
        else:
            masks = yolo_masks

        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

        return self._dilate(combined_mask).astype(bool)

    def segment_faces(
        self, faces: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """CubeMap 6-face 딕셔너리에 대해 각각 segment 적용."""
        return {name: self.segment(face) for name, face in faces.items()}

    # ── 내부 처리 ─────────────────────────────────────────────────────────

    def _yolo_predict(
        self, image: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        results = self.yolo.predict(
            source=image,
            classes=[self.person_class_id],
            verbose=False,
            device=self.device,
        )

        boxes: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        h, w = image.shape[:2]

        for result in results:
            if result.masks is None:
                continue
            for seg_mask, box in zip(result.masks.data, result.boxes.xyxy):
                mask_resized = cv2.resize(
                    seg_mask.cpu().numpy().astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                masks.append((mask_resized > 0.5).astype(np.uint8))
                boxes.append(box.cpu().numpy())

        return boxes, masks

    def _sam2_refine(
        self, image: np.ndarray, boxes: list[np.ndarray]
    ) -> list[np.ndarray]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam2_predictor.set_image(rgb)

        masks: list[np.ndarray] = []
        for box in boxes:
            pred_masks, _, _ = self.sam2_predictor.predict(
                box=box[None],
                multimask_output=False,
            )
            masks.append(pred_masks[0].astype(np.uint8))

        return masks

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_dilate_px <= 0:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.mask_dilate_px * 2 + 1, self.mask_dilate_px * 2 + 1),
        )
        return cv2.dilate(mask, kernel, iterations=1)
