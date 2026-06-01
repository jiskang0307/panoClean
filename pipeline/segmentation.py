"""
segmentation.py — CubeMap face별 사람 검출, 역할 분류, 마스크 생성.

처리 흐름:
  1. YOLO11-seg로 person bbox + 거친 mask 추출 (conf >= 0.4)
  2. SAM2 box prompt로 정밀 mask 생성
  3. classify_persons()로 PHOTOGRAPHER / BACKGROUND 판별
  4. PHOTOGRAPHER mask: dilate(15px) + fill_holes
  5. BACKGROUND: 얼굴 bbox 크롭으로 모자이크 영역 추정
"""

from __future__ import annotations

import math
from enum import Enum
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy import ndimage as ndi


# ── 역할 정의 ─────────────────────────────────────────────────────────────

class PersonRole(Enum):
    PHOTOGRAPHER = "photographer"   # 완전 제거
    BACKGROUND   = "background"     # 얼굴 모자이크


class _Detection(NamedTuple):
    box: np.ndarray          # xyxy float32 (face 좌표계)
    mask: torch.Tensor       # (H,W) bool, face 좌표계
    conf: float
    pixel_count: int         # mask True 픽셀 수
    bbox_area: float


# ── face별 ERP y 범위 (정규화 0~1) ───────────────────────────────────────
# face 중심 y를 ERP 전체 높이 기준으로 환산할 때 쓰는 매핑
_FACE_ERP_Y_NORM: dict[str, tuple[float, float]] = {
    "front": (0.25, 0.75),
    "back":  (0.25, 0.75),
    "left":  (0.25, 0.75),
    "right": (0.25, 0.75),
    "up":    (0.0,  0.25),
    "down":  (0.75, 1.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# PersonSegmenter
# ═══════════════════════════════════════════════════════════════════════════

class PersonSegmenter:
    """YOLO11-seg + SAM2 기반 사람 검출 및 역할 분류."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 불가 — CPU fallback")
            self.device = "cpu"

        self.person_class_id: int  = config.get("person_class_id", 0)
        self.yolo_conf: float      = config.get("yolo_conf", 0.4)
        self.mask_dilate_px: int   = config.get("mask_dilate_px", 15)
        self.photographer_y_ratio: float     = config.get("photographer_y_ratio", 0.40)
        self.photographer_size_weight: float = config.get("photographer_size_weight", 0.5)

        self._load_yolo(config.get("yolo_model", "yolo11x-seg.pt"))
        if config.get("sam2_model"):
            self._load_sam2(config["sam2_model"], config.get("sam2_config", "sam2_hiera_l.yaml"))
        else:
            self.sam2 = None
            logger.info("SAM2 비활성화 — YOLO mask 단독 사용")

    # ── 모델 로드 ─────────────────────────────────────────────────────────

    def _load_yolo(self, path: str) -> None:
        from ultralytics import YOLO
        logger.info(f"YOLO 로드: {path}")
        self.yolo = YOLO(str(path))
        self.yolo.to(self.device)

    def _load_sam2(self, model_path: str, config_path: str) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            logger.info(f"SAM2 로드: {model_path}")
            model = build_sam2(config_path, str(model_path), device=self.device)
            self.sam2 = SAM2ImagePredictor(model)
        except ImportError:
            logger.warning("SAM2 미설치 — YOLO mask 단독 사용")
            self.sam2 = None

    # ── 역할 분류 ─────────────────────────────────────────────────────────

    def classify_persons(
        self,
        detections: list[_Detection],
        face_name: str,
        erp_h: int,
        erp_w: int,
        face_size: int,
    ) -> list[tuple[PersonRole, float]]:
        """
        각 detection에 대해 (역할, score)를 반환.

        PHOTOGRAPHER는 전체에서 최대 1명.
        """
        if not detections:
            return []

        # ── down face: 가장 큰 mask 1명만 PHOTOGRAPHER ────────────────────
        if face_name == "down":
            biggest_idx = self._biggest_mask_index(detections)
            return [
                (PersonRole.PHOTOGRAPHER, 1.0) if i == biggest_idx
                else (PersonRole.BACKGROUND, 0.0)
                for i in range(len(detections))
            ]

        # ── up face: 무조건 전부 BACKGROUND ──────────────────────────────
        if face_name == "up":
            return [(PersonRole.BACKGROUND, 0.0)] * len(detections)

        # ── 나머지 face: ERP y 오프셋 기반 조건1 + 조건2 ─────────────────
        # face 픽셀 y → ERP 절대 y (px) 환산
        # front/back/left/right는 ERP 25%~75% 구간에 매핑됨
        offset = erp_h * 0.25
        scale  = erp_h * 0.5 / face_size  # face_size px → ERP 50% 범위
        biggest_idx = self._biggest_mask_index(detections)

        scores = []
        for i, det in enumerate(detections):
            bbox_center_y = (det.box[1] + det.box[3]) / 2.0
            erp_y = offset + bbox_center_y * scale
            cond1 = erp_y >= erp_h * 0.75   # down face 경계 이상 = 바닥 근처

            cond2 = (i == biggest_idx)

            if cond1 and cond2:
                s = 1.0
            else:
                s = (0.6 if cond1 else 0.0) + (0.4 if cond2 else 0.0)

            scores.append(s)

        return self._scores_to_roles(scores, detections)

    # ── 단일 face 세그멘테이션 ────────────────────────────────────────────

    def segment_face(
        self,
        face_img: torch.Tensor,
        face_name: str,
        erp_h: int,
        erp_w: int,
    ) -> dict:
        """
        단일 CubeMap face를 처리해 결과 dict 반환.

        Args:
            face_img : (3, H, W) float32 tensor [0,1].
            face_name: "front" | "right" | "back" | "left" | "up" | "down".
            erp_h, erp_w: 원본 ERP 해상도 (역할 분류 좌표 환산용).

        Returns:
            {
              "photographer_mask"    : (H,W) bool tensor,
              "background_masks"     : list[(H,W) bool tensor],
              "background_face_masks": list[(H,W) bool tensor],
              "roles"                : list[PersonRole],
              "detections"           : list[_Detection],
              "role_scores"          : list[float],
            }
        """
        _, fh, fw = face_img.shape
        empty = {
            "photographer_mask":     torch.zeros(fh, fw, dtype=torch.bool),
            "background_masks":      [],
            "background_face_masks": [],
            "roles":                 [],
            "detections":            [],
            "role_scores":           [],
        }

        # 1) YOLO 검출
        img_np = self._tensor_to_bgr(face_img)
        detections = self._yolo_detect(img_np, fh, fw)
        if not detections:
            logger.debug(f"[{face_name}] 검출 없음")
            return empty

        # 2) SAM2 정밀화
        if self.sam2 is not None:
            detections = self._sam2_refine(img_np, detections, fh, fw)

        # 3) 역할 분류
        role_scores = self.classify_persons(detections, face_name, erp_h, erp_w, fh)

        # 4) 결과 조립
        photographer_mask = torch.zeros(fh, fw, dtype=torch.bool)
        bg_masks: list[torch.Tensor] = []
        bg_face_masks: list[torch.Tensor] = []
        roles: list[PersonRole] = []
        scores: list[float] = []

        n_photo = sum(1 for role, _ in role_scores if role == PersonRole.PHOTOGRAPHER)
        n_bg    = sum(1 for role, _ in role_scores if role == PersonRole.BACKGROUND)
        logger.info(
            f"[{face_name}] {len(detections)}명 검출 → "
            f"PHOTOGRAPHER x{n_photo}, BACKGROUND x{n_bg}"
        )
        if n_photo == 0:
            logger.warning(
                f"[{face_name}] PHOTOGRAPHER 미검출 — "
                "소스 이미지가 부족하거나 촬영자가 해당 면에 없을 수 있습니다."
            )

        raw_photo_masks: list[torch.Tensor] = []
        photo_boxes: list[np.ndarray] = []
        for det, (role, score) in zip(detections, role_scores):
            roles.append(role)
            scores.append(score)

            if role == PersonRole.PHOTOGRAPHER:
                raw_photo_masks.append(det.mask)
                photo_boxes.append(det.box)
            else:
                bg_masks.append(det.mask)
                bg_face_masks.append(self._face_region_mask(det.box, fh, fw))

        # 모든 PHOTOGRAPHER raw mask를 합산 후 face별 후처리
        if raw_photo_masks:
            raw_union = raw_photo_masks[0].clone()
            for m in raw_photo_masks[1:]:
                raw_union = raw_union | m

            # PHOTOGRAPHER bbox union (x1,y1,x2,y2)
            if photo_boxes:
                boxes = np.stack(photo_boxes)
                combined_bbox = (
                    int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                    int(boxes[:, 2].max()), int(boxes[:, 3].max()),
                )
            else:
                combined_bbox = None

            photographer_mask = self._postprocess_photographer_mask(
                raw_union, face_name, photographer_bbox=combined_bbox
            )

        # down face: 고정 타원과 OR 합산
        if face_name == "down":
            photographer_mask = self._down_face_ellipse_mask(
                photographer_mask, fh, fw
            )

        return {
            "photographer_mask":     photographer_mask,
            "background_masks":      bg_masks,
            "background_face_masks": bg_face_masks,
            "roles":                 roles,
            "detections":            detections,
            "role_scores":           scores,
        }

    # ── 전체 face 처리 ────────────────────────────────────────────────────

    def segment_all_faces(
        self,
        faces: dict[str, torch.Tensor],
        erp_h: int,
        erp_w: int,
    ) -> dict[str, dict]:
        """6개 face 전부 처리, 결과 dict 반환."""
        return {
            name: self.segment_face(face_img, name, erp_h, erp_w)
            for name, face_img in faces.items()
        }

    # ── 배치 처리 ─────────────────────────────────────────────────────────

    def segment_batch(
        self,
        faces_list: list[dict[str, torch.Tensor]],
        erp_sizes: list[tuple[int, int]],
    ) -> list[dict[str, dict]]:
        """
        여러 이미지 배치 처리. GPU OOM 시 배치를 절반으로 자동 축소.

        Args:
            faces_list: 각 이미지의 face dict 리스트.
            erp_sizes : 각 이미지의 (erp_h, erp_w) 리스트.
        """
        results: list[dict[str, dict]] = []
        i = 0
        while i < len(faces_list):
            try:
                r = self.segment_all_faces(faces_list[i], erp_sizes[i][0], erp_sizes[i][1])
                results.append(r)
                i += 1
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"GPU OOM — 이미지 {i} 처리 중 OOM, 재시도 (캐시 비우기)")
                torch.cuda.empty_cache()
                # 재시도 (배치 단위가 아닌 이미지 단위이므로 그냥 재시도)
                r = self.segment_all_faces(faces_list[i], erp_sizes[i][0], erp_sizes[i][1])
                results.append(r)
                i += 1
        return results

    # ── 시각화 ───────────────────────────────────────────────────────────

    def visualize_classification(
        self,
        face_img: torch.Tensor,
        result: dict,
    ) -> np.ndarray:
        """
        역할별 오버레이 시각화.

        PHOTOGRAPHER → 파란색, BACKGROUND → 빨간색.
        각 인물 위에 역할 + score 텍스트 표시.

        Returns:
            HxWx3 uint8 BGR ndarray.
        """
        img = self._tensor_to_bgr(face_img).copy()

        photo_mask = result["photographer_mask"]
        if photo_mask.any():
            _overlay_mask(img, photo_mask.numpy(), color=(255, 60, 60), alpha=0.4)

        for mask in result["background_masks"]:
            _overlay_mask(img, mask.numpy(), color=(60, 60, 255), alpha=0.35)

        for det, role, score in zip(
            result["detections"], result["roles"], result["role_scores"]
        ):
            x1, y1, x2, y2 = det.box.astype(int)
            cx, cy = (x1 + x2) // 2, y1 - 10
            label = f"{'PHOTO' if role == PersonRole.PHOTOGRAPHER else 'BG'} {score:.2f}"
            color = (255, 80, 0) if role == PersonRole.PHOTOGRAPHER else (0, 80, 255)
            cv2.putText(img, label, (cx - 30, max(cy, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        return img

    # ── 내부: YOLO 검출 ───────────────────────────────────────────────────

    def _yolo_detect(self, img_bgr: np.ndarray, fh: int, fw: int) -> list[_Detection]:
        results = self.yolo.predict(
            source=img_bgr,
            classes=[self.person_class_id],
            conf=self.yolo_conf,
            verbose=False,
            device=self.device,
        )
        detections: list[_Detection] = []
        for result in results:
            if result.masks is None:
                continue
            for seg_mask, box, conf in zip(
                result.masks.data, result.boxes.xyxy, result.boxes.conf
            ):
                mask_np = cv2.resize(
                    seg_mask.cpu().numpy().astype(np.float32),
                    (fw, fh),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_bool = torch.from_numpy(mask_np > 0.5)
                box_np = box.cpu().numpy().astype(np.float32)
                x1, y1, x2, y2 = box_np
                detections.append(_Detection(
                    box=box_np,
                    mask=mask_bool,
                    conf=float(conf),
                    pixel_count=int(mask_bool.sum()),
                    bbox_area=float((x2 - x1) * (y2 - y1)),
                ))
        return detections

    # ── 내부: SAM2 정밀화 ─────────────────────────────────────────────────

    def _sam2_refine(
        self, img_bgr: np.ndarray, detections: list[_Detection], fh: int, fw: int
    ) -> list[_Detection]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.sam2.set_image(rgb)
        refined: list[_Detection] = []
        for det in detections:
            try:
                pred_masks, _, _ = self.sam2.predict(
                    box=det.box[None],
                    multimask_output=False,
                )
                mask_bool = torch.from_numpy(pred_masks[0].astype(bool))
                refined.append(_Detection(
                    box=det.box,
                    mask=mask_bool,
                    conf=det.conf,
                    pixel_count=int(mask_bool.sum()),
                    bbox_area=det.bbox_area,
                ))
            except Exception as e:
                logger.warning(f"SAM2 실패 ({e}) — YOLO mask 유지")
                refined.append(det)
        return refined

    # ── 내부: 마스크 후처리 ───────────────────────────────────────────────

    @staticmethod
    def _dilate_and_fill(mask: torch.Tensor, dilate_px: int) -> torch.Tensor:
        """dilate + fill_holes."""
        arr = mask.numpy().astype(np.uint8)
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_px * 2 + 1, dilate_px * 2 + 1),
            )
            arr = cv2.dilate(arr, kernel, iterations=1)
        filled = ndi.binary_fill_holes(arr).astype(np.uint8)
        return torch.from_numpy(filled.astype(bool))

    @staticmethod
    def _postprocess_photographer_mask(
        raw_mask: torch.Tensor,
        face_name: str,
        photographer_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        PHOTOGRAPHER raw mask 후처리.

        down face: dilate로 조각 연결 → convex hull → YOLO bbox OR → 2차 hull → 여유 dilate
        나머지   : dilate(15px) + fill_holes
        """
        mask = raw_mask.cpu().numpy().astype(np.uint8) * 255

        if face_name == "down":
            H, W = mask.shape

            # 1. 연결용 dilate
            k_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
            mask = cv2.dilate(mask, k_connect)

            # 2. 1차 convex hull
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                hull = cv2.convexHull(all_points)
                cv2.fillPoly(mask, [hull], 255)

            # 3. YOLO bbox 전체를 OR 합산 (카메라 장비 등 mask 누락 보완)
            if photographer_bbox is not None:
                x1, y1, x2, y2 = photographer_bbox
                x1 = max(0, x1 - 20)
                y1 = max(0, y1 - 20)
                x2 = min(W, x2 + 20)
                y2 = min(H, y2 + 20)
                mask[y1:y2, x1:x2] = 255

                # bbox 포함 후 2차 convex hull
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.concatenate(contours)
                    hull = cv2.convexHull(all_points)
                    cv2.fillPoly(mask, [hull], 255)

            # 4. 최종 여유 dilate
            k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.dilate(mask, k_final)

        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # dilate_px=15
            mask = cv2.dilate(mask, k)
            mask = (ndi.binary_fill_holes(mask > 0).astype(np.uint8)) * 255

        return torch.from_numpy(mask > 0).to(raw_mask.device)

    @staticmethod
    def _down_face_ellipse_mask(
        hull_mask: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        down face 전용: 고정 타원 OR hull.

        타원 중심·크기 고정 (YOLO 결과와 무관):
          center = (W*0.45, H*0.45)
          axes   = (W*0.38, H*0.38)
        hull_mask가 비어있으면 타원만 반환.
        """
        hull_np = hull_mask.cpu().numpy().astype(np.uint8) * 255

        ellipse = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(
            ellipse,
            center=(int(W * 0.45), int(H * 0.45)),
            axes=(int(W * 0.38), int(H * 0.38)),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1,
        )

        final = cv2.bitwise_or(hull_np, ellipse)
        return torch.from_numpy(final > 0).to(hull_mask.device)

    @staticmethod
    def _face_region_mask(box: np.ndarray, fh: int, fw: int) -> torch.Tensor:
        """bbox 상단 30%를 얼굴 영역으로 추정한 mask."""
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        face_h = max(1, int((y2 - y1) * 0.30))
        mask = torch.zeros(fh, fw, dtype=torch.bool)
        mask[y1 : y1 + face_h, x1:x2] = True
        return mask

    # ── 내부: 역할 판별 헬퍼 ─────────────────────────────────────────────

    @staticmethod
    def _biggest_mask_index(detections: list[_Detection]) -> int:
        """pixel_count 최대 index (동률 시 bbox_area 큰 쪽)."""
        best_i, best_px, best_area = 0, -1, -1.0
        for i, det in enumerate(detections):
            if det.pixel_count > best_px or (
                det.pixel_count == best_px and det.bbox_area > best_area
            ):
                best_i, best_px, best_area = i, det.pixel_count, det.bbox_area
        return best_i

    @staticmethod
    def _scores_to_roles(
        scores: list[float], detections: list[_Detection]
    ) -> list[tuple[PersonRole, float]]:
        """score → role 변환, PHOTOGRAPHER 최대 1명 강제."""
        roles = []
        for s in scores:
            if s >= 0.6:
                roles.append((PersonRole.PHOTOGRAPHER, s))
            else:
                roles.append((PersonRole.BACKGROUND, s))
        return PersonSegmenter._enforce_single_photographer(roles, detections)

    @staticmethod
    def _enforce_single_photographer(
        roles: list[tuple[PersonRole, float]],
        detections: list[_Detection],
    ) -> list[tuple[PersonRole, float]]:
        """PHOTOGRAPHER가 복수이면 score 가장 높은 1명만 유지."""
        photo_indices = [i for i, (r, _) in enumerate(roles) if r == PersonRole.PHOTOGRAPHER]
        if len(photo_indices) <= 1:
            return roles

        # score 동률이면 pixel_count 큰 쪽
        best = max(
            photo_indices,
            key=lambda i: (roles[i][1], detections[i].pixel_count),
        )
        result = list(roles)
        for i in photo_indices:
            if i != best:
                result[i] = (PersonRole.BACKGROUND, roles[i][1])
        return result

    # ── 내부: 이미지 변환 ─────────────────────────────────────────────────

    @staticmethod
    def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
        """(3,H,W) float32 [0,1] → HxWx3 uint8 BGR."""
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════
# FaceMosaicker
# ═══════════════════════════════════════════════════════════════════════════

class FaceMosaicker:
    """배경 인물 얼굴 영역 모자이크 처리기."""

    def __init__(
        self,
        mosaic_block_size: int = 20,
        feather_px: int = 8,
    ) -> None:
        self.block_size = mosaic_block_size
        self.feather_px = feather_px

    def mosaic_face(
        self,
        face_img: torch.Tensor,
        face_bbox_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        face_bbox_mask 영역을 블록 평균 모자이크 후 경계 feathering.

        Args:
            face_img      : (3,H,W) float32 tensor.
            face_bbox_mask: (H,W) bool tensor.

        Returns:
            (3,H,W) float32 tensor.
        """
        _, h, w = face_img.shape
        img_np = face_img.permute(1, 2, 0).cpu().numpy().copy()  # HxWx3

        # 마스크 bbox 범위만 처리
        ys, xs = torch.where(face_bbox_mask)
        if len(ys) == 0:
            return face_img

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        roi = img_np[y1:y2, x1:x2].copy()
        bs = self.block_size
        rh, rw = roi.shape[:2]

        # 블록 평균
        for by in range(0, rh, bs):
            for bx in range(0, rw, bs):
                block = roi[by : by + bs, bx : bx + bs]
                roi[by : by + bs, bx : bx + bs] = block.mean(axis=(0, 1), keepdims=True)

        # mask 영역에만 적용
        mosaic_full = img_np.copy()
        mosaic_full[y1:y2, x1:x2] = roi

        # Gaussian feathering
        alpha = _gaussian_feather(face_bbox_mask.numpy(), self.feather_px)  # HxWx1
        blended = img_np * (1 - alpha) + mosaic_full * alpha
        return torch.from_numpy(blended).permute(2, 0, 1).float()

    def apply_background_mosaics(
        self,
        face_img: torch.Tensor,
        seg_result: dict,
    ) -> torch.Tensor:
        """seg_result의 background_face_masks 전부에 모자이크 적용."""
        result = face_img
        for mask in seg_result.get("background_face_masks", []):
            result = self.mosaic_face(result, mask)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# MaskPostProcessor
# ═══════════════════════════════════════════════════════════════════════════

class MaskPostProcessor:
    """CubeMap face mask → ERP 좌표계 역투영 통합."""

    def expand_mask_to_erp(
        self,
        face_masks: dict[str, torch.Tensor],
        uv_maps: dict[str, torch.Tensor],
        erp_h: int,
        erp_w: int,
    ) -> torch.Tensor:
        """
        PHOTOGRAPHER mask(face 좌표계)를 ERP 좌표계로 역투영해 통합.

        Args:
            face_masks: {"front": (H,W) bool, ...} — PHOTOGRAPHER mask만 전달.
            uv_maps   : CubeMapConverter.get_face_uv_map() 결과.
                        {"front": (2,H,W) float32 [0,1], ...}
            erp_h, erp_w: 출력 ERP 해상도.

        Returns:
            (erp_h, erp_w) bool tensor — ERP 상의 PHOTOGRAPHER 영역.
        """
        erp_mask = torch.zeros(erp_h, erp_w, dtype=torch.bool)

        for face_name, fmask in face_masks.items():
            if not fmask.any():
                continue
            uv = uv_maps.get(face_name)
            if uv is None:
                logger.warning(f"UV 맵 없음: {face_name} — skip")
                continue

            # uv: (2,H,W)  →  u=(0), v=(1)
            u = uv[0]  # (H,W) [0,1]
            v = uv[1]  # (H,W) [0,1]

            # face mask True 위치의 ERP 픽셀 좌표 계산
            ys_face, xs_face = torch.where(fmask)
            u_vals = u[ys_face, xs_face]
            v_vals = v[ys_face, xs_face]

            ex = (u_vals * (erp_w - 1)).long().clamp(0, erp_w - 1)
            ey = (v_vals * (erp_h - 1)).long().clamp(0, erp_h - 1)

            erp_mask[ey, ex] = True

        # 역투영 시 생기는 구멍 메우기 (소규모 dilate)
        erp_np = erp_mask.numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erp_np = cv2.dilate(erp_np, kernel, iterations=2)
        return torch.from_numpy(erp_np.astype(bool))


# ═══════════════════════════════════════════════════════════════════════════
# 내부 유틸
# ═══════════════════════════════════════════════════════════════════════════

def _overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """img 위에 mask 영역을 color로 반투명 오버레이 (in-place)."""
    overlay = img.copy()
    overlay[mask.astype(bool)] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _gaussian_feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """mask를 Gaussian blur해 HxWx1 float32 alpha 맵 반환."""
    if radius <= 0:
        return mask.astype(np.float32)[..., np.newaxis]
    ksize = radius * 2 + 1
    alpha = cv2.GaussianBlur(
        mask.astype(np.float32), (ksize, ksize), radius
    )
    return np.clip(alpha, 0, 1)[..., np.newaxis]
