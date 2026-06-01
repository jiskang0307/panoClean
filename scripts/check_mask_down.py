"""
check_mask_down.py — down face PHOTOGRAPHER mask 경계 확인.

실행:
    python scripts/check_mask_down.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEBUG_OUT = ROOT / "debug_output"
DEBUG_OUT.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE = 512

# ── 헬퍼 ──────────────────────────────────────────────────────────────────

def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def draw_contour(bgr: np.ndarray, mask_np: np.ndarray,
                 color=(0, 255, 0), thick=2) -> np.ndarray:
    vis = bgr.copy()
    m = mask_np.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis

def dilate_mask(mask_np: np.ndarray, px: int) -> np.ndarray:
    from scipy import ndimage as ndi
    arr = mask_np.astype(np.uint8)
    if px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1)
        )
        arr = cv2.dilate(arr, kernel, iterations=1)
    filled = ndi.binary_fill_holes(arr).astype(np.uint8)
    return filled.astype(bool)

# ── 모듈 로드 ─────────────────────────────────────────────────────────────

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter, PersonRole

img_paths = sorted((ROOT / "img").glob("*.jpg"))
if not img_paths:
    print("img/ 폴더에 이미지 없음")
    sys.exit(1)

cfg = {
    "device": DEVICE,
    "yolo_model":   "yolo11x-seg.pt",
    "yolo_conf":    0.4,
    "mask_dilate_px": 15,
}

conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)

erp0 = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
faces = conv.erp_to_cubemap(erp0)

face_t  = faces["down"]
img_bgr = t2bgr(face_t)
fh = fw = FACE_SIZE

# ── YOLO 검출 ─────────────────────────────────────────────────────────────
dets = segmenter._yolo_detect(img_bgr, fh, fw)
print(f"[down] YOLO 검출: {len(dets)}명")

# ── 역할 분류 ─────────────────────────────────────────────────────────────
role_scores = segmenter.classify_persons(dets, "down", erp_h, erp_w, fh)

# ── 현재 dilate 커널 크기 출력 ─────────────────────────────────────────────
DILATE_CURRENT = cfg["mask_dilate_px"]
DILATE_NEW     = 40
kernel_current = DILATE_CURRENT * 2 + 1
kernel_new     = DILATE_NEW     * 2 + 1
print(f"\n현재 mask_dilate_px : {DILATE_CURRENT}  (커널 {kernel_current}×{kernel_current})")
print(f"비교 mask_dilate_px : {DILATE_NEW}  (커널 {kernel_new}×{kernel_new})")

# ── PHOTOGRAPHER detection 찾기 ───────────────────────────────────────────
photo_dets = [
    (det, score)
    for det, (role, score) in zip(dets, role_scores)
    if role == PersonRole.PHOTOGRAPHER
]

if not photo_dets:
    print("\nPHOTOGRAPHER 검출 없음 — down face에 사람이 없거나 인식 실패")
    sys.exit(0)

# PHOTOGRAPHER가 여러 명이면 mask를 합산
raw_union     = np.zeros((fh, fw), dtype=bool)
d15_union     = np.zeros((fh, fw), dtype=bool)
d40_union     = np.zeros((fh, fw), dtype=bool)

for det, score in photo_dets:
    raw_np  = det.mask.numpy().astype(bool)
    raw_union  |= raw_np
    d15_union  |= dilate_mask(raw_np, DILATE_CURRENT)
    d40_union  |= dilate_mask(raw_np, DILATE_NEW)

    x1, y1, x2, y2 = det.box.astype(int)
    print(f"\n  PHOTOGRAPHER  bbox=({x1},{y1},{x2-x1},{y2-y1})"
          f"  mask_px(raw)={int(raw_np.sum())}"
          f"  score={score:.2f}")

print(f"\n  mask_px  raw={int(raw_union.sum())}"
      f"  dilate{DILATE_CURRENT}={int(d15_union.sum())}"
      f"  dilate{DILATE_NEW}={int(d40_union.sum())}")

# ── 시각화 저장 ───────────────────────────────────────────────────────────

# 반투명 오버레이 + 경계선
def overlay_and_contour(bgr, mask_np, fill_color, contour_color=(0, 255, 0)):
    vis = bgr.copy()
    overlay = vis.copy()
    overlay[mask_np.astype(bool)] = fill_color
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    m = mask_np.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, contour_color, 2)
    return vis

# 1) 현재 mask (dilate 15)
vis_15 = overlay_and_contour(img_bgr, d15_union, (255, 60, 60), (0, 255, 0))
cv2.putText(vis_15, f"dilate={DILATE_CURRENT}px  px={int(d15_union.sum())}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite(str(DEBUG_OUT / "mask_contour_down.jpg"), vis_15)
print(f"\n저장: debug_output/mask_contour_down.jpg")

# 2) 비교 mask (dilate 40)
vis_40 = overlay_and_contour(img_bgr, d40_union, (60, 60, 255), (0, 200, 255))
cv2.putText(vis_40, f"dilate={DILATE_NEW}px  px={int(d40_union.sum())}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
cv2.imwrite(str(DEBUG_OUT / "mask_contour_down_dilate40.jpg"), vis_40)
print(f"저장: debug_output/mask_contour_down_dilate40.jpg")

# 3) 나란히 비교 (raw / dilate15 / dilate40)
vis_raw = overlay_and_contour(img_bgr, raw_union, (80, 80, 80), (255, 255, 0))
cv2.putText(vis_raw, f"raw  px={int(raw_union.sum())}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

comparison = np.hstack([vis_raw, vis_15, vis_40])
cv2.imwrite(str(DEBUG_OUT / "mask_comparison_down.jpg"), comparison)
print(f"저장: debug_output/mask_comparison_down.jpg  (raw | dilate{DILATE_CURRENT} | dilate{DILATE_NEW})")
