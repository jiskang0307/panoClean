"""
debug_mask_fixed.py — down face PHOTOGRAPHER mask 수정 결과 확인.

실행:
    python scripts/debug_mask_fixed.py

저장 파일:
    debug_output/mask_fixed_down_v2.jpg  — 4단계 그리드 (hull전 / 1차hull / bbox+hull / 최종)
    debug_output/mask_fixed_down_seg.jpg — segment_face() 실제 출력
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

def overlay(bgr: np.ndarray, mask_u8: np.ndarray,
            fill=(80, 80, 255), alpha=0.40) -> np.ndarray:
    vis = bgr.copy()
    ov  = vis.copy()
    ov[mask_u8 > 0] = fill
    cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0, vis)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis

def label(img: np.ndarray, text: str, y: int = 30) -> np.ndarray:
    cv2.putText(img, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return img

# ── 모델 로드 ──────────────────────────────────────────────────────────────

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter, PersonRole

img_paths = sorted((ROOT / "img").glob("*.jpg"))
if not img_paths:
    print("img/ 폴더에 이미지 없음"); sys.exit(1)

cfg = {"device": DEVICE, "yolo_model": "yolo11x-seg.pt",
       "yolo_conf": 0.4, "mask_dilate_px": 15}

conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)

erp0            = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
face_t          = conv.erp_to_cubemap(erp0)["down"]
img_bgr         = t2bgr(face_t)
fh = fw         = FACE_SIZE

# ── YOLO + 역할 분류 ───────────────────────────────────────────────────────

dets        = segmenter._yolo_detect(img_bgr, fh, fw)
role_scores = segmenter.classify_persons(dets, "down", erp_h, erp_w, fh)

photo_dets = [(d, s) for d, (r, s) in zip(dets, role_scores)
              if r == PersonRole.PHOTOGRAPHER]
print(f"[down] PHOTOGRAPHER: {len(photo_dets)}명 / 전체 {len(dets)}명")

if not photo_dets:
    print("PHOTOGRAPHER 없음 — 종료"); sys.exit(0)

# raw union mask + combined bbox
raw_union = photo_dets[0][0].mask.clone()
boxes     = [photo_dets[0][0].box]
for d, _ in photo_dets[1:]:
    raw_union = raw_union | d.mask
    boxes.append(d.box)

boxes_np      = np.stack(boxes)
combined_bbox = (int(boxes_np[:, 0].min()), int(boxes_np[:, 1].min()),
                 int(boxes_np[:, 2].max()), int(boxes_np[:, 3].max()))

raw_np = raw_union.cpu().numpy().astype(np.uint8) * 255
H, W   = raw_np.shape
print(f"  YOLO combined bbox : {combined_bbox}")
print(f"  raw mask_px        : {int((raw_np > 0).sum())}")

# ── 4단계 시각화 ───────────────────────────────────────────────────────────

# [1] dilate 60 (연결)
k_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
m1        = cv2.dilate(raw_np, k_connect)
print(f"  dilate60 mask_px   : {int((m1 > 0).sum())}")

# [2] 1차 convex hull
m2 = m1.copy()
cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    all_pts = np.concatenate(cnts)
    hull    = cv2.convexHull(all_pts)
    cv2.fillPoly(m2, [hull], 255)
print(f"  1차 hull mask_px   : {int((m2 > 0).sum())}")

# [3] bbox OR + 2차 hull
m3 = m2.copy()
x1b, y1b, x2b, y2b = combined_bbox
x1b = max(0, x1b - 20); y1b = max(0, y1b - 20)
x2b = min(W, x2b + 20); y2b = min(H, y2b + 20)
m3[y1b:y2b, x1b:x2b] = 255

# 2차 convex hull
cnts, _ = cv2.findContours(m3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    all_pts = np.concatenate(cnts)
    hull    = cv2.convexHull(all_pts)
    cv2.fillPoly(m3, [hull], 255)
print(f"  2차 hull mask_px   : {int((m3 > 0).sum())}")

# [4] 최종 dilate 20
k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
m4      = cv2.dilate(m3, k_final)
print(f"  최종 mask_px       : {int((m4 > 0).sum())}")

# ── bbox 표시용 helper ────────────────────────────────────────────────────

def draw_bbox(img: np.ndarray, bbox, color=(0, 200, 255), thick=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    return img

# 패널 생성
vis1 = label(overlay(img_bgr, m1),        f"[1] dilate60  px={int((m1>0).sum())}")
vis2 = label(overlay(img_bgr, m2),        f"[2] 1st hull  px={int((m2>0).sum())}")
vis3 = label(overlay(img_bgr, m3),        f"[3] +bbox hull  px={int((m3>0).sum())}")
vis4 = label(overlay(img_bgr, m4),        f"[4] +d20 (final)  px={int((m4>0).sum())}")

# [3] 패널에 bbox 경계 표시
draw_bbox(vis3, (x1b, y1b, x2b, y2b), color=(0, 200, 255))

# 2×2 그리드
grid = np.vstack([np.hstack([vis1, vis2]), np.hstack([vis3, vis4])])
out_v2 = DEBUG_OUT / "mask_fixed_down_v2.jpg"
cv2.imwrite(str(out_v2), grid)
print(f"\n저장: {out_v2.name}  (2×2: dilate60 | 1차hull | bbox+2차hull | 최종)")

# ── segment_face() 실제 출력 ──────────────────────────────────────────────
seg     = segmenter.segment_face(face_t, "down", erp_h, erp_w)
final_m = seg["photographer_mask"].cpu().numpy().astype(np.uint8) * 255
vis_seg = overlay(img_bgr, final_m)
draw_bbox(vis_seg, combined_bbox, color=(0, 200, 255))
label(vis_seg, f"segment_face()  px={int((final_m>0).sum())}")
out_seg = DEBUG_OUT / "mask_fixed_down_seg.jpg"
cv2.imwrite(str(out_seg), vis_seg)
print(f"저장: {out_seg.name}  (청록 bbox=YOLO combined, 초록=mask 경계)")
print(f"  segment_face mask_px: {int((final_m>0).sum())}")
