"""
debug_mask_ellipse.py — down face 타원 mask 확인.

실행:
    python scripts/debug_mask_ellipse.py
    python scripts/debug_mask_ellipse.py --axes 0.55 0.65   # 기본값
    python scripts/debug_mask_ellipse.py --axes 0.60 0.70   # 더 크게

저장:
    debug_output/mask_ellipse_down.jpg       — hull / 타원 / 최종(OR) 3단 비교
    debug_output/mask_ellipse_down_seg.jpg   — segment_face() 실제 출력
"""

from __future__ import annotations

import argparse
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

# ── CLI ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--axes", nargs=2, type=float, default=[0.55, 0.65],
                    metavar=("RX", "RY"),
                    help="타원 반축 비율 (단축/W, 장축/H), 기본 0.55 0.65")
cli = parser.parse_args()
AX_W, AX_H = cli.axes[0], cli.axes[1]

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

def label(img: np.ndarray, *lines: str) -> np.ndarray:
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 26 + i * 26),
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
H = W           = FACE_SIZE

# ── YOLO + 분류 + hull mask 생성 ─────────────────────────────────────────

dets        = segmenter._yolo_detect(img_bgr, H, W)
role_scores = segmenter.classify_persons(dets, "down", erp_h, erp_w, H)

photo_dets  = [(d, s) for d, (r, s) in zip(dets, role_scores)
               if r == PersonRole.PHOTOGRAPHER]
print(f"[down] 검출: {len(dets)}명  PHOTOGRAPHER: {len(photo_dets)}명")

# hull mask (_postprocess_photographer_mask 결과)
if photo_dets:
    raw_union = photo_dets[0][0].mask.clone()
    boxes     = [photo_dets[0][0].box]
    for d, _ in photo_dets[1:]:
        raw_union = raw_union | d.mask
        boxes.append(d.box)
    boxes_np  = np.stack(boxes)
    bbox      = (int(boxes_np[:, 0].min()), int(boxes_np[:, 1].min()),
                 int(boxes_np[:, 2].max()), int(boxes_np[:, 3].max()))
    hull_mask = segmenter._postprocess_photographer_mask(raw_union, "down", bbox)
    print(f"  YOLO bbox          : {bbox}")
else:
    hull_mask = torch.zeros(H, W, dtype=torch.bool)
    bbox      = None
    print("  PHOTOGRAPHER 미검출 → 중앙 고정")

hull_np = hull_mask.cpu().numpy().astype(np.uint8) * 255

# ── 타원 중심점 결정 ───────────────────────────────────────────────────────

if hull_np.any():
    cnts, _ = cv2.findContours(hull_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(np.concatenate(cnts))
        cx, cy = x + w // 2, y + h // 2
    else:
        cx, cy = W // 2, H // 2
else:
    cx, cy = W // 2, H // 2

print(f"  타원 중심점        : ({cx}, {cy})")
print(f"  타원 반축          : axes_w={int(W*AX_W)}px ({AX_W:.2f}×W), "
      f"axes_h={int(H*AX_H)}px ({AX_H:.2f}×H)")

# ── 타원 mask 생성 ─────────────────────────────────────────────────────────

ellipse_np = np.zeros((H, W), dtype=np.uint8)
cv2.ellipse(ellipse_np,
            center=(cx, cy),
            axes=(int(W * AX_W), int(H * AX_H)),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1)

final_np = cv2.bitwise_or(hull_np, ellipse_np)

print(f"  hull mask_px       : {int((hull_np > 0).sum())}")
print(f"  ellipse mask_px    : {int((ellipse_np > 0).sum())}")
print(f"  final mask_px      : {int((final_np > 0).sum())}")

# ── 시각화 ───────────────────────────────────────────────────────────────

vis_hull    = overlay(img_bgr, hull_np,    fill=(80, 80, 255))
vis_ellipse = overlay(img_bgr, ellipse_np, fill=(80, 200, 80))
vis_final   = overlay(img_bgr, final_np,   fill=(80, 80, 255))

# hull 패널: bbox 표시
if bbox:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis_hull, (x1, y1), (x2, y2), (0, 200, 255), 2)

# 타원 패널: 중심점 + 축 표시
cv2.drawMarker(vis_ellipse, (cx, cy), (255, 0, 0),
               cv2.MARKER_CROSS, 20, 2)

# 최종 패널: 타원 경계도 추가
cnts_e, _ = cv2.findContours(ellipse_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis_final, cnts_e, -1, (0, 200, 80), 1)

label(vis_hull,
      f"[1] hull (YOLO+convex)",
      f"px={int((hull_np>0).sum())}")
label(vis_ellipse,
      f"[2] ellipse  axes=({AX_W:.2f}W, {AX_H:.2f}H)",
      f"center=({cx},{cy})  px={int((ellipse_np>0).sum())}")
label(vis_final,
      f"[3] hull OR ellipse  (final)",
      f"px={int((final_np>0).sum())}")

# 3단 가로 배치
row = np.hstack([vis_hull, vis_ellipse, vis_final])
out1 = DEBUG_OUT / "mask_ellipse_down.jpg"
cv2.imwrite(str(out1), row)
print(f"\n저장: {out1.name}  (좌=hull | 중=타원 | 우=최종OR)")

# ── segment_face() 실제 출력 ──────────────────────────────────────────────

# axes_ratio를 기본값으로 테스트
seg     = segmenter.segment_face(face_t, "down", erp_h, erp_w)
seg_np  = seg["photographer_mask"].cpu().numpy().astype(np.uint8) * 255
vis_seg = overlay(img_bgr, seg_np)
if bbox:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis_seg, (x1, y1), (x2, y2), (0, 200, 255), 2)
label(vis_seg,
      f"segment_face() result",
      f"px={int((seg_np>0).sum())}  axes=(0.55W, 0.65H)")

out2 = DEBUG_OUT / "mask_ellipse_down_seg.jpg"
cv2.imwrite(str(out2), vis_seg)
print(f"저장: {out2.name}  (파이프라인 실제 출력, 청록=YOLO bbox)")
print(f"\n  타원 크기 조정이 필요하면:")
print(f"  python scripts/debug_mask_ellipse.py --axes 0.60 0.70  # 더 크게")
print(f"  python scripts/debug_mask_ellipse.py --axes 0.50 0.60  # 더 작게")
