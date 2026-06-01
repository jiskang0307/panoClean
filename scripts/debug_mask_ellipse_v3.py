"""
debug_mask_ellipse_v3.py — down face 고정 타원 mask 최종 확인.

실행:
    python scripts/debug_mask_ellipse_v3.py

저장:
    debug_output/mask_ellipse_down_v3.jpg
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

AX_RATIO  = 0.35

def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def overlay(bgr, mask_u8, fill=(80, 80, 255), alpha=0.40):
    vis = bgr.copy()
    ov  = vis.copy()
    ov[mask_u8 > 0] = fill
    cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0, vis)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis

def label(img, *lines):
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 26 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return img

# ── 로드 ──────────────────────────────────────────────────────────────────

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter

img_paths = sorted((ROOT / "img").glob("*.jpg"))
if not img_paths:
    print("img/ 폴더에 이미지 없음"); sys.exit(1)

cfg = {"device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4}
conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)

erp0            = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
face_t          = conv.erp_to_cubemap(erp0)["down"]
img_bgr         = t2bgr(face_t)
H = W           = FACE_SIZE

# ── 파라미터 ──────────────────────────────────────────────────────────────

cx, cy = W // 2, H // 2
ax_w   = int(W * AX_RATIO)
ax_h   = int(H * AX_RATIO)

print(f"center : ({cx}, {cy})")
print(f"axes   : ({ax_w}, {ax_h})  [{AX_RATIO:.2f}×W, {AX_RATIO:.2f}×H]")

# ── segment_face() 실제 출력 ─────────────────────────────────────────────

seg    = segmenter.segment_face(face_t, "down", erp_h, erp_w)
seg_np = seg["photographer_mask"].cpu().numpy().astype(np.uint8) * 255

print(f"mask_px: {int((seg_np > 0).sum())}")
print(f"roles  : {seg['roles']}  (YOLO 건너뜀)")

# ── 시각화 ───────────────────────────────────────────────────────────────

vis = overlay(img_bgr, seg_np)
cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)

label(vis,
      f"down face  ellipse only (no YOLO)",
      f"center=({cx},{cy})  axes=({ax_w},{ax_h})  ratio={AX_RATIO}",
      f"mask_px={int((seg_np>0).sum())}")

out = DEBUG_OUT / "mask_ellipse_down_v3.jpg"
cv2.imwrite(str(out), vis)
print(f"\n저장: {out.name}")
