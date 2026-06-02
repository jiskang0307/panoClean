"""
debug_left_mask.py — left face mask 상태 및 블러 결과 확인.

저장:
  debug_output/seg_left_final.jpg       — left face mask contour (green)
  debug_output/cubemap_faces_result.jpg — 처리 후 6개 face 나란히

실행:
    python scripts/debug_left_mask.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEBUG_OUT = ROOT / "debug_output"
DEBUG_OUT.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE = 512

from pipeline.cubemap import CubeMapConverter, FACE_NAMES, load_erp
from pipeline.segmentation import PersonSegmenter
from pipeline.inpainting import LamaInpainter

cfg = {
    "device": DEVICE,
    "yolo_model": "yolo11x-seg.pt",
    "yolo_conf": 0.4,
    "mask_dilate_px": 15,
}

img_paths  = sorted((ROOT / "img").glob("*.jpg"))
conv       = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
seg        = PersonSegmenter(cfg)
inpainter  = LamaInpainter(
    device=DEVICE,
    down_blur_kernel=251, down_blur_feather=101, down_blur_passes=2,
)

target_path = img_paths[0]
erp         = load_erp(str(target_path))
_, erp_h, erp_w = erp.shape
faces       = conv.erp_to_cubemap(erp)

seg_results = seg.segment_all_faces(faces, erp_h, erp_w)

# ── 1. connected components 수 출력 ──────────────────────────────────────
left_mask = seg_results["left"]["photographer_mask"]
mask_np   = left_mask.cpu().numpy().astype(np.uint8) * 255

num_labels, labels = cv2.connectedComponents(mask_np)
num_pieces = num_labels - 1   # 0번 = background
print(f"[DEBUG] left mask pieces: {num_pieces}")
print(f"        mask_px={int(left_mask.sum())}  ratio={left_mask.float().mean():.4f}")

# ── 2. seg_left_final.jpg — contour + 각 조각 번호 표시 ──────────────────
def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

left_bgr = t2bgr(faces["left"]).copy()

# 조각별 색상 (녹색 계열, 조각이 여러 개면 다른 색)
COLORS = [(0, 255, 0), (0, 200, 100), (0, 150, 200), (0, 100, 255), (50, 255, 150)]

for piece_id in range(1, num_labels):
    piece_mask = (labels == piece_id).astype(np.uint8) * 255
    px_count   = int((labels == piece_id).sum())
    color      = COLORS[(piece_id - 1) % len(COLORS)]

    # contour
    cnts, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left_bgr, cnts, -1, color, 2)

    # 조각 번호 + 픽셀 수
    if cnts:
        M = cv2.moments(cnts[0])
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, ww, hh = cv2.boundingRect(cnts[0])
            cx, cy = x + ww // 2, y + hh // 2
        cv2.putText(left_bgr, f"#{piece_id} {px_count}px",
                    (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# 전체 마스크 반투명 오버레이
overlay       = left_bgr.copy()
overlay_mask  = mask_np > 0
overlay[overlay_mask] = (overlay[overlay_mask].astype(np.float32) * 0.5 +
                         np.array([0, 60, 0], dtype=np.float32)).clip(0, 255).astype(np.uint8)
left_vis = cv2.addWeighted(left_bgr, 0.7, overlay, 0.3, 0)

cv2.putText(left_vis, f"left  pieces={num_pieces}  px={int(left_mask.sum())}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

out1 = DEBUG_OUT / "seg_left_final.jpg"
cv2.imwrite(str(out1), left_vis)
print(f"저장: {out1}")

# ── 3. cubemap_faces_result.jpg — 처리 후 6개 face 나란히 ─────────────────
THUMB = 400   # 썸네일 크기

result_faces: dict[str, torch.Tensor] = {}
for face_name in FACE_NAMES:
    face_img  = faces[face_name]
    photo_mask = seg_results[face_name]["photographer_mask"].to(DEVICE)
    if photo_mask.any():
        face_img = inpainter.blur_face(face_img, photo_mask, face_name)
    result_faces[face_name] = face_img

panels = []
for face_name in FACE_NAMES:
    bgr   = t2bgr(result_faces[face_name])
    thumb = cv2.resize(bgr, (THUMB, THUMB))
    cv2.putText(thumb, face_name, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # mask contour 표시 (빨간 선)
    m = seg_results[face_name]["photographer_mask"].cpu().numpy().astype(np.uint8) * 255
    m_small = cv2.resize(m, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
    cnts2, _ = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thumb, cnts2, -1, (0, 0, 255), 1)

    panels.append(thumb)

# 3×2 배열 (front/right/back / left/up/down)
row1 = np.hstack(panels[:3])
row2 = np.hstack(panels[3:])
grid = np.vstack([row1, row2])

out2 = DEBUG_OUT / "cubemap_faces_result.jpg"
cv2.imwrite(str(out2), grid)
print(f"저장: {out2}")
