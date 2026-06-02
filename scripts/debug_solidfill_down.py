"""
debug_solidfill_down.py — down face solid fill 처리 확인.

저장:
  debug_output/solidfill_down_result.jpg  — before / solid fill / diff 나란히
  debug_output/result_erp_nadir.jpg       — ERP 하단 25% 크롭

실행:
    python scripts/debug_solidfill_down.py
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

from pipeline.cubemap import CubeMapConverter, load_erp, save_erp
from pipeline.segmentation import PersonSegmenter
from pipeline.inpainting import LamaInpainter

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4,
    "mask_dilate_px": 15,
}
img_paths = sorted((ROOT / "img").glob("*.jpg"))
conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
seg       = PersonSegmenter(cfg)
inpainter = LamaInpainter(
    device=DEVICE,
    debug_dir=DEBUG_OUT,
    down_face_method="solid",
)

# ── target 로드 ───────────────────────────────────────────────────────────
target_path = img_paths[0]
erp         = load_erp(str(target_path))
_, erp_h, erp_w = erp.shape
faces       = conv.erp_to_cubemap(erp)

# ── down face 처리 ────────────────────────────────────────────────────────
seg_down   = seg.segment_face(faces["down"], "down", erp_h, erp_w)
photo_mask = seg_down["photographer_mask"]

print(f"target  : {target_path.name}")
print(f"mask_px={int(photo_mask.sum())}  ratio={photo_mask.float().mean():.3f}")

result_down, did_process = inpainter.inpaint_residual(
    faces["down"], photo_mask, filled_mask=False, face_name="down"
)
print(f"solid fill 처리: {did_process}")

# 샘플링된 평균 색상 출력 (참고용)
img_np  = (faces["down"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
mask_np = photo_mask.cpu().numpy().astype(np.uint8)
kernel  = np.ones((21, 21), np.uint8)
ring    = (cv2.dilate(mask_np, kernel) - mask_np).astype(bool)
if ring.any():
    avg = img_np[ring].mean(axis=0)
    print(f"ring px={ring.sum()}  avg_color(RGB)=({avg[0]:.0f}, {avg[1]:.0f}, {avg[2]:.0f})")

# ── 시각화 ───────────────────────────────────────────────────────────────
def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def draw_contour(bgr, mask_t, color=(0, 255, 255), thick=2):
    vis  = bgr.copy()
    m_u8 = mask_t.cpu().numpy().astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis

before_bgr = draw_contour(t2bgr(faces["down"]), photo_mask)
after_bgr  = draw_contour(t2bgr(result_down),   photo_mask)
diff_np    = np.abs(before_bgr.astype(np.int16) - after_bgr.astype(np.int16)).astype(np.uint8)
diff_vis   = cv2.applyColorMap(np.clip(diff_np * 3, 0, 255), cv2.COLORMAP_JET)
diff_vis   = draw_contour(diff_vis, photo_mask)

for img, lbl in [(before_bgr, "before"), (after_bgr, "solid fill"), (diff_vis, "diff x3")]:
    cv2.putText(img, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

panel = np.hstack([before_bgr, after_bgr, diff_vis])
out1  = DEBUG_OUT / "solidfill_down_result.jpg"
cv2.imwrite(str(out1), panel)
print(f"저장: {out1}  ({out1.stat().st_size // 1024} KB)")

# ── ERP 재합성 + nadir 크롭 ───────────────────────────────────────────────
result_faces        = dict(faces)
result_faces["down"] = result_down
result_erp          = conv.cubemap_to_erp(result_faces, erp_h, erp_w)

erp_path = DEBUG_OUT / "result_erp.jpg"
save_erp(result_erp, str(erp_path))

erp_bgr    = cv2.imread(str(erp_path))
nadir      = erp_bgr[int(erp_h * 0.75):, :]
nadir_path = DEBUG_OUT / "result_erp_nadir.jpg"
cv2.imwrite(str(nadir_path), nadir)
print(f"저장: {nadir_path}  ({nadir_path.stat().st_size // 1024} KB)")
