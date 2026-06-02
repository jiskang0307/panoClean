"""
debug_blur_down_v2.py — down face double blur (kernel=251×2, feather=101) 확인.

저장:
  debug_output/blur_down_v2.jpg       — before / after / diff 나란히
  debug_output/result_erp_nadir.jpg   — ERP 하단 25% 크롭

실행:
    python scripts/debug_blur_down_v2.py
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
    down_face_method="blur",
    down_blur_kernel=251,
    down_blur_feather=101,
)

target_path = img_paths[0]
erp         = load_erp(str(target_path))
_, erp_h, erp_w = erp.shape
faces       = conv.erp_to_cubemap(erp)

seg_down   = seg.segment_face(faces["down"], "down", erp_h, erp_w)
photo_mask = seg_down["photographer_mask"]

print(f"target  : {target_path.name}")
print(f"mask_px={int(photo_mask.sum())}  ratio={photo_mask.float().mean():.3f}")

result_down, _ = inpainter.inpaint_residual(
    faces["down"], photo_mask, filled_mask=False, face_name="down"
)

# ── 시각화 ───────────────────────────────────────────────────────────────
def t2bgr(t):
    return cv2.cvtColor(
        (t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)

def draw_contour(bgr, mask_t, color=(0,255,255), thick=2):
    vis  = bgr.copy()
    cnts, _ = cv2.findContours(
        mask_t.cpu().numpy().astype(np.uint8)*255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis

before = draw_contour(t2bgr(faces["down"]), photo_mask)
after  = draw_contour(t2bgr(result_down),   photo_mask)
diff   = cv2.applyColorMap(
    np.clip(np.abs(before.astype(np.int16)-after.astype(np.int16)).astype(np.uint8)*3, 0, 255),
    cv2.COLORMAP_JET)
diff   = draw_contour(diff, photo_mask)

for img, lbl in [(before, "before"), (after, "blur x2 k=251 f=101"), (diff, "diff x3")]:
    cv2.putText(img, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

out1 = DEBUG_OUT / "blur_down_v2.jpg"
cv2.imwrite(str(out1), np.hstack([before, after, diff]))
print(f"저장: {out1}  ({out1.stat().st_size//1024} KB)")

# ── ERP 재합성 + nadir 크롭 ───────────────────────────────────────────────
result_faces        = dict(faces)
result_faces["down"] = result_down
result_erp          = conv.cubemap_to_erp(result_faces, erp_h, erp_w)

erp_out = DEBUG_OUT / "result_erp.jpg"
save_erp(result_erp, str(erp_out))

nadir_path = DEBUG_OUT / "result_erp_nadir.jpg"
erp_bgr    = cv2.imread(str(erp_out))
cv2.imwrite(str(nadir_path), erp_bgr[int(erp_h * 0.75):, :])
print(f"저장: {nadir_path}  ({nadir_path.stat().st_size//1024} KB)")
