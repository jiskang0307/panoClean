"""
debug_down_check.py — 1779263615 down face 검출 및 처리 확인.
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

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter
from pipeline.inpainting import LamaInpainter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt",
    "yolo_conf": 0.4, "mask_dilate_px": 15,
}

conv = CubeMapConverter(face_size=1024, device=DEVICE)
seg  = PersonSegmenter(cfg)
inp  = LamaInpainter(device=DEVICE,
                     down_blur_kernel=251, down_blur_feather=101, down_blur_passes=2)

p = ROOT / "input" / "1779263615.694478.jpg"
erp_t = load_erp(str(p))
_, eh, ew = erp_t.shape
faces = conv.erp_to_cubemap(erp_t)
seg_results = seg.segment_all_faces(faces, eh, ew)

# 5. down 검출 결과
down_res = seg_results["down"]
n_det   = len(down_res["detections"])
n_photo = sum(1 for r in down_res["roles"] if r.value == "photographer")
n_bg    = sum(1 for r in down_res["roles"] if r.value == "background")
print(f"[down] 검출: {n_det}명  PHOTOGRAPHER:{n_photo}  BACKGROUND:{n_bg}")
for i, (role, det) in enumerate(zip(down_res["roles"], down_res["detections"])):
    x1, y1, x2, y2 = det.box
    print(f"  det[{i}] {role.value}  bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  conf={det.conf:.3f}")
if n_det == 0:
    print("  [down] 검출 없음")

# 1. mask px
mask = down_res["photographer_mask"]
print(f"[DEBUG] down face photographer_mask px: {int(mask.sum())}  ratio={mask.float().mean():.4f}")


def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


down_bgr = t2bgr(faces["down"])
mask_np  = mask.cpu().numpy().astype(np.uint8) * 255
H, W = down_bgr.shape[:2]

# 2. 원본
cv2.imwrite(str(DEBUG_OUT / "face_down_original.jpg"), down_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95])
print("저장: face_down_original.jpg")

# 3. mask 시각화
overlay = down_bgr.copy()
overlay[mask_np > 0] = (
    overlay[mask_np > 0].astype("float32") * 0.3
    + np.array([0, 180, 0]) * 0.7
).clip(0, 255).astype(np.uint8)
cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)
cv2.putText(overlay, f"mask_px={int(mask.sum())} ({mask.float().mean()*100:.1f}%)",
            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
for det in down_res["detections"]:
    x1, y1, x2, y2 = [int(v) for v in det.box]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(overlay, f"{det.conf:.2f}", (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
cv2.imwrite(str(DEBUG_OUT / "face_down_mask.jpg"), overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 95])
print("저장: face_down_mask.jpg")

# 4. blur 결과 or ellipse 위치 표시
if mask.any():
    result = inp.blur_face(faces["down"].to(DEVICE), mask.to(DEVICE), "down")
    result_bgr = t2bgr(result)
    diff = (result.cpu() - faces["down"]).abs().mean().item()
    cv2.imwrite(str(DEBUG_OUT / "face_down_result.jpg"), result_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"저장: face_down_result.jpg  diff={diff:.6f}")
else:
    print("[down] mask px=0 → blur 미적용")
    ellipse_vis = down_bgr.copy()
    cv2.ellipse(ellipse_vis,
                (int(W * 0.45), int(H * 0.45)),
                (int(W * 0.38), int(H * 0.38)),
                0, 0, 360, (0, 0, 255), 2)
    cv2.putText(ellipse_vis, "ellipse (YOLO 검출 없음)",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(DEBUG_OUT / "face_down_result.jpg"), ellipse_vis,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("저장: face_down_result.jpg (ellipse 위치 표시)")
