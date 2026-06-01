"""
debug_mask_final_down.py — down face ellipse OR hull 최종 확인.

실행:
    python scripts/debug_mask_final_down.py

저장:
    debug_output/mask_final_down.jpg  — ellipse / hull / final OR 나란히
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

def t2bgr(t):
    arr = t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    return cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def overlay(bgr, mask_u8, fill=(80,80,255), alpha=0.40):
    vis = bgr.copy(); ov = vis.copy()
    ov[mask_u8>0] = fill
    cv2.addWeighted(ov, alpha, vis, 1-alpha, 0, vis)
    cnts,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0,255,0), 2)
    return vis

def label(img, *lines):
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 26+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,255,255), 2)
    return img

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter, PersonRole

img_paths = sorted((ROOT/"img").glob("*.jpg"))
if not img_paths: print("img/ 없음"); sys.exit(1)

cfg = {"device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4, "mask_dilate_px": 15}
conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)

erp0            = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
face_t          = conv.erp_to_cubemap(erp0)["down"]
img_bgr         = t2bgr(face_t)
H = W           = FACE_SIZE

# ── 고정 타원 ─────────────────────────────────────────────────────────────
cx, cy = int(W*0.45), int(H*0.45)
ax_w, ax_h = int(W*0.38), int(H*0.38)

ellipse_np = np.zeros((H, W), dtype=np.uint8)
cv2.ellipse(ellipse_np, (cx, cy), (ax_w, ax_h), 0, 0, 360, 255, -1)

# ── hull mask (YOLO + _postprocess) ──────────────────────────────────────
dets        = segmenter._yolo_detect(img_bgr, H, W)
role_scores = segmenter.classify_persons(dets, "down", erp_h, erp_w, H)
photo_dets  = [(d, s) for d, (r, s) in zip(dets, role_scores)
               if r == PersonRole.PHOTOGRAPHER]

print(f"[down] 검출: {len(dets)}명  PHOTOGRAPHER: {len(photo_dets)}명")

if photo_dets:
    raw_union = photo_dets[0][0].mask.clone()
    boxes     = [photo_dets[0][0].box]
    for d, _ in photo_dets[1:]:
        raw_union = raw_union | d.mask
        boxes.append(d.box)
    boxes_np  = np.stack(boxes)
    bbox      = (int(boxes_np[:,0].min()), int(boxes_np[:,1].min()),
                 int(boxes_np[:,2].max()), int(boxes_np[:,3].max()))
    hull_mask = segmenter._postprocess_photographer_mask(raw_union, "down", bbox)
    hull_np   = hull_mask.cpu().numpy().astype(np.uint8) * 255
    print(f"  YOLO bbox : {bbox}")
    print(f"  hull px   : {int((hull_np>0).sum())}")
else:
    hull_np = np.zeros((H, W), dtype=np.uint8)
    bbox    = None
    print("  PHOTOGRAPHER 미검출 → hull 없음")

# ── OR 합산 ───────────────────────────────────────────────────────────────
final_np = cv2.bitwise_or(ellipse_np, hull_np)

print(f"  ellipse px: {int((ellipse_np>0).sum())}")
print(f"  final px  : {int((final_np>0).sum())}")

# ── 패널 ─────────────────────────────────────────────────────────────────
vis_ell  = overlay(img_bgr, ellipse_np, fill=(80,200,80))
cv2.drawMarker(vis_ell, (cx,cy), (0,0,255), cv2.MARKER_CROSS, 24, 3)
label(vis_ell,
      f"[1] ellipse",
      f"center=({cx},{cy})  axes=({ax_w},{ax_h})",
      f"px={int((ellipse_np>0).sum())}")

vis_hull = overlay(img_bgr, hull_np)
if bbox:
    x1,y1,x2,y2 = bbox
    cv2.rectangle(vis_hull, (x1,y1),(x2,y2),(0,200,255),2)
label(vis_hull,
      f"[2] hull (YOLO+convex)",
      f"px={int((hull_np>0).sum())}")

vis_final = overlay(img_bgr, final_np)
cnts_e,_ = cv2.findContours(ellipse_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis_final, cnts_e, -1, (0,220,80), 1)
cv2.drawMarker(vis_final, (cx,cy), (0,0,255), cv2.MARKER_CROSS, 20, 2)
label(vis_final,
      f"[3] ellipse OR hull  (final)",
      f"px={int((final_np>0).sum())}")

row = np.hstack([vis_ell, vis_hull, vis_final])
out = DEBUG_OUT / "mask_final_down.jpg"
cv2.imwrite(str(out), row)
print(f"\n저장: {out.name}  (좌=타원 | 중=hull | 우=최종OR)")
