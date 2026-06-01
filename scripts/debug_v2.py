"""
debug_v2.py — 버그 수정 후 확인 이미지 생성.

저장:
  debug_output/seg_left_v2.jpg       — 수정된 classify_persons 결과 확인
  debug_output/inpaint_before_after_down.jpg — down face inpainting (feather=0)
  debug_output/result_erp.jpg        — 최종 ERP
  debug_output/erp_nadir_crop_v2.jpg — ERP 하단 25% 크롭

실행:
    python scripts/debug_v2.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEBUG_OUT = ROOT / "debug_output"
DEBUG_OUT.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE = 512

from pipeline.cubemap import CubeMapConverter, load_erp, save_erp
from pipeline.segmentation import PersonSegmenter, PersonRole
from pipeline.inpainting import LamaInpainter
from pipeline.matching import BackgroundMatcher

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4,
    "mask_dilate_px": 15, "feature_matcher": "sift", "min_match_count": 10,
}

img_paths = sorted((ROOT / "img").glob("*.jpg"))
conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)
inpainter = LamaInpainter(device=DEVICE, debug_dir=DEBUG_OUT, down_feather_px=0)
matcher   = BackgroundMatcher(cfg)

erp0            = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
faces           = conv.erp_to_cubemap(erp0)


# ══════════════════════════════════════════════════════════════════════════
# 1. seg_left_v2.jpg — 수정된 threshold 확인
# ══════════════════════════════════════════════════════════════════════════

seg  = segmenter.segment_face(faces["left"], "left", erp_h, erp_w)
arr  = faces["left"].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
vis  = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# 임계선 표시 (face_size * 0.70)
thresh_y = int(FACE_SIZE * 0.70)
cv2.line(vis, (0, thresh_y), (FACE_SIZE, thresh_y), (0, 140, 255), 1)
cv2.putText(vis, f"threshold y={thresh_y} (70%)", (4, thresh_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

# BACKGROUND 마스크: 노란색
for bm in seg.get("background_face_masks", []):
    m = bm.cpu().numpy().astype(np.uint8) * 255
    ov = vis.copy(); ov[m > 0] = [0, 200, 255]
    cv2.addWeighted(ov, 0.3, vis, 0.7, 0, vis)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 200, 255), 2)

# PHOTOGRAPHER 마스크: 초록색
pm = seg["photographer_mask"]
if pm.any():
    m = pm.cpu().numpy().astype(np.uint8) * 255
    ov = vis.copy(); ov[m > 0] = [0, 200, 0]
    cv2.addWeighted(ov, 0.3, vis, 0.7, 0, vis)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 3)
    cv2.putText(vis, "PHOTOGRAPHER (green)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
else:
    cv2.putText(vis, "PHOTOGRAPHER: none", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

for det, role in zip(seg["detections"], seg["roles"]):
    x1, y1, x2, y2 = det.box.astype(int)
    c = (0, 255, 0) if role == PersonRole.PHOTOGRAPHER else (0, 200, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
    cv2.putText(vis, f"{role.value[:4]} y2={y2}", (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

cv2.imwrite(str(DEBUG_OUT / "seg_left_v2.jpg"), vis)
print(f"[1] seg_left_v2.jpg 저장")
for det, role, sc in zip(seg["detections"], seg["roles"], seg["role_scores"]):
    thresh_y_f = FACE_SIZE * 0.70
    print(f"    {role.value}: score={sc:.2f}  y2={det.box[3]:.0f}  "
          f"threshold={thresh_y_f:.0f}  cond1={det.box[3] >= thresh_y_f}  px={det.pixel_count}")


# ══════════════════════════════════════════════════════════════════════════
# 2. 전체 파이프라인 실행 (소스 3장 사용)
# ══════════════════════════════════════════════════════════════════════════

src_faces_list = []
for sp in img_paths[1:4]:
    src_faces_list.append(conv.erp_to_cubemap(load_erp(str(sp))))

result_faces: dict[str, torch.Tensor] = {}

for fname, face_t in faces.items():
    seg_f = segmenter.segment_face(face_t, fname, erp_h, erp_w)
    pm    = seg_f["photographer_mask"]

    if not pm.any():
        result_faces[fname] = face_t
        continue

    srcs = [sf[fname] for sf in src_faces_list]
    best = matcher.select_best_sources(face_t, srcs, pm, top_k=3, face_name=fname)
    restored = (
        matcher.blend_multiple_sources(face_t, pm, best, face_name=fname)
        if best else face_t
    )
    cov = best[0][2] if best else 0.0

    inpainted, did = inpainter.inpaint_residual(
        restored, pm, filled_mask=(cov > 0.3), face_name=fname
    )
    result_faces[fname] = inpainted

    print(f"    [{fname}] photographer: coverage={cov:.2f}  inpaint={did}")

# ERP 재합성 (down face no_blend)
result_erp = conv.cubemap_to_erp(result_faces, erp_h, erp_w, no_blend_faces=("down",))
save_erp(result_erp, str(DEBUG_OUT / "result_erp.jpg"))
print(f"[2] result_erp.jpg 저장")

# ── 3. erp_nadir_crop_v2.jpg ──────────────────────────────────────────────
erp_bgr = cv2.imread(str(DEBUG_OUT / "result_erp.jpg"))
H_e, W_e = erp_bgr.shape[:2]
nadir    = erp_bgr[int(H_e * 0.75):, :]
cv2.imwrite(str(DEBUG_OUT / "erp_nadir_crop_v2.jpg"), nadir)
print(f"[3] erp_nadir_crop_v2.jpg: {W_e}x{nadir.shape[0]}px")

# ── 파일 목록 ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
for name in ["seg_left_v2.jpg", "inpaint_before_after_down.jpg",
             "erp_nadir_crop_v2.jpg", "result_erp.jpg"]:
    p = DEBUG_OUT / name
    if p.exists():
        print(f"  OK  {name:<42} {p.stat().st_size//1024:>5} KB")
    else:
        print(f"  --  {name}")
