"""
debug_check.py — 세 가지 확인 이미지 생성.

저장:
  debug_output/seg_left.jpg          — left face 분류 시각화 (PHOTOGRAPHER=초록, BACKGROUND=노랑)
  debug_output/inpaint_before_after_down.jpg — 기존 파일 확인 (재생성)
  debug_output/erp_nadir_crop.jpg    — result_erp.jpg 하단 25% 크롭

실행:
    python scripts/debug_check.py
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


def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def overlay_mask(bgr: np.ndarray, mask_u8: np.ndarray,
                 fill_bgr: tuple, contour_bgr: tuple,
                 alpha: float = 0.35) -> np.ndarray:
    vis = bgr.copy()
    ov  = vis.copy()
    ov[mask_u8 > 0] = fill_bgr
    cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0, vis)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, contour_bgr, 2)
    return vis


# ── 모듈 로드 ─────────────────────────────────────────────────────────────
from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter, PersonRole
from pipeline.inpainting import LamaInpainter

cfg = {
    "device":         DEVICE,
    "yolo_model":     "yolo11x-seg.pt",
    "yolo_conf":      0.4,
    "mask_dilate_px": 15,
}

img_paths = sorted((ROOT / "img").glob("*.jpg"))
if not img_paths:
    print("img/ 이미지 없음"); sys.exit(1)

conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)
inpainter = LamaInpainter(device=DEVICE, debug_dir=DEBUG_OUT)

erp0           = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
faces          = conv.erp_to_cubemap(erp0)


# ══════════════════════════════════════════════════════════════════════════
# 1. seg_left.jpg — PHOTOGRAPHER=초록, BACKGROUND=노랑
# ══════════════════════════════════════════════════════════════════════════

left_face = faces["left"]
seg       = segmenter.segment_face(left_face, "left", erp_h, erp_w)
img_bgr   = t2bgr(left_face)
vis       = img_bgr.copy()

# BACKGROUND 마스크: 노란색
for bm in seg.get("background_face_masks", []):
    bm_np = bm.cpu().numpy().astype(np.uint8) * 255
    vis = overlay_mask(vis, bm_np, fill_bgr=(0, 200, 255), contour_bgr=(0, 200, 255))

# PHOTOGRAPHER 마스크: 초록색
photo_mask = seg["photographer_mask"]
if photo_mask.any():
    pm_np = photo_mask.cpu().numpy().astype(np.uint8) * 255
    vis = overlay_mask(vis, pm_np, fill_bgr=(0, 200, 0), contour_bgr=(0, 255, 0))
    cv2.putText(vis, "PHOTOGRAPHER (green)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
else:
    cv2.putText(vis, "PHOTOGRAPHER: none", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# 검출 인원 레이블
for det, role in zip(seg["detections"], seg["roles"]):
    x1, y1, x2, y2 = det.box.astype(int)
    role_str = "PHOTO" if role == PersonRole.PHOTOGRAPHER else "BG"
    color    = (0, 255, 0) if role == PersonRole.PHOTOGRAPHER else (0, 200, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    cv2.putText(vis, f"{role_str} px={det.pixel_count}", (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

cv2.imwrite(str(DEBUG_OUT / "seg_left.jpg"), vis)
print(f"[1] seg_left.jpg 저장")
print(f"    검출 {len(seg['detections'])}명  photographer_mask={'있음' if photo_mask.any() else '없음(BACKGROUND만)'}")
for det, role, score in zip(seg["detections"], seg["roles"], seg["role_scores"]):
    print(f"    → {role.value}  score={score:.2f}  px={det.pixel_count}  "
          f"bbox=({det.box[0]:.0f},{det.box[1]:.0f},{det.box[2]:.0f},{det.box[3]:.0f})")


# ══════════════════════════════════════════════════════════════════════════
# 2. inpaint_before_after_down.jpg — 재생성
# ══════════════════════════════════════════════════════════════════════════

down_face = faces["down"]
seg_down  = segmenter.segment_face(down_face, "down", erp_h, erp_w)
photo_down = seg_down["photographer_mask"]

if photo_down.any() and inpainter.available:
    result_down, _ = inpainter.inpaint_residual(
        down_face, photo_down, filled_mask=False, face_name="down"
    )
    # inpainter._save_diff_debug already saves inpaint_before_after_down.jpg
    # but let's ensure it's there:
    out_path = DEBUG_OUT / "inpaint_before_after_down.jpg"
    if out_path.exists():
        print(f"[2] inpaint_before_after_down.jpg 재생성 완료 ({out_path.stat().st_size//1024} KB)")
    else:
        print("[2] inpaint_before_after_down.jpg 저장 실패")
else:
    print("[2] down face photographer 없음 또는 LaMa 비활성화 — 스킵")


# ══════════════════════════════════════════════════════════════════════════
# 3. erp_nadir_crop.jpg — result_erp.jpg 하단 25% 크롭
# ══════════════════════════════════════════════════════════════════════════

result_erp_path = DEBUG_OUT / "result_erp.jpg"
if not result_erp_path.exists():
    # result_erp가 없으면 직접 생성
    from pipeline.matching import BackgroundMatcher
    from pipeline.cubemap import save_erp

    matcher      = BackgroundMatcher(cfg)
    src_items    = []
    for sp in img_paths[1:4]:  # 소스 3장만
        src_erp   = load_erp(str(sp))
        src_faces = conv.erp_to_cubemap(src_erp)
        src_items.append(src_faces)

    result_faces: dict[str, torch.Tensor] = {}
    for fname, face_t in faces.items():
        seg_f = segmenter.segment_face(face_t, fname, erp_h, erp_w)
        pm    = seg_f["photographer_mask"]
        if not pm.any():
            result_faces[fname] = face_t
            continue
        srcs = [sf[fname] for sf in src_items]
        best = matcher.select_best_sources(face_t, srcs, pm, top_k=3, face_name=fname)
        if best:
            restored = matcher.blend_multiple_sources(face_t, pm, best, face_name=fname)
        else:
            restored = face_t
        result_faces[fname], _ = inpainter.inpaint_residual(restored, pm, False, face_name=fname)

    result_erp = conv.cubemap_to_erp(result_faces, erp_h, erp_w)
    save_erp(result_erp, str(result_erp_path))
    print(f"[3] result_erp.jpg 생성 후 크롭")
else:
    print(f"[3] 기존 result_erp.jpg 사용 ({result_erp_path.stat().st_size//1024} KB)")

erp_bgr = cv2.imread(str(result_erp_path))
H_e, W_e = erp_bgr.shape[:2]
nadir_start = int(H_e * 0.75)
nadir       = erp_bgr[nadir_start:, :]
cv2.imwrite(str(DEBUG_OUT / "erp_nadir_crop.jpg"), nadir)
print(f"    erp_nadir_crop.jpg: {W_e}×{nadir.shape[0]}px (하단 25%)")

# ── 저장 파일 목록 ────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
targets = ["seg_left.jpg", "inpaint_before_after_down.jpg", "erp_nadir_crop.jpg"]
for name in targets:
    p = DEBUG_OUT / name
    if p.exists():
        print(f"  ✓ {name:<42} {p.stat().st_size//1024:>5} KB")
    else:
        print(f"  ✗ {name} 없음")
