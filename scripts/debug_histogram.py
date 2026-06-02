"""
debug_histogram.py — down face inpainting 경계 색상 차이 측정 + histogram matching.

저장:
  debug_output/erp_nadir_crop_v4.jpg — histogram matching 적용 후 나디르 크롭

실행:
    python scripts/debug_histogram.py
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


def t2rgb(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


from pipeline.cubemap import CubeMapConverter, load_erp, save_erp
from pipeline.segmentation import PersonSegmenter
from pipeline.inpainting import LamaInpainter
from pipeline.matching import BackgroundMatcher

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4,
    "mask_dilate_px": 15, "feature_matcher": "sift", "min_match_count": 10,
}
img_paths     = sorted((ROOT / "img").glob("*.jpg"))
conv          = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter     = PersonSegmenter(cfg)
inpainter     = LamaInpainter(device=DEVICE, debug_dir=DEBUG_OUT, down_feather_px=0)
matcher       = BackgroundMatcher(cfg)

erp0             = load_erp(str(img_paths[0]))
_, erp_h, erp_w  = erp0.shape
faces            = conv.erp_to_cubemap(erp0)

# ── down face 처리 ─────────────────────────────────────────────────────────
seg_down  = segmenter.segment_face(faces["down"], "down", erp_h, erp_w)
pm        = seg_down["photographer_mask"]   # (512,512) bool

src_faces_list = [conv.erp_to_cubemap(load_erp(str(p))) for p in img_paths[1:4]]
srcs      = [sf["down"] for sf in src_faces_list]
best      = matcher.select_best_sources(faces["down"], srcs, pm, top_k=3, face_name="down")
warped    = matcher.blend_multiple_sources(faces["down"], pm, best, face_name="down") if best else faces["down"]
result_t, _ = inpainter.inpaint_residual(warped, pm, filled_mask=False, face_name="down")

# numpy 변환 (RGB, float32 [0,1])
result_rgb  = t2rgb(result_t).astype(np.float32) / 255.0   # (H,W,3) float32
mask_u8     = pm.cpu().numpy().astype(np.uint8) * 255        # (H,W) uint8

# ══════════════════════════════════════════════════════════════════════════
# 1. 경계 내/외 5px 픽셀 색상 측정
# ══════════════════════════════════════════════════════════════════════════

kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 5px radius

inner_ring = cv2.dilate(mask_u8,  kernel5) - mask_u8              # 아직 밖(잘못됨 → 수정)
# 실제로: inner_ring = mask & ~erode(mask, 5)
eroded5      = cv2.erode(mask_u8,  kernel5)
inner_ring   = (mask_u8 > 0).astype(np.uint8) * 255 - (eroded5 > 0).astype(np.uint8) * 255
inner_ring   = np.clip(inner_ring, 0, 255)

# outer_ring = dilate(mask,5) & ~mask
dilated5     = cv2.dilate(mask_u8, kernel5)
outer_ring   = np.clip(dilated5.astype(np.int16) - mask_u8.astype(np.int16), 0, 255).astype(np.uint8)

inner_px  = result_rgb[inner_ring > 0]   # (N,3) float32
outer_px  = result_rgb[outer_ring > 0]   # (N,3) float32

inner_mean = inner_px.mean(axis=0) if len(inner_px) else np.zeros(3)
outer_mean = outer_px.mean(axis=0) if len(outer_px) else np.zeros(3)
diff_mean  = np.abs(inner_mean - outer_mean)

print(f"\n[1] 경계 색상 측정")
print(f"    inner_ring px : {len(inner_px)}")
print(f"    outer_ring px : {len(outer_px)}")
print(f"    inner mean (RGB): {inner_mean.round(4)}")
print(f"    outer mean (RGB): {outer_mean.round(4)}")
print(f"    diff:             {diff_mean.round(4)}  (max={diff_mean.max():.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 2. Histogram matching (mask 내부 → 외부 분포에 맞게 보정)
# ══════════════════════════════════════════════════════════════════════════

def match_histograms_np(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    source 픽셀 분포를 reference 분포에 맞게 보정 (채널별 독립 처리).
    source, reference: (N, 3) float32 [0,1]
    반환: (N, 3) float32 [0,1]
    """
    result = np.empty_like(source)
    for c in range(source.shape[1]):
        s_vals, s_idx = np.unique(source[:, c], return_inverse=True)
        r_vals        = np.sort(reference[:, c])
        # CDF 정규화
        s_cdf = np.linspace(0, 1, len(s_vals))
        r_cdf = np.linspace(0, 1, len(r_vals))
        # source CDF → reference 값으로 매핑
        mapped = np.interp(s_cdf, r_cdf, r_vals)
        result[:, c] = mapped[s_idx]
    return result.astype(np.float32)

if True:
    mask_bool   = pm.cpu().numpy()                                  # (H,W) bool
    result_f32  = result_rgb.copy()                                 # float32 [0,1]

    inpainted_region = result_f32[mask_bool]                        # (N,3)
    reference_region = result_f32[~mask_bool]                       # (M,3)

    matched = match_histograms_np(inpainted_region, reference_region)

    result_matched        = result_f32.copy()
    result_matched[mask_bool] = matched

    # 경계 재측정
    inner_px_m   = result_matched[inner_ring > 0]
    outer_px_m   = result_matched[outer_ring > 0]
    inner_mean_m = inner_px_m.mean(axis=0) if len(inner_px_m) else np.zeros(3)
    outer_mean_m = outer_px_m.mean(axis=0) if len(outer_px_m) else np.zeros(3)
    diff_mean_m  = np.abs(inner_mean_m - outer_mean_m)

    print(f"\n[2] Histogram matching 후")
    print(f"    inner mean (RGB): {inner_mean_m.round(4)}")
    print(f"    outer mean (RGB): {outer_mean_m.round(4)}")
    print(f"    diff:             {diff_mean_m.round(4)}  (max={diff_mean_m.max():.4f})")

    # matched result → tensor
    result_matched_t = torch.from_numpy(result_matched).permute(2, 0, 1).float().to(DEVICE)

    # ── 전체 파이프라인 재합성 ──────────────────────────────────────────────
    result_faces = dict(faces)
    for fname in ["front", "right", "back", "left", "up"]:
        seg_f = segmenter.segment_face(faces[fname], fname, erp_h, erp_w)
        p_m   = seg_f["photographer_mask"]
        if not p_m.any():
            result_faces[fname] = faces[fname]
            continue
        s      = [sf[fname] for sf in src_faces_list]
        bst    = matcher.select_best_sources(faces[fname], s, p_m, top_k=3, face_name=fname)
        warped_f = matcher.blend_multiple_sources(faces[fname], p_m, bst, face_name=fname) if bst else faces[fname]
        result_faces[fname], _ = inpainter.inpaint_residual(warped_f, p_m, False, face_name=fname)

    result_faces["down"] = result_matched_t
    result_erp = conv.cubemap_to_erp(result_faces, erp_h, erp_w)
    save_erp(result_erp, str(DEBUG_OUT / "result_erp_v4.jpg"))

    erp_bgr = cv2.imread(str(DEBUG_OUT / "result_erp_v4.jpg"))
    H_e, W_e = erp_bgr.shape[:2]
    nadir    = erp_bgr[int(H_e * 0.75):, :]
    cv2.imwrite(str(DEBUG_OUT / "erp_nadir_crop_v4.jpg"), nadir)

    # ── 비교 패널 (before / after / diff 나란히) ────────────────────────────
    before_bgr  = cv2.cvtColor((result_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    after_bgr   = cv2.cvtColor((result_matched * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    diff_vis    = np.abs(before_bgr.astype(np.int16) - after_bgr.astype(np.int16)).astype(np.uint8)
    diff_vis    = cv2.applyColorMap(diff_vis * 5, cv2.COLORMAP_JET)

    # 마스크 경계 표시
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for img in (before_bgr, after_bgr, diff_vis):
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)

    cv2.putText(before_bgr, "before hm", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(after_bgr,  "after hm",  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(diff_vis,   "diff x5",   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    panel = np.hstack([before_bgr, after_bgr, diff_vis])
    cv2.imwrite(str(DEBUG_OUT / "histogram_match_down.jpg"), panel)

    print(f"\n저장 완료:")
    for name in ["erp_nadir_crop_v4.jpg", "histogram_match_down.jpg"]:
        p = DEBUG_OUT / name
        print(f"  {name}: {p.stat().st_size // 1024} KB")
