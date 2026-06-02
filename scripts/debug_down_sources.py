"""
debug_down_sources.py — 소스 586~590의 down face coverage + person 비율 분석.

실행:
    python scripts/debug_down_sources.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter
from pipeline.matching import BackgroundMatcher

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4,
    "mask_dilate_px": 15, "feature_matcher": "sift", "min_match_count": 10,
}
img_paths = sorted((ROOT / "img").glob("*.jpg"))
conv    = CubeMapConverter(face_size=512, device=DEVICE)
seg     = PersonSegmenter(cfg)
matcher = BackgroundMatcher(cfg)

# ── 타깃 down face ────────────────────────────────────────────────────────
erp0            = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
target_down     = conv.erp_to_cubemap(erp0)["down"]
target_seg      = seg.segment_face(target_down, "down", erp_h, erp_w)
target_pm       = target_seg["photographer_mask"]
mask_total      = int(target_pm.sum())
mask_dev        = target_pm.to(target_down.device)

print(f"target  : {img_paths[0].name}")
print(f"target down  mask_px={mask_total}  person_ratio={mask_total/(512*512):.3f}")
print()
print(f"{'src':<30} {'coverage':>10} {'person_ratio':>14} {'inlier':>8} {'kp':>6}")
print("-" * 70)

# ── 소스 5장 분석 ─────────────────────────────────────────────────────────
for sp in img_paths[1:6]:
    src_down = conv.erp_to_cubemap(load_erp(str(sp)))["down"]

    # source down face의 person(촬영자) 비율
    src_seg      = seg.segment_face(src_down, "down", erp_h, erp_w)
    src_pm       = src_seg["photographer_mask"]
    person_ratio = float(src_pm.sum()) / (512 * 512)

    # keypoint 수
    pts0, pts1 = matcher._match_keypoints(src_down, target_down, force_sift=True)
    n_kp = len(pts0) if pts0 is not None else 0

    # homography + coverage
    try:
        H, inlier_ratio = matcher.find_homography(src_down, target_down, "down")
    except Exception:
        H, inlier_ratio = None, 0.0

    if H is None:
        coverage = 0.0
    else:
        warped        = matcher.warp_background(src_down, mask_dev, H)
        valid_in_mask = mask_dev & (warped.sum(0) > 0.01)
        coverage      = float(valid_in_mask.sum()) / max(mask_total, 1)

    print(
        f"{sp.name[:28]:<30} {coverage:>10.2f} {person_ratio:>14.3f}"
        f" {inlier_ratio:>8.2f} {n_kp:>6}"
    )
