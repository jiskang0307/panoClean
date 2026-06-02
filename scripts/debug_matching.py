"""
debug_matching.py — source 선택 로직 / left face matching / 배경 교체 확인.

실행:
    python scripts/debug_matching.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACE   = "left"

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

erp0            = load_erp(str(img_paths[0]))
_, erp_h, erp_w = erp0.shape
target_face     = conv.erp_to_cubemap(erp0)[FACE]
seg_result      = seg.segment_face(target_face, FACE, erp_h, erp_w)
photo_mask      = seg_result["photographer_mask"]
mask_total      = int(photo_mask.sum())

print(f"target : {img_paths[0].name}")
print(f"FACE   : {FACE}  mask_px={mask_total}  ({mask_total/(512*512)*100:.1f}%)")
print()

# ══════════════════════════════════════════════════════════════════════════
# 1. source 선택 로직
# ══════════════════════════════════════════════════════════════════════════
print("=== [1] batch_runner._process_single 소스 선택 로직 ===")
print("  line 86 : source_faces_list = [f for i,f in enumerate(all_faces) if i != img_idx]")
print("             → 배치 내 모든 이미지 사용 (자신 제외)")
print("  line 140: src_candidates = [sf[face_name] for sf in source_faces_list]")
print("             → 해당 face의 tensor 목록")
print("  line 124: top_k = cfg.get('top_k_sources', 3)  → 기본값 3")
print("  select_best_sources() → coverage 기준 상위 top_k 개 반환")
print()

# ══════════════════════════════════════════════════════════════════════════
# 2. 소스 5장 matching 상세
# ══════════════════════════════════════════════════════════════════════════
print("=== [2] left face matching 상세 (소스 5장) ===")
mask_dev = photo_mask.to(target_face.device)
rows = []

for i, sp in enumerate(img_paths[1:6]):
    src_face = conv.erp_to_cubemap(load_erp(str(sp)))[FACE]

    # keypoint 수 측정
    pts0, pts1 = matcher._match_keypoints(src_face, target_face, force_sift=True)
    n_matched = len(pts0) if pts0 is not None else 0

    # homography + coverage
    try:
        H, inlier_ratio = matcher.find_homography(src_face, target_face, FACE)
    except Exception:
        rows.append((sp.name[:30], n_matched, 0.0, 0.0))
        continue

    if H is None:
        rows.append((sp.name[:30], n_matched, 0.0, 0.0))
        continue

    warped        = matcher.warp_background(src_face, mask_dev, H)
    valid_in_mask = mask_dev & (warped.sum(0) > 0.01)
    cov           = float(valid_in_mask.sum()) / max(mask_total, 1)
    rows.append((sp.name[:30], n_matched, inlier_ratio, cov))

header = f"  {'source':<32} {'matched_kp':>10} {'inlier':>8} {'coverage':>10}"
print(header)
print("  " + "-" * 62)
for name, kp, inl, cov in rows:
    flag = " OK" if cov > 0.1 else " --"
    print(f"  {name:<32} {kp:>10}  {inl:>7.2f}  {cov:>9.2f} {flag}")

# ══════════════════════════════════════════════════════════════════════════
# 3. 실제 배경 교체 결과
# ══════════════════════════════════════════════════════════════════════════
print()
print("=== [3] 실제 배경 교체 결과 ===")
src_list = [
    conv.erp_to_cubemap(load_erp(str(p)))[FACE]
    for p in img_paths[1:6]
]
best = matcher.select_best_sources(
    target_face, src_list, photo_mask, top_k=3, face_name=FACE
)
print(f"  select_best_sources → {len(best)}개 선택됨")
for rank, (src_t, H_mat, cov) in enumerate(best):
    print(f"    rank{rank}: coverage={cov:.3f}")

if best:
    restored  = matcher.blend_multiple_sources(
        target_face, photo_mask, best, face_name=FACE
    )
    filled    = photo_mask & (
        (restored.cpu() - target_face.cpu()).abs().sum(0) > 0.02
    )
    filled_r  = float(filled.sum()) / max(mask_total, 1)
    print(f"  blend 후 실제 변경된 픽셀 비율 (filled_ratio): {filled_r:.3f}")
    print(f"  coverage=0.0 여부: {filled_r < 0.01}")
    print(f"  배경 교체 성공: {filled_r > 0.01}")
else:
    print("  best_sources = [] → 배경 교체 불가  coverage=0.0")
