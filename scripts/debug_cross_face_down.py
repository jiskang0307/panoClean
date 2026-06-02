"""
debug_cross_face_down.py — 인접 10장 × 5 face에서 down face 배경 탐색.

target 앞뒤 5장씩 (총 최대 10장)의 front/back/left/right/down face를
target down face에 매칭하여 coverage 상위 3개로 blend 복원.

저장:
  debug_output/down_cross_face_replace.jpg

실행:
    python scripts/debug_cross_face_down.py
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

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE    = 512
SEARCH_FACES = ["front", "back", "left", "right", "down"]

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter
from pipeline.matching import BackgroundMatcher

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4,
    "mask_dilate_px": 15, "feature_matcher": "sift", "min_match_count": 10,
}
img_paths = sorted((ROOT / "img").glob("*.jpg"))
conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
seg       = PersonSegmenter(cfg)
matcher   = BackgroundMatcher(cfg)

# ── target 설정 ──────────────────────────────────────────────────────────
target_path = img_paths[0]
target_idx  = 0
erp0        = load_erp(str(target_path))
_, erp_h, erp_w = erp0.shape
target_down = conv.erp_to_cubemap(erp0)["down"]
seg_down    = seg.segment_face(target_down, "down", erp_h, erp_w)
target_pm   = seg_down["photographer_mask"]   # (512,512) bool
mask_dev    = target_pm.to(DEVICE)
mask_total  = int(target_pm.sum())

print(f"target  : {target_path.name}")
print(f"down mask_px={mask_total}  person_ratio={mask_total / FACE_SIZE**2:.3f}")
print()

# ── 인접 10장 (앞뒤 5장씩) ───────────────────────────────────────────────
n = len(img_paths)
before = list(range(max(0, target_idx - 5), target_idx))
after  = list(range(target_idx + 1, min(n, target_idx + 6)))
adjacent_indices = before + after
adjacent_paths   = [img_paths[i] for i in adjacent_indices]
print(f"인접 소스: {len(adjacent_paths)}장  "
      f"(앞{len(before)}장 + 뒤{len(after)}장)")
print()

# ── 후보 수집 ────────────────────────────────────────────────────────────
candidates = []  # (warped_tensor, coverage, src_name, face_name)

print(f"{'src':<30} {'face':<6} {'clean%':>7} {'inlier':>7} {'coverage':>9}  note")
print("─" * 72)

for sp in adjacent_paths:
    try:
        src_cubemap = conv.erp_to_cubemap(load_erp(str(sp)))
    except Exception as e:
        print(f"{sp.name[:28]:<30} {'':6} {'':7}  load error: {e}")
        continue

    for face_name in SEARCH_FACES:
        src_face = src_cubemap[face_name]

        # 사람 없는 clean 픽셀 비율 측정
        src_seg      = seg.segment_face(src_face, face_name, erp_h, erp_w)
        src_pm       = src_seg["photographer_mask"]
        person_ratio = float(src_pm.sum()) / FACE_SIZE**2
        clean_ratio  = 1.0 - person_ratio

        if clean_ratio < 0.30:
            print(f"{sp.name[:28]:<30} {face_name:<6} {clean_ratio*100:>6.1f}%"
                  f"  {'':>7}  {'':>9}  skip")
            continue

        # homography (down 기준으로 force SIFT)
        try:
            H, inlier_ratio = matcher.find_homography(src_face, target_down, "down")
        except Exception as e:
            print(f"{sp.name[:28]:<30} {face_name:<6} {clean_ratio*100:>6.1f}%"
                  f"  {'err':>7}  {'':>9}  {e}")
            continue

        if H is None:
            print(f"{sp.name[:28]:<30} {face_name:<6} {clean_ratio*100:>6.1f}%"
                  f"  {'—':>7}  {'':>9}  no H")
            continue

        if inlier_ratio < 0.25:
            print(f"{sp.name[:28]:<30} {face_name:<6} {clean_ratio*100:>6.1f}%"
                  f"  {inlier_ratio:>7.2f}  {'':>9}  low inlier")
            continue

        # warp + coverage
        warped   = matcher.warp_background(src_face, mask_dev, H)
        valid    = mask_dev & (warped.sum(0) > 0.01)
        coverage = float(valid.sum()) / max(mask_total, 1)

        note = "OK" if clean_ratio >= 0.50 else "OK (low clean)"
        print(f"{sp.name[:28]:<30} {face_name:<6} {clean_ratio*100:>6.1f}%"
              f"  {inlier_ratio:>7.2f}  {coverage:>9.3f}  {note}")
        candidates.append((warped, coverage, sp.name, face_name, clean_ratio))

print()
print(f"총 후보: {len(candidates)}개")

if not candidates:
    print("유효 후보 없음 — 종료")
    sys.exit(0)

# ── 촬영자 50% 이상인 소스 제외 후 정렬 ──────────────────────────────────
filtered = [c for c in candidates if c[4] >= 0.50]
excluded = len(candidates) - len(filtered)
print(f"clean_ratio < 0.50 제외: {excluded}개 → 남은 후보: {len(filtered)}개")

if not filtered:
    print("필터 후 후보 없음 — clean_ratio < 0.50 완화하여 전체 사용")
    filtered = candidates

# clean_ratio 높은 순 → coverage 높은 순
filtered.sort(key=lambda x: (-x[4], -x[1]))
top3 = filtered[:3]

print(f"\n=== 필터링 후 TOP{len(top3)} ===")
for i, (_, cov, sname, fname, clean) in enumerate(top3):
    print(f"  rank{i}: {sname} [{fname}]  coverage={cov:.3f}  clean={clean:.3f}")

# softmax 가중치 (coverage 기준)
cov_arr = np.array([c[1] for c in top3], dtype=np.float32)
exp_c   = np.exp(cov_arr - cov_arr.max())
weights = exp_c / exp_c.sum()

# blend
_, h, w = target_down.shape
accum   = torch.zeros(3, h, w, device=DEVICE)
w_map   = torch.zeros(h, w, device=DEVICE)

for (warped, _, _, _, _), weight in zip(top3, weights):
    valid = mask_dev & (warped.sum(0) > 0.01)
    if not valid.any():
        continue
    accum[:, valid] += warped[:, valid] * float(weight)
    w_map[valid]    += float(weight)

covered   = w_map > 0
blend_res = target_down.clone()
if covered.any():
    accum[:, covered] = accum[:, covered] / w_map[covered].unsqueeze(0)
    blend_res[:, mask_dev] = accum[:, mask_dev]

filled_r = float((mask_dev & ((blend_res - target_down).abs().sum(0) > 0.02)).sum()) \
           / max(mask_total, 1)
uncov_r  = float((mask_dev & ~covered).sum()) / max(mask_total, 1)

print(f"\nblend 결과: coverage={float(covered.sum())/max(mask_total,1):.3f}  "
      f"filled={filled_r:.3f}  uncovered={uncov_r:.3f}")

# ── 시각화 ──────────────────────────────────────────────────────────────
def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def draw_contour(bgr: np.ndarray, mask: torch.Tensor,
                 color=(0, 255, 255), thick=2) -> np.ndarray:
    vis  = bgr.copy()
    m_u8 = mask.cpu().numpy().astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis


def label(img: np.ndarray, *lines: str) -> np.ndarray:
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (6, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    return img


# row1: top-3 소스 개별 warp
panels_top = []
for rank, (warped, cov, sname, fname, clean) in enumerate(top3):
    vis = t2bgr(warped)
    vis = draw_contour(vis, target_pm)
    uncov_np = (mask_dev & (warped.sum(0) <= 0.01)).cpu().numpy()
    vis[uncov_np] = [50, 50, 200]  # 미채움 파란색
    label(vis,
          f"rank{rank}  [{fname}]  cov={cov:.2f}  clean={clean:.2f}",
          sname[:26])
    panels_top.append(vis)

while len(panels_top) < 3:
    blank = np.zeros((FACE_SIZE, FACE_SIZE, 3), np.uint8)
    cv2.putText(blank, "NO SOURCE", (130, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
    panels_top.append(blank)

# row2: blend + original + diff
vis_blend = t2bgr(blend_res)
vis_blend = draw_contour(vis_blend, target_pm)
if (mask_dev & ~covered).any():
    vis_blend[(mask_dev & ~covered).cpu().numpy()] = [50, 50, 200]
label(vis_blend,
      f"blend (top{len(top3)}, clean>=0.50)",
      f"filled={filled_r:.3f}  uncov={uncov_r:.3f}")

vis_orig = t2bgr(target_down)
vis_orig = draw_contour(vis_orig, target_pm)
label(vis_orig, "target original")

diff_np = np.abs(
    t2bgr(blend_res).astype(np.int16) - t2bgr(target_down).astype(np.int16)
).astype(np.uint8)
diff_vis = cv2.applyColorMap(np.clip(diff_np * 4, 0, 255), cv2.COLORMAP_JET)
diff_vis = draw_contour(diff_vis, target_pm)
label(diff_vis, "diff x4 (blend vs orig)")

panels_bot = [vis_blend, vis_orig, diff_vis]

row1 = np.hstack(panels_top)
row2 = np.hstack(panels_bot)
result_img = np.vstack([row1, row2])

out_path = DEBUG_OUT / "down_cross_face_v2.jpg"
cv2.imwrite(str(out_path), result_img)
print(f"\n저장: {out_path}  ({out_path.stat().st_size // 1024} KB)")
