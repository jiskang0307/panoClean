"""
debug_replace_left.py — left face 배경 교체 단계별 시각화.

저장:
  debug_output/warped_sources_left.jpg  — 소스 3장 개별 warp 나란히
  debug_output/blended_bg_left.jpg      — 3장 weighted blend 결과
  debug_output/replace_only_left.jpg    — SeamlessBlend까지 완료 (LaMa 전)

실행:
    python scripts/debug_replace_left.py
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
photo_mask      = seg_result["photographer_mask"]   # (512,512) bool


def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def draw_mask_contour(bgr: np.ndarray, mask: torch.Tensor,
                      color=(0, 255, 255), thick=2) -> np.ndarray:
    vis  = bgr.copy()
    m_u8 = mask.cpu().numpy().astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis


def label(img: np.ndarray, *lines: str) -> np.ndarray:
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (8, 26 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


# ── 소스 5장 → top 3 선택 ─────────────────────────────────────────────────
src_list = [
    conv.erp_to_cubemap(load_erp(str(p)))[FACE]
    for p in img_paths[1:6]
]
best = matcher.select_best_sources(
    target_face, src_list, photo_mask, top_k=3, face_name=FACE
)
print(f"top {len(best)} sources selected")
for r, (_, _, cov) in enumerate(best):
    print(f"  rank{r}: coverage={cov:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# 1. warped sources 나란히  (mask contour 표시)
# ══════════════════════════════════════════════════════════════════════════

mask_dev   = photo_mask.to(target_face.device)
cov_arr    = np.array([c for _, _, c in best], dtype=np.float32)
exp_c      = np.exp(cov_arr - cov_arr.max())
weights    = exp_c / exp_c.sum()

warp_panels = []
for rank, ((warped, H, cov), w) in enumerate(zip(best, weights)):
    vis = draw_mask_contour(t2bgr(warped), photo_mask)

    # mask 외부(warped=0)를 회색으로 표시
    m_np = photo_mask.cpu().numpy()
    outside_warp = (warped.cpu().sum(0) < 0.01).numpy()
    gray_overlay = vis.copy()
    gray_overlay[outside_warp & m_np] = [80, 80, 80]
    cv2.addWeighted(gray_overlay, 0.4, vis, 0.6, 0, vis)
    cv2.drawContours(vis,
        cv2.findContours(
            (outside_warp & m_np).astype(np.uint8)*255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0], -1, (0,0,200), 1)

    label(vis,
          f"rank{rank}  cov={cov:.2f}  w={w:.2f}",
          f"src: {img_paths[rank+1].name[:20]}")
    warp_panels.append(vis)

if len(warp_panels) < 3:
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(blank, "NO SOURCE", (150, 256),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    while len(warp_panels) < 3:
        warp_panels.append(blank)

cv2.imwrite(str(DEBUG_OUT / "warped_sources_left.jpg"),
            np.hstack(warp_panels))
print(f"[1] warped_sources_left.jpg 저장")


# ══════════════════════════════════════════════════════════════════════════
# 2. weighted blend 결과 (SeamlessBlend 전 단계)
# ══════════════════════════════════════════════════════════════════════════

_, h, w = target_face.shape
accum  = torch.zeros_like(target_face)
w_map  = torch.zeros(h, w, device=target_face.device)

for (warped, _, _), weight in zip(best, weights):
    valid = mask_dev & (warped.sum(0) > 0.01)
    if not valid.any():
        continue
    accum[:, valid] += warped[:, valid] * float(weight)
    w_map[valid]    += float(weight)

covered = w_map > 0
blend_raw = target_face.clone()
if covered.any():
    accum[:, covered] = accum[:, covered] / w_map[covered].unsqueeze(0)
    blend_raw[:, mask_dev] = accum[:, mask_dev]

vis_blend = draw_mask_contour(t2bgr(blend_raw), photo_mask)
uncovered = mask_dev & ~covered
if uncovered.any():
    uc_np = uncovered.cpu().numpy()
    vis_blend[uc_np] = [0, 0, 200]   # 미채움 영역 빨간색
    label(vis_blend,
          f"blend (raw) cov={float(covered.sum())/max(int(mask_dev.sum()),1):.2f}",
          f"red=uncovered ({int(uncovered.sum())}px)")
else:
    label(vis_blend,
          f"blend (raw) cov={float(covered.sum())/max(int(mask_dev.sum()),1):.2f}",
          "fully covered")

cv2.imwrite(str(DEBUG_OUT / "blended_bg_left.jpg"), vis_blend)
print(f"[2] blended_bg_left.jpg 저장  covered={float(covered.sum())/max(int(mask_dev.sum()),1):.3f}")


# ══════════════════════════════════════════════════════════════════════════
# 3. blend_multiple_sources 완료 (SeamlessBlend 포함) — LaMa 전
# ══════════════════════════════════════════════════════════════════════════

restored = matcher.blend_multiple_sources(
    target_face, photo_mask, best, face_name=FACE
)
filled   = photo_mask & ((restored.cpu() - target_face.cpu()).abs().sum(0) > 0.02)
filled_r = float(filled.sum()) / max(int(photo_mask.sum()), 1)

vis_final = draw_mask_contour(t2bgr(restored), photo_mask)
label(vis_final,
      f"replace_only (SeamlessBlend)",
      f"filled={filled_r:.3f}  ({int(filled.sum())}/{int(photo_mask.sum())}px)")

cv2.imwrite(str(DEBUG_OUT / "replace_only_left.jpg"), vis_final)
print(f"[3] replace_only_left.jpg 저장  filled_ratio={filled_r:.3f}")

# ── 파일 크기 요약 ─────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
for name in ["warped_sources_left.jpg", "blended_bg_left.jpg", "replace_only_left.jpg"]:
    p = DEBUG_OUT / name
    if p.exists():
        print(f"  {name}: {p.stat().st_size//1024} KB")
