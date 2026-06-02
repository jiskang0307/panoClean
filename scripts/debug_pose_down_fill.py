"""
debug_pose_down_fill.py — 구면 좌표 기반 relative pose → down face 복원.

target(582)과 5번째 이미지(589) ERP에서 SIFT 매칭
→ RANSAC rotation 추정 → target down face를 source ERP에 재투영
→ source의 촬영자 없는 픽셀로 target down face 복원.

저장:
  debug_output/pose_based_down_fill.jpg   — 복원 결과 + match 시각화

실행:
    python scripts/debug_pose_down_fill.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEBUG_OUT = ROOT / "debug_output"
DEBUG_OUT.mkdir(exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE = 512

from pipeline.cubemap import CubeMapConverter, load_erp
from pipeline.segmentation import PersonSegmenter

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt", "yolo_conf": 0.4,
    "mask_dilate_px": 15,
}
img_paths = sorted((ROOT / "img").glob("*.jpg"))
conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)

target_path = img_paths[0]   # 582
src_path    = img_paths[4]   # 589 (5번째)
print(f"target : {target_path.name}")
print(f"source : {src_path.name}")

# ── Load ERPs ─────────────────────────────────────────────────────────────
target_erp = load_erp(str(target_path))   # (3, H, W)
src_erp    = load_erp(str(src_path))
_, erp_h, erp_w = target_erp.shape
print(f"ERP: {erp_h}×{erp_w}")

# ── Target down face photographer mask ───────────────────────────────────
target_faces = conv.erp_to_cubemap(target_erp)
target_down  = target_faces["down"]
seg_down     = segmenter.segment_face(target_down, "down", erp_h, erp_w)
target_pm    = seg_down["photographer_mask"]   # (512,512) bool
mask_total   = int(target_pm.sum())
print(f"target down mask: {mask_total}px ({mask_total/FACE_SIZE**2:.3f})")

# ═════════════════════════════════════════════════════════════════════════
# 유틸: ERP ↔ 3D 단위벡터 변환
# ═════════════════════════════════════════════════════════════════════════

def erp_pixel_to_unit(u, v, W, H):
    """ERP pixel (u,v) → 3D 단위벡터. u,v: 배열 가능."""
    lon = 2 * np.pi * (np.asarray(u, float) / W - 0.5)
    lat = np.pi  * (0.5 - np.asarray(v, float) / H)
    x   = np.cos(lat) * np.sin(lon)
    y   = np.sin(lat)
    z   = np.cos(lat) * np.cos(lon)
    d   = np.stack([x, y, z], axis=-1)
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)

def unit_to_erp_pixel(d, W, H):
    """3D 단위벡터 d → ERP pixel (u, v). d: (...,3)."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.sqrt(x**2 + z**2 + 1e-12))
    u   = W * (lon / (2 * np.pi) + 0.5)
    v   = H * (0.5 - lat / np.pi)
    return u, v

# face UV (fx,fy)∈[-1,1] → 3D 방향벡터 (정규화 전)
# 역변환: _lonlat_to_face_xy에서 유도
#   front : fy = -vy/vz → vy = -fy·vz;  fx = vx/vz → vx = fx·vz  → dir=(fx,-fy,1)
#   back  : fy =  vy/vz → vy = fy·|vz|; fx = vx/vz → vx = fx·|vz| (vz<0)
#           → dir=(-fx,-fy,-1)  [d=vz<0이므로 vx=fx*d=fx*(-1)=-fx]
#   right : fy =-vy/vx → vy=-fy;  fx=-vz/vx → vz=-fx  → dir=(1,-fy,-fx)
#   left  : fy = vy/|vx|→ vy=-fy; fx=-vz/|vx|→ vz=fx   → dir=(-1,-fy,fx)
#   up    : fy = vz/vy → vz=fy;   fx=vx/vy → vx=fx       → dir=(fx,1,fy)
#   down  : fy = vz/|vy|→ vz=-fy; fx=-vx/|vy|→ vx=fx    → dir=(fx,-1,-fy)
_FACE_DIR = {
    "front": lambda fx, fy: (fx,  -fy,  np.ones_like(fx)),
    "back":  lambda fx, fy: (-fx, -fy, -np.ones_like(fx)),
    "right": lambda fx, fy: (np.ones_like(fx),  -fy, -fx),
    "left":  lambda fx, fy: (-np.ones_like(fx), -fy,  fx),
    "up":    lambda fx, fy: (fx,  np.ones_like(fx),  fy),
    "down":  lambda fx, fy: (fx, -np.ones_like(fx), -fy),
}

def face_pixels_to_unit(face_name, rows=None, cols=None, face_size=FACE_SIZE):
    """face 픽셀 (rows,cols) → 3D 단위벡터 배열."""
    if rows is None:
        rows = np.arange(face_size)
        cols = np.arange(face_size)
        jj, ii = np.meshgrid(cols, rows)
    else:
        ii, jj = np.asarray(rows), np.asarray(cols)
    fx = 2.0 * jj / (face_size - 1) - 1.0
    fy = 2.0 * ii / (face_size - 1) - 1.0
    dx, dy, dz = _FACE_DIR[face_name](fx, fy)
    d = np.stack([dx, dy, dz], axis=-1).astype(float)
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)

# ═════════════════════════════════════════════════════════════════════════
# Step 1: SIFT match on ERP (10% 크기)
# ═════════════════════════════════════════════════════════════════════════
SIFT_SCALE = 0.30
Hs = int(erp_h * SIFT_SCALE)
Ws = int(erp_w * SIFT_SCALE)

def t2rgb_np(t):
    return (t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)

th_small  = cv2.resize(t2rgb_np(target_erp), (Ws, Hs))
src_small = cv2.resize(t2rgb_np(src_erp),    (Ws, Hs))

sift = cv2.SIFT_create(nfeatures=5000)
kp0, des0 = sift.detectAndCompute(cv2.cvtColor(th_small,  cv2.COLOR_RGB2GRAY), None)
kp1, des1 = sift.detectAndCompute(cv2.cvtColor(src_small, cv2.COLOR_RGB2GRAY), None)

flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 100})
raw   = flann.knnMatch(des0, des1, k=2)
good  = [m for m, n in raw if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
print(f"\nSIFT matches: {len(good)}")

pts0_s = np.float32([kp0[m.queryIdx].pt for m in good])   # small 좌표
pts1_s = np.float32([kp1[m.trainIdx].pt for m in good])
pts0_o = pts0_s / SIFT_SCALE   # 원본 ERP 좌표
pts1_o = pts1_s / SIFT_SCALE

# ═════════════════════════════════════════════════════════════════════════
# Step 2: 3D 단위벡터 변환 + 적도 필터링
# ═════════════════════════════════════════════════════════════════════════
d0 = erp_pixel_to_unit(pts0_o[:, 0], pts0_o[:, 1], erp_w, erp_h)   # (N,3)
d1 = erp_pixel_to_unit(pts1_o[:, 0], pts1_o[:, 1], erp_w, erp_h)

lat0 = np.arcsin(np.clip(d0[:, 1], -1, 1))
equatorial = np.abs(lat0) < np.radians(60)
d0_eq  = d0[equatorial]
d1_eq  = d1[equatorial]
pts0_eq = pts0_o[equatorial]
pts1_eq = pts1_o[equatorial]
print(f"Equatorial matches (|lat|<60°): {equatorial.sum()}")

# ═════════════════════════════════════════════════════════════════════════
# Step 3: RANSAC rotation 추정 (Kabsch)
# ═════════════════════════════════════════════════════════════════════════

def kabsch(d0, d1):
    """R such that R @ d0 ≈ d1 (least-squares, 3-pt minimum)."""
    H_mat = d0.T @ d1
    U, _, Vt = np.linalg.svd(H_mat)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R

def ransac_rotation(d0, d1, n_iter=1000, thresh_deg=2.0):
    thresh_cos = np.cos(np.radians(thresh_deg))
    best_R, best_n = None, 0
    N = len(d0)
    np.random.seed(42)
    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        try:
            R = kabsch(d0[idx], d1[idx])
        except Exception:
            continue
        cos_a   = ((R @ d0.T).T * d1).sum(1)
        inliers = cos_a > thresh_cos
        if inliers.sum() > best_n:
            best_n = inliers.sum()
            best_R = R
    # Refine on all inliers
    if best_R is not None:
        cos_a   = ((best_R @ d0.T).T * d1).sum(1)
        inliers = cos_a > thresh_cos
        if inliers.sum() >= 3:
            best_R = kabsch(d0[inliers], d1[inliers])
            best_n = inliers.sum()
    return best_R, best_n

R_kabsch, n_inliers_k = ransac_rotation(d0_eq, d1_eq)
print(f"Kabsch R:   inliers={n_inliers_k}/{len(d0_eq)} "
      f"({n_inliers_k/max(len(d0_eq),1):.2f})")

# ── Essential Matrix (virtual K for equatorial strip) ────────────────────
Hs_eq = int(erp_h * SIFT_SCALE)
Ws_eq = int(erp_w * SIFT_SCALE)
f_eff = Ws_eq / (2 * np.pi)
K_virt = np.array([[f_eff, 0, Ws_eq / 2],
                   [0, f_eff, Hs_eq / 2],
                   [0, 0,     1]], dtype=np.float64)

# |lat| < 45° 범위의 match만 사용
lat0_all = np.arcsin(np.clip(d0[:, 1], -1, 1))
strict_eq = np.abs(lat0_all) < np.radians(45)
pts0_eq45  = pts0_o[strict_eq] * SIFT_SCALE    # small 좌표로 변환
pts1_eq45  = pts1_o[strict_eq] * SIFT_SCALE
print(f"|lat|<45° matches: {strict_eq.sum()}")

R_ess = R_kabsch    # fallback
if len(pts0_eq45) >= 8:
    try:
        E, e_mask = cv2.findEssentialMat(
            pts0_eq45, pts1_eq45, K_virt,
            method=cv2.RANSAC, prob=0.999, threshold=1.5,
        )
        if E is not None and e_mask is not None:
            e_inliers = int(e_mask.sum())
            print(f"Essential Matrix: inliers={e_inliers}/{len(pts0_eq45)}")
            _, R_rec, t_rec, rec_mask = cv2.recoverPose(
                E, pts0_eq45, pts1_eq45, K_virt, mask=e_mask.copy()
            )
            print(f"recoverPose: R_rec valid, t_dir={t_rec.flatten().round(3)}")
            R_ess = R_rec
        else:
            print("Essential Matrix 추정 실패 — Kabsch fallback")
    except Exception as ex:
        print(f"Essential Matrix error: {ex} — Kabsch fallback")
else:
    print(f"|lat|<45° 매칭 부족 ({len(pts0_eq45)}) — Kabsch fallback")

R = R_ess
print(f"사용 R 행렬:\n{R.round(4)}")

# ═════════════════════════════════════════════════════════════════════════
# Step 4: source ERP 촬영자 마스크 (face별 → ERP 좌표로 back-project)
# ═════════════════════════════════════════════════════════════════════════
print("\n소스 ERP 촬영자 마스크 생성 중...")
src_faces     = conv.erp_to_cubemap(src_erp)
src_erp_pm    = np.zeros((erp_h, erp_w), dtype=bool)

for face_name in ("front", "back", "left", "right", "down", "up"):
    src_face = src_faces[face_name]
    seg_res  = segmenter.segment_face(src_face, face_name, erp_h, erp_w)
    face_pm  = seg_res["photographer_mask"].cpu().numpy()  # (512,512)
    if not face_pm.any():
        continue
    d_face   = face_pixels_to_unit(face_name)               # (512,512,3)
    u_f, v_f = unit_to_erp_pixel(d_face, erp_w, erp_h)
    ui = np.clip(u_f.astype(int), 0, erp_w - 1)
    vi = np.clip(v_f.astype(int), 0, erp_h - 1)
    src_erp_pm[vi[face_pm], ui[face_pm]] = True
    print(f"  {face_name}: person_px={face_pm.sum()}  "
          f"(clean_ratio={1-face_pm.mean():.3f})")

print(f"  source ERP 촬영자 마스크 총 px: {src_erp_pm.sum()}")

# ═════════════════════════════════════════════════════════════════════════
# Step 5: target down face → R 적용 → source ERP UV 재투영
# ═════════════════════════════════════════════════════════════════════════
d_down = face_pixels_to_unit("down")           # (512,512,3)
d_flat = d_down.reshape(-1, 3)                 # (N,3)

d_src_flat = (R @ d_flat.T).T                  # (N,3)  — rotation 적용
d_src      = d_src_flat.reshape(FACE_SIZE, FACE_SIZE, 3)

u_src, v_src = unit_to_erp_pixel(d_src, erp_w, erp_h)   # (512,512)

# 촬영자 마스크 체크 (source ERP에서)
ui_src = np.clip(u_src.astype(int), 0, erp_w - 1)
vi_src = np.clip(v_src.astype(int), 0, erp_h - 1)
is_person_src = src_erp_pm[vi_src, ui_src]                # (512,512) bool

# target down face의 촬영자 마스크 & source가 clean한 픽셀
target_pm_np   = target_pm.cpu().numpy()                   # (512,512)
fill_candidate = target_pm_np & ~is_person_src             # 채울 수 있는 픽셀
clean_coverage = fill_candidate.sum() / max(mask_total, 1)
print(f"\n재투영 결과:")
print(f"  target mask:       {mask_total}px")
print(f"  source clean:      {(~is_person_src).sum()}px / {FACE_SIZE**2}px")
print(f"  fill candidate:    {fill_candidate.sum()}px")
print(f"  clean coverage:    {clean_coverage:.3f}")

# ═════════════════════════════════════════════════════════════════════════
# Step 6: bilinear 샘플링 → 복원
# ═════════════════════════════════════════════════════════════════════════
grid_u = torch.from_numpy((u_src / (erp_w - 1) * 2 - 1).astype(np.float32))  # (512,512)
grid_v = torch.from_numpy((v_src / (erp_h - 1) * 2 - 1).astype(np.float32))
grid   = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)  # (1,512,512,2)

sampled = F.grid_sample(
    src_erp.unsqueeze(0).to(DEVICE),
    grid.to(DEVICE),
    mode="bilinear", align_corners=True, padding_mode="border",
).squeeze(0)   # (3,512,512)

# 촬영자 영역 제거 + target에 합성
fill_mask  = torch.from_numpy(fill_candidate).to(DEVICE)    # (512,512) bool
result_down = target_down.clone()
if fill_mask.any():
    result_down[:, fill_mask] = sampled[:, fill_mask]

filled_r = float(fill_mask.sum()) / max(mask_total, 1)
print(f"  실제 filled_ratio: {filled_r:.3f}")

# ═════════════════════════════════════════════════════════════════════════
# Step 7: 시각화
# ═════════════════════════════════════════════════════════════════════════

def t2bgr(t):
    arr = t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    return cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def draw_contour(bgr, mask_np, color=(0,255,255), thick=2):
    vis  = bgr.copy()
    cnts, _ = cv2.findContours(
        (mask_np.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis

def label(img, *lines):
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (6, 22+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 2)
    return img

# ── 패널 A: target down 원본 + 복원 결과 나란히 ──────────────────────────
vis_orig   = draw_contour(t2bgr(target_down), target_pm_np)
label(vis_orig, "target down (original)")

vis_result = draw_contour(t2bgr(result_down), target_pm_np)
# 미채움 영역 파란색
uncov      = target_pm_np & ~fill_candidate
vis_result[uncov] = [50, 50, 200]
label(vis_result,
      f"pose-based fill",
      f"filled={filled_r:.3f}  uncov={uncov.sum()/max(mask_total,1):.3f}")

# ── 패널 B: source ERP 어디에서 샘플링됐는지 시각화 ────────────────────────
VIZ_H = 360
VIZ_W = int(erp_w / erp_h * VIZ_H)   # 720
th_viz  = cv2.resize(t2bgr(target_erp),  (VIZ_W, VIZ_H))
src_viz = cv2.resize(t2bgr(src_erp),     (VIZ_W, VIZ_H))

# target: down face 영역 하이라이트 (반투명 노란색)
down_region_y = int(VIZ_H * 0.75)
overlay_t  = th_viz.copy()
overlay_t[down_region_y:] = cv2.addWeighted(
    th_viz[down_region_y:], 0.5,
    np.full_like(th_viz[down_region_y:], [0, 255, 255]), 0.5, 0)
cv2.line(overlay_t, (0, down_region_y), (VIZ_W, down_region_y), (0,255,255), 2)
cv2.putText(overlay_t, "target ERP (down region highlighted)", (6,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

# source: fill_candidate 픽셀이 어느 ERP 위치에서 왔는지 녹색 점 표시
src_viz_copy = src_viz.copy()
ui_viz = (ui_src[fill_candidate] * VIZ_W / erp_w).astype(int)
vi_viz = (vi_src[fill_candidate] * VIZ_H / erp_h).astype(int)
sample_size = min(len(ui_viz), 3000)
for uu, vv in zip(ui_viz[:sample_size:max(1,len(ui_viz)//sample_size)],
                  vi_viz[:sample_size:max(1,len(vi_viz)//sample_size)]):
    cv2.circle(src_viz_copy, (int(uu), int(vv)), 2, (0, 255, 0), -1)
cv2.putText(src_viz_copy, f"source ERP (green=sampled clean px)", (6,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ── SIFT match 라인 (down face 영역 필터링) ──────────────────────────────
match_vis = np.hstack([th_viz.copy(), src_viz.copy()])
down_kp_mask = (pts0_eq[:, 1] / erp_h) > 0.75
n_drawn = 0
for (u0, v0), (u1, v1) in zip(pts0_eq[down_kp_mask], pts1_eq[down_kp_mask]):
    p0 = (int(u0 * VIZ_W / erp_w), int(v0 * VIZ_H / erp_h))
    p1 = (int(u1 * VIZ_W / erp_w + VIZ_W), int(v1 * VIZ_H / erp_h))
    cv2.line(match_vis, p0, p1, (0,255,100), 1)
    cv2.circle(match_vis, p0, 3, (0,255,255), -1)
    cv2.circle(match_vis, p1, 3, (0,255,100), -1)
    n_drawn += 1
cv2.putText(match_vis, f"SIFT matches in down region ({n_drawn})", (6,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
print(f"  down 영역 SIFT match: {n_drawn}개")

# ── 최종 합성 ────────────────────────────────────────────────────────────
row1 = np.hstack([vis_orig, vis_result,
                  np.zeros((FACE_SIZE, FACE_SIZE, 3), np.uint8)])

# ERP 패널: 512 높이로 맞춤
erp_h_p = FACE_SIZE
erp_w_p = int(VIZ_W * erp_h_p / VIZ_H)
th_p    = cv2.resize(overlay_t,   (erp_w_p, erp_h_p))
src_p   = cv2.resize(src_viz_copy,(erp_w_p, erp_h_p))
match_p = cv2.resize(match_vis,   (erp_w_p * 2, erp_h_p))

row2 = np.hstack([th_p, src_p])   # 각각 erp_w_p 폭
row3 = match_p

target_w = row1.shape[1]
def pad_or_crop(img, W):
    h, w = img.shape[:2]
    if w < W:
        return np.hstack([img, np.zeros((h, W-w, 3), np.uint8)])
    return img[:, :W]

row2 = pad_or_crop(row2, target_w)
row3 = pad_or_crop(row3, target_w)

out_img  = np.vstack([row1, row2, row3])
out_path = DEBUG_OUT / "pose_based_down_fill.jpg"
cv2.imwrite(str(out_path), out_img)
print(f"\n저장: {out_path}  ({out_path.stat().st_size // 1024} KB)")
