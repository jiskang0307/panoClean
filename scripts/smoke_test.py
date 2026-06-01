"""
smoke_test.py — YOLO/SAM2 모델 없이 Phase 2·4 핵심 기능 검증.

실행:
    python scripts/smoke_test.py

필요 패키지:
    torch, equilib (또는 py360convert), opencv-python, kornia (선택)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────

def ok(msg: str):   print(f"\033[92m  ✔\033[0m  {msg}")
def fail(msg: str): print(f"\033[91m  ✘\033[0m  {msg}"); sys.exit(1)
def section(t: str): print(f"\n\033[1m{'─'*55}\n  {t}\n{'─'*55}\033[0m")

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)

def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    return cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: CUDA 확인
# ═══════════════════════════════════════════════════════════════════════════

section("Step 1: CUDA / torch 확인")
try:
    import torch
    ok(f"torch {torch.__version__}")
except ImportError:
    fail("torch 미설치 → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    ok(f"GPU: {torch.cuda.get_device_name(0)}  "
       f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("  ⚠  CUDA 불가 — CPU 모드로 실행합니다.")


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: CubeMap 변환 (실제 이미지)
# ═══════════════════════════════════════════════════════════════════════════

section("Step 2: ERP → CubeMap → ERP 라운드트립")

from pipeline.cubemap import CubeMapConverter, load_erp, save_erp

img_dir = ROOT / "img"
img_paths = sorted(img_dir.glob("*.jpg"))
if not img_paths:
    fail(f"이미지 없음: {img_dir}")

ok(f"이미지 {len(img_paths)}장 발견")

FACE_SIZE = 512   # smoke test는 512로 빠르게
conv = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)

# 첫 이미지로 라운드트립 테스트
t0 = time.time()
erp0 = load_erp(str(img_paths[0]))
ok(f"ERP 로드: {erp0.shape}  ({time.time()-t0:.2f}s)")

t1 = time.time()
faces = conv.erp_to_cubemap(erp0)
ok(f"ERP→CubeMap: {len(faces)}면, 각 {next(iter(faces.values())).shape}  ({time.time()-t1:.2f}s)")

t2 = time.time()
erp_restored = conv.cubemap_to_erp(faces, erp0.shape[1], erp0.shape[2])
ok(f"CubeMap→ERP: {erp_restored.shape}  ({time.time()-t2:.2f}s)")

# PSNR 계산
orig_np = (erp0.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
rest_np = (erp_restored.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
p = psnr(orig_np, rest_np)
if p >= 30.0:
    ok(f"라운드트립 PSNR: {p:.1f} dB ✓")
else:
    print(f"  ⚠  PSNR {p:.1f} dB — 30 dB 미만 (face_size가 작으면 낮을 수 있음)")

# face 시각화 저장
grid_rows = []
for row_names in [["front","right","back"], ["left","up","down"]]:
    row = np.hstack([t2bgr(faces[n]) for n in row_names])
    grid_rows.append(row)
cubemap_grid = np.vstack(grid_rows)
cv2.imwrite(str(OUTPUT / "smoke_cubemap_faces.jpg"), cubemap_grid)
ok(f"face grid 저장: output/smoke_cubemap_faces.jpg")

# 원본 vs 복원 비교 저장
orig_small = cv2.resize(t2bgr(erp0),         (960, 480))
rest_small = cv2.resize(t2bgr(erp_restored),  (960, 480))
cv2.putText(orig_small, "Original",  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.putText(rest_small, "Restored",  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
compare = np.hstack([orig_small, rest_small])
cv2.imwrite(str(OUTPUT / "smoke_roundtrip.jpg"), compare)
ok(f"비교 이미지 저장: output/smoke_roundtrip.jpg")


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: SIFT 매칭 + warp (모델 불필요)
# ═══════════════════════════════════════════════════════════════════════════

section("Step 3: SIFT Feature Matching (두 이미지 간)")

if len(img_paths) < 2:
    print("  ⚠  이미지 2장 이상 필요 — 매칭 테스트 skip")
else:
    from pipeline.matching import BackgroundMatcher

    cfg = {
        "device": DEVICE,
        "feature_matcher": "sift",
        "min_match_count": 10,
        "min_coverage_ratio": 0.5,
    }
    matcher = BackgroundMatcher(cfg)

    erp1   = load_erp(str(img_paths[1]))
    faces1 = conv.erp_to_cubemap(erp1)

    for face_name in ["front", "right"]:
        t_start = time.time()
        H_mat, inlier_ratio = matcher.find_homography(
            faces1[face_name], faces[face_name], face_name=face_name
        )
        elapsed = time.time() - t_start

        if H_mat is not None:
            ok(f"[{face_name}] homography 추정 성공  inlier={inlier_ratio:.2f}  ({elapsed:.2f}s)")
        else:
            print(f"  ⚠  [{face_name}] homography 실패 — 이미지 변화가 너무 크거나 매칭 불충분")

    # dummy mask로 warp 테스트
    h = w = FACE_SIZE
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[h//4 : h*3//4, w//4 : w*3//4] = True   # 중앙 50%

    H_mat, _ = matcher.find_homography(faces1["front"], faces["front"], "front")
    if H_mat is not None:
        warped = matcher.warp_background(faces1["front"], mask, H_mat)
        ok(f"warp_background: {warped.shape}, range=[{warped.min():.3f}, {warped.max():.3f}]")

        # coverage 계산
        valid = mask & (warped.sum(0) > 0.01)
        coverage = float(valid.sum()) / float(mask.sum())
        ok(f"dummy mask coverage: {coverage:.3f}")

        # warp 결과 저장
        warp_vis = t2bgr(warped)
        mask_vis = (mask.numpy().astype(np.uint8) * 128)
        mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(OUTPUT / "smoke_warp.jpg"), np.hstack([t2bgr(faces["front"]), warp_vis]))
        ok("warp 시각화 저장: output/smoke_warp.jpg")


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: 고정 mask로 블렌딩 테스트
# ═══════════════════════════════════════════════════════════════════════════

section("Step 4: 배경 블렌딩 (dummy mask)")

if len(img_paths) >= 2:
    # front face에서 중앙 영역을 소스 이미지로 교체
    best_sources = matcher.select_best_sources(
        faces["front"],
        [faces1["front"]],
        mask,
        top_k=1,
        face_name="front",
    )

    if best_sources:
        restored = matcher.blend_multiple_sources(
            faces["front"], mask, best_sources, face_name="front"
        )
        ok(f"blend_multiple_sources 완료: {restored.shape}")

        # 결과를 ERP로 재합성
        result_faces = dict(faces)
        result_faces["front"] = restored
        result_erp = conv.cubemap_to_erp(result_faces, erp0.shape[1], erp0.shape[2])
        save_erp(result_erp, str(OUTPUT / "smoke_blend_result.jpg"))
        ok("블렌딩 결과 저장: output/smoke_blend_result.jpg")
    else:
        print("  ⚠  소스 없음 — 블렌딩 skip")


# ═══════════════════════════════════════════════════════════════════════════
# 완료
# ═══════════════════════════════════════════════════════════════════════════

section("완료")
print(f"  출력 파일: {OUTPUT}/")
for f in sorted(OUTPUT.glob("smoke_*.jpg")):
    print(f"    {f.name}")
print()
ok("smoke test 통과 — 모델 설치 후 전체 파이프라인 실행 가능")
