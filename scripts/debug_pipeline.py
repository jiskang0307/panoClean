"""
debug_pipeline.py — 파이프라인 중간 결과물 디버그 저장.

실행:
    python scripts/debug_pipeline.py
    python scripts/debug_pipeline.py path/to/target.jpg
    python scripts/debug_pipeline.py path/to/target.jpg --sources src1.jpg src2.jpg

기본값: img/ 폴더의 첫 번째 이미지를 target, 나머지를 source로 사용.
"""

from __future__ import annotations

import argparse
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


# ── 헬퍼 ──────────────────────────────────────────────────────────────────

def save(img_bgr: np.ndarray, name: str) -> None:
    cv2.imwrite(str(DEBUG_OUT / name), img_bgr)

def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def _erp_y(box: np.ndarray, face_size: int, erp_h: int, face_name: str) -> float | None:
    if face_name in ("up", "down"):
        return None
    cy = (box[1] + box[3]) / 2.0
    return erp_h * 0.25 + cy * (erp_h * 0.5 / face_size)

def _draw_matches_vis(img0: np.ndarray, img1: np.ndarray,
                      pts0: np.ndarray, pts1: np.ndarray,
                      inlier_mask: np.ndarray | None = None,
                      max_lines: int = 300) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    out = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    out[:h0, :w0] = img0
    out[:h1, w0:] = img1
    n = min(len(pts0), max_lines)
    for i in range(n):
        is_inlier = inlier_mask is None or bool(inlier_mask[i])
        color = (0, 200, 0) if is_inlier else (0, 0, 180)
        p0 = (int(pts0[i][0]), int(pts0[i][1]))
        p1 = (int(pts1[i][0]) + w0, int(pts1[i][1]))
        cv2.line(out, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(out, p0, 3, color, -1)
        cv2.circle(out, p1, 3, color, -1)
    label = f"matches={len(pts0)}"
    if inlier_mask is not None:
        label += f"  inliers={int(inlier_mask.sum())}"
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return out

def _draw_contours(bgr: np.ndarray, mask_cpu: torch.Tensor,
                   color=(0, 0, 255), thick=2) -> np.ndarray:
    vis = bgr.copy()
    m = mask_cpu.numpy().astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, thick)
    return vis


# ── 인수 파싱 ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("target", nargs="?")
parser.add_argument("--sources", nargs="*")
cli = parser.parse_args()

img_dir   = ROOT / "img"
img_paths = sorted(img_dir.glob("*.jpg"))

target_path  = Path(cli.target) if cli.target else (img_paths[0] if img_paths else None)
if target_path is None:
    print("이미지 없음. img/ 폴더에 ERP 이미지를 넣거나 경로를 인수로 전달하세요.")
    sys.exit(1)

source_paths = [Path(p) for p in cli.sources] if cli.sources else [
    p for p in img_paths if p.resolve() != target_path.resolve()
]

print(f"타깃  : {target_path.name}")
print(f"소스  : {[p.name for p in source_paths]}")
print(f"디바이스: {DEVICE}  |  face_size={FACE_SIZE}")
print(f"출력  : {DEBUG_OUT}/")


# ── 모듈 로드 ─────────────────────────────────────────────────────────────

from pipeline.cubemap import CubeMapConverter, load_erp, save_erp
from pipeline.inpainting import LamaInpainter
from pipeline.segmentation import PersonSegmenter, PersonRole
from pipeline.matching import BackgroundMatcher, POLAR_FACES

cfg = {
    "device":                  DEVICE,
    "yolo_model":              "yolo11x-seg.pt",
    "yolo_conf":               0.4,
    "mask_dilate_px":          15,
    "photographer_y_ratio":    0.40,
    "photographer_size_weight": 0.5,
    "feature_matcher":         "sift",
    "min_match_count":         10,
    "min_coverage_ratio":      0.5,
}

conv      = CubeMapConverter(face_size=FACE_SIZE, device=DEVICE)
segmenter = PersonSegmenter(cfg)
matcher   = BackgroundMatcher(cfg)
inpainter = LamaInpainter(device=DEVICE, debug_dir=DEBUG_OUT)


# ── ERP 로드 + CubeMap 변환 ───────────────────────────────────────────────

erp0 = load_erp(str(target_path))
_, erp_h, erp_w = erp0.shape
faces = conv.erp_to_cubemap(erp0)

for fname, face_t in faces.items():
    save(t2bgr(face_t), f"face_{fname}.jpg")

src_items: list[tuple[str, dict[str, torch.Tensor]]] = [
    (sp.name, conv.erp_to_cubemap(load_erp(str(sp)))) for sp in source_paths
]


# ══════════════════════════════════════════════════════════════════════════
# face 단위 처리 — 이미지 저장 + 정보 수집
# ══════════════════════════════════════════════════════════════════════════

result_faces: dict[str, torch.Tensor] = {}

for fname, face_t in faces.items():
    fh = fw = FACE_SIZE

    # ── YOLO 검출 ─────────────────────────────────────────────────────────
    img_bgr = t2bgr(face_t)
    dets    = segmenter._yolo_detect(img_bgr, fh, fw)

    yolo_vis = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box.astype(int)
        cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(yolo_vis, f"{d.conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    save(yolo_vis, f"yolo_{fname}.jpg")

    # ── 역할 분류 ─────────────────────────────────────────────────────────
    seg = segmenter.segment_face(face_t, fname, erp_h, erp_w)
    save(segmenter.visualize_classification(face_t, seg), f"seg_{fname}.jpg")

    photo_mask = seg["photographer_mask"]

    # ── Feature Matching ──────────────────────────────────────────────────
    match_infos: list[dict] = []   # {src_name, n_matched, n_inlier, inlier_ratio}

    for src_idx, (src_name, src_faces) in enumerate(src_items):
        src_t = src_faces[fname]
        force_sift = fname in POLAR_FACES
        pts0, pts1 = matcher._match_keypoints(src_t, face_t, force_sift=force_sift)

        if pts0 is None or len(pts0) < 4:
            save(np.hstack([t2bgr(src_t), img_bgr]),
                 f"match_{fname}_src{src_idx}.jpg")
            match_infos.append({"src_name": src_name,
                                 "n_matched": 0, "n_inlier": 0, "inlier_ratio": 0.0})
            continue

        H_cv, inl = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
        n_inlier     = int(inl.sum()) if inl is not None else 0
        inlier_ratio = n_inlier / len(pts0)

        vis = _draw_matches_vis(t2bgr(src_t), img_bgr, pts0, pts1,
                                inl.ravel() if inl is not None else None)
        save(vis, f"match_{fname}_src{src_idx}.jpg")
        match_infos.append({"src_name": src_name,
                             "n_matched": len(pts0),
                             "n_inlier": n_inlier,
                             "inlier_ratio": inlier_ratio})

    # best_sources (파이프라인용, photographer_mask 기준)
    src_face_ts = [sf[fname] for _, sf in src_items]
    best_sources = (
        matcher.select_best_sources(face_t, src_face_ts, photo_mask,
                                    top_k=3, face_name=fname)
        if src_face_ts and photo_mask.any() else []
    )

    # ── 배경 교체 ─────────────────────────────────────────────────────────
    coverage = 0.0
    if photo_mask.any() and best_sources:
        restored = matcher.blend_multiple_sources(
            face_t, photo_mask, best_sources, face_name=fname
        )
        diff     = (restored.cpu() - face_t.cpu()).abs().sum(0)
        filled   = photo_mask & (diff > 0.02)
        coverage = float(filled.sum()) / max(int(photo_mask.sum()), 1)

        before = _draw_contours(t2bgr(face_t),   photo_mask.cpu(), (0, 0, 255))
        after  = _draw_contours(t2bgr(restored), photo_mask.cpu(), (0, 255, 0))
        cv2.putText(before, "BEFORE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(after,  "AFTER",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        save(np.hstack([before, after]), f"replace_before_after_{fname}.jpg")
    else:
        restored = face_t

    # ── LaMa inpainting ───────────────────────────────────────────────────
    if photo_mask.any() and inpainter.available:
        filled_mask = photo_mask & ((restored.cpu() - face_t.cpu()).abs().sum(0) > 0.02)
        inpainted, did_inpaint = inpainter.inpaint_residual(
            restored, photo_mask, filled_mask, face_name=fname
        )
        print(f"  - inpainting: did={did_inpaint}, coverage_before={coverage:.2f}")
        result_faces[fname] = inpainted
    else:
        result_faces[fname] = restored

    # ══ 콘솔 출력 (face 단위) ═════════════════════════════════════════════
    print(f"\n[{fname}]")
    print(f"  - 검출 인원: {len(dets)}명")

    for det, role, score in zip(seg["detections"], seg["roles"], seg["role_scores"]):
        x1, y1, x2, y2 = det.box.astype(int)
        bw, bh = x2 - x1, y2 - y1
        ey = _erp_y(det.box, FACE_SIZE, erp_h, fname)
        ey_str   = f"{ey:.2f}" if ey is not None else "N/A"
        role_str = "PHOTOGRAPHER" if role == PersonRole.PHOTOGRAPHER else "BACKGROUND"
        print(f"  - {role_str}: bbox=({x1},{y1},{bw},{bh}), "
              f"mask_px={det.pixel_count}, erp_y_computed={ey_str}, score={score:.1f}")

    if not seg["detections"]:
        print("  - (검출 없음)")

    for mi in match_infos:
        print(f"  - matching: source={mi['src_name']}, "
              f"inlier_ratio={mi['inlier_ratio']:.2f}, "
              f"matched_keypoints={mi['n_matched']}")

    if not match_infos:
        print("  - matching: 소스 없음")

    print(f"  - coverage_before_inpaint: {coverage:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# Step 6: 최종 ERP 저장
# ══════════════════════════════════════════════════════════════════════════

result_erp = conv.cubemap_to_erp(result_faces, erp_h, erp_w)
save_erp(result_erp, str(DEBUG_OUT / "result_erp.jpg"))

print(f"\n{'─'*55}")
print(f"완료: {DEBUG_OUT}/")
saved = sorted(DEBUG_OUT.glob("*.jpg"))
print(f"저장된 파일 {len(saved)}개:")
for p in saved:
    print(f"  {p.name:<45} {p.stat().st_size // 1024:>5} KB")
