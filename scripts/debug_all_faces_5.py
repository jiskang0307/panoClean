"""
debug_all_faces_5.py — 5개 문제 이미지에 대해
  all_faces_{img_id}.jpg  : 6개 face 2×3 그리드 (PHOTOGRAPHER=파랑, BACKGROUND=빨강 오버레이)
  erp_bottom_{img_id}.jpg : 처리 후 ERP 하단 40% 크롭
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

from pipeline.cubemap import CubeMapConverter, FACE_NAMES, load_erp
from pipeline.segmentation import PersonSegmenter, FaceMosaicker
from pipeline.inpainting import LamaInpainter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THUMB  = 512   # 썸네일 크기

IMG_IDS = [
    "1779263690.762159",
    "1779263615.694478",
    "1779263604.684549",
    "1779263644.720649",
    "1779263677.750428",
]

cfg = {
    "device": DEVICE, "yolo_model": "yolo11x-seg.pt",
    "yolo_conf": 0.4, "mask_dilate_px": 15,
}
conv = CubeMapConverter(face_size=1024, device=DEVICE)
seg  = PersonSegmenter(cfg)
inp  = LamaInpainter(device=DEVICE,
                     down_blur_kernel=251, down_blur_feather=101, down_blur_passes=2)
mos  = FaceMosaicker()


def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def overlay_mask(bgr: np.ndarray, mask_np: np.ndarray,
                 color: tuple, alpha: float = 0.5) -> np.ndarray:
    vis = bgr.copy()
    where = mask_np > 0
    vis[where] = (
        bgr[where].astype("float32") * (1 - alpha)
        + np.array(color, dtype="float32") * alpha
    ).clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, 2)
    return vis


for img_id in IMG_IDS:
    p = ROOT / "input" / f"{img_id}.jpg"
    if not p.exists():
        print(f"파일 없음: {p}")
        continue

    erp_t = load_erp(str(p))
    _, eh, ew = erp_t.shape
    faces = conv.erp_to_cubemap(erp_t)
    seg_results = seg.segment_all_faces(faces, eh, ew)

    panels = []
    result_faces: dict[str, torch.Tensor] = {}

    for fn in FACE_NAMES:
        face_bgr = t2bgr(faces[fn])
        res      = seg_results[fn]

        # PHOTOGRAPHER mask — 파란 오버레이
        pm_np = res["photographer_mask"].cpu().numpy().astype(np.uint8) * 255
        vis   = overlay_mask(face_bgr, pm_np, color=(255, 60, 0), alpha=0.5)  # BGR 파랑

        # BACKGROUND masks — 빨간 오버레이
        for bg_mask in res.get("background_masks", []):
            bg_np = bg_mask.cpu().numpy().astype(np.uint8) * 255
            vis   = overlay_mask(vis, bg_np, color=(0, 0, 220), alpha=0.35)  # BGR 빨강

        # face 이름 + 검출 수
        n_p = sum(1 for r in res["roles"] if r.value == "photographer")
        n_b = sum(1 for r in res["roles"] if r.value == "background")
        label = f"{fn}  P={n_p} B={n_b}  px={int(res['photographer_mask'].sum())}"
        cv2.putText(vis, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 썸네일 리사이즈
        thumb = cv2.resize(vis, (THUMB, THUMB))
        panels.append(thumb)

        # 처리
        fi = mos.apply_background_mosaics(faces[fn], res)
        pm = res["photographer_mask"].to(DEVICE)
        if pm.any():
            fi = inp.blur_face(fi, pm, fn)
            print(f"  [{fn}] blur  px={int(pm.sum())}")
        result_faces[fn] = fi

    # 2×3 그리드 (front right back / left up down)
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])
    grid = np.vstack([row1, row2])

    out1 = DEBUG_OUT / f"all_faces_{img_id}.jpg"
    cv2.imwrite(str(out1), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"저장: {out1}")

    # ERP 재합성 + 하단 40% 크롭
    result_erp = conv.cubemap_to_erp(result_faces, eh, ew)
    erp_bgr = (result_erp.cpu().permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8)
    bottom  = erp_bgr[int(eh * 0.6):, :, :]

    out2 = DEBUG_OUT / f"erp_bottom_{img_id}.jpg"
    cv2.imwrite(str(out2), bottom, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"저장: {out2}")
    print()
