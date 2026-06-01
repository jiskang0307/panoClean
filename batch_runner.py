"""
batch_runner.py — 360° 이미지 사람 제거 파이프라인 배치 실행 진입점.

사용법:
    python batch_runner.py --input ./input --output ./output --config config/default.yaml

전체 파이프라인:
  1. ERP → CubeMap 변환 (Phase 2)
  2. YOLO11-seg + SAM2 사람 마스크 + 역할 분류 (Phase 3)
  3. 배경 인물 얼굴 모자이크 (Phase 3)
  4. Feature matching으로 실제 배경 대체 (Phase 4)
  5. 잔여 영역 LaMa inpainting (Phase 4)
  6. CubeMap → ERP 재합성 (Phase 2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from pipeline.blending import PatchBlender, SeamlessBlender
from pipeline.cubemap import FACE_NAMES, CubeMapConverter
from pipeline.cubemap import load_erp as load_erp_tensor
from pipeline.cubemap import save_erp as save_erp_tensor
from pipeline.inpainting import LamaInpainter
from pipeline.matching import BackgroundMatcher
from pipeline.segmentation import FaceMosaicker, PersonSegmenter
from utils.image_io import collect_images


# ── CLI 파서 ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="360° ERP 이미지에서 사람을 배경으로 대체합니다."
    )
    p.add_argument("--input",    default="./input",              help="입력 디렉토리")
    p.add_argument("--output",   default="./output",             help="출력 디렉토리")
    p.add_argument("--config",   default="config/default.yaml",  help="설정 파일 경로")
    p.add_argument("--save-comparison", action="store_true",     help="원본/결과 비교 이미지 저장")
    p.add_argument("--debug",    action="store_true",            help="디버그 로그 활성화")
    return p


# ── 설정 로드 ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 배치 처리 ─────────────────────────────────────────────────────────────

def process_batch(
    batch_paths: list[Path],
    cfg: dict,
    converter: CubeMapConverter,
    segmenter: PersonSegmenter,
    matcher: BackgroundMatcher,
    inpainter: LamaInpainter,
    mosaicker: FaceMosaicker,
    output_dir: Path,
    save_cmp: bool,
) -> None:
    """하나의 배치(동일 공간을 촬영한 이미지 묶음)를 처리."""

    # 1) 모든 이미지 로드 + CubeMap 변환
    all_erps: list[torch.Tensor] = [load_erp_tensor(str(p)) for p in batch_paths]
    erp_h, erp_w = all_erps[0].shape[1], all_erps[0].shape[2]
    logger.info(f"ERP 해상도: {erp_w}x{erp_h}")

    all_faces: list[dict[str, torch.Tensor]] = [
        converter.erp_to_cubemap(erp) for erp in all_erps
    ]

    for img_idx, (erp_path, target_erp, target_faces) in enumerate(
        zip(batch_paths, all_erps, all_faces)
    ):
        logger.info(f"[{img_idx + 1}/{len(batch_paths)}] {erp_path.name} 처리 중")
        source_faces_list = [f for i, f in enumerate(all_faces) if i != img_idx]

        try:
            result_faces = _process_single(
                target_faces, source_faces_list, erp_h, erp_w,
                segmenter, matcher, inpainter, mosaicker, cfg,
            )
        except Exception as e:
            logger.error(f"{erp_path.name} 처리 실패: {e} — 원본 유지")
            result_faces = target_faces

        # ERP 재합성 + 저장
        result_erp = converter.cubemap_to_erp(result_faces, erp_height=erp_h, erp_width=erp_w)
        out_path = output_dir / erp_path.name
        save_erp_tensor(result_erp, str(out_path))

        if save_cmp:
            _save_cmp(target_erp, result_erp, output_dir / f"cmp_{erp_path.stem}.png")

        logger.success(f"저장: {out_path}")


def _process_single(
    target_faces: dict[str, torch.Tensor],
    source_faces_list: list[dict],
    erp_h: int,
    erp_w: int,
    segmenter: PersonSegmenter,
    matcher: BackgroundMatcher,
    inpainter: LamaInpainter,
    mosaicker: FaceMosaicker,
    cfg: dict,
) -> dict[str, torch.Tensor]:
    """단일 이미지의 6개 face를 순차 처리."""

    # Segment all faces at once
    seg_results = segmenter.segment_all_faces(target_faces, erp_h, erp_w)

    result_faces: dict[str, torch.Tensor] = {}
    top_k = cfg.get("top_k_sources", 3)
    min_cov = cfg.get("min_coverage_ratio", 0.85)
    lama_on = cfg.get("lama_enabled", True)

    for face_name in FACE_NAMES:
        target_face = target_faces[face_name]
        seg = seg_results[face_name]

        # ── 배경 인물 모자이크 ──────────────────────────────────────────
        face_img = mosaicker.apply_background_mosaics(target_face, seg)

        photo_mask: torch.Tensor = seg["photographer_mask"]
        if not photo_mask.any():
            result_faces[face_name] = face_img
            continue

        # ── 소스 후보 선택 ──────────────────────────────────────────────
        src_candidates = [sf[face_name] for sf in source_faces_list]
        if not src_candidates:
            logger.warning(f"[{face_name}] 소스 이미지 없음 — mask 그대로 유지")
            result_faces[face_name] = face_img
            continue

        best_sources = matcher.select_best_sources(
            face_img, src_candidates, photo_mask,
            top_k=top_k, face_name=face_name,
        )

        # ── 배경 블렌딩 ─────────────────────────────────────────────────
        if best_sources:
            restored = matcher.blend_multiple_sources(
                face_img, photo_mask, best_sources, face_name=face_name
            )
            best_cov = best_sources[0][2]
        else:
            restored = face_img
            best_cov = 0.0

        logger.debug(f"[{face_name}] best_coverage={best_cov:.3f}")

        # ── 잔여 영역 LaMa inpainting ───────────────────────────────────
        if lama_on:
            filled = photo_mask & ((restored - face_img).abs().sum(0) > 0.02)
            coverage = SeamlessBlender.compute_coverage(photo_mask, filled)

            if coverage < min_cov:
                logger.info(f"[{face_name}] LaMa 보완 시작 (coverage={coverage:.3f})")
                restored = _lama_inpaint_tensor(restored, photo_mask, filled, inpainter)

        result_faces[face_name] = restored

    return result_faces


def _lama_inpaint_tensor(
    img: torch.Tensor,
    full_mask: torch.Tensor,
    filled_mask: torch.Tensor,
    inpainter: LamaInpainter,
) -> torch.Tensor:
    """tensor (3,H,W) → LaMa inpainting → tensor."""
    arr = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    result_bgr = inpainter.inpaint_residual(bgr, full_mask.numpy(), filled_mask.numpy())
    rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().permute(2, 0, 1).to(img.device) / 255.0


def _save_cmp(src: torch.Tensor, dst: torch.Tensor, path: Path) -> None:
    from utils.visualization import save_comparison
    def t2bgr(t):
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    save_comparison(t2bgr(src), t2bgr(dst), path)


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()
    cfg  = load_config(args.config)

    log_level = "DEBUG" if args.debug else cfg.get("log_level", "INFO")
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level, colorize=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")

    # ── 모델 초기화 ──────────────────────────────────────────────────────
    converter = CubeMapConverter(
        face_size=cfg.get("cubemap_face_size", 1024),
        device=device,
    )
    segmenter = PersonSegmenter(cfg)
    matcher   = BackgroundMatcher(cfg)
    inpainter = LamaInpainter(device=device)
    mosaicker = FaceMosaicker(
        mosaic_block_size=cfg.get("mosaic_block_size", 20),
        feather_px=cfg.get("mosaic_feather_px", 8),
    )

    # ── 배치 처리 ────────────────────────────────────────────────────────
    all_paths = collect_images(args.input)
    batch_size = cfg.get("batch_size", 4)
    batches = [all_paths[i : i + batch_size] for i in range(0, len(all_paths), batch_size)]
    logger.info(f"총 {len(all_paths)}장 / {len(batches)}배치")

    for batch in tqdm(batches, desc="배치 처리"):
        process_batch(
            batch_paths=batch,
            cfg=cfg,
            converter=converter,
            segmenter=segmenter,
            matcher=matcher,
            inpainter=inpainter,
            mosaicker=mosaicker,
            output_dir=output_dir,
            save_cmp=args.save_comparison,
        )

    logger.success("전체 처리 완료.")


if __name__ == "__main__":
    main()
