"""
batch_runner.py — 360° 이미지 사람 제거 파이프라인 배치 실행 진입점.

사용법:
    python batch_runner.py --input ./input --output ./output --config config/default.yaml

전체 파이프라인:
  1. ERP → CubeMap 변환
  2. 각 face에서 사람 마스크 생성 (YOLO11-seg / SAM2)
  3. 다른 이미지의 동일 face에서 feature matching으로 배경 복원
  4. 잔여 영역 LaMa inpainting
  5. CubeMap → ERP 재합성
  6. 결과 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

from pipeline.blending import PatchBlender
from pipeline.cubemap import CubeMapConverter
from pipeline.inpainting import LamaInpainter
from pipeline.matching import BackgroundMatcher
from pipeline.segmentation import PersonSegmentor
from utils.image_io import batch_collect, load_erp, save_erp
from utils.visualization import save_comparison


# ── CLI 파서 ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="360° ERP 이미지에서 사람을 배경으로 대체합니다."
    )
    p.add_argument("--input", default="./input", help="입력 디렉토리")
    p.add_argument("--output", default="./output", help="출력 디렉토리")
    p.add_argument("--config", default="config/default.yaml", help="설정 파일 경로")
    p.add_argument("--save-comparison", action="store_true", help="원본/결과 비교 이미지 저장")
    p.add_argument("--debug", action="store_true", help="디버그 로그 활성화")
    return p


# ── 설정 로드 ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 파이프라인 ────────────────────────────────────────────────────────────

def process_batch(
    batch_paths: list[Path],
    cfg: dict,
    converter: CubeMapConverter,
    segmentor: PersonSegmentor,
    matcher: BackgroundMatcher,
    inpainter: LamaInpainter,
    blender: PatchBlender,
    output_dir: Path,
    save_cmp: bool,
) -> None:
    """하나의 배치(동일 공간을 촬영한 이미지 묶음)를 처리."""

    # 1) 모든 이미지 로드 및 CubeMap 변환
    all_erp = [load_erp(p) for p in batch_paths]
    all_faces = [converter.erp_to_cubemap(img) for img in all_erp]

    for img_idx, (erp_path, target_erp, target_faces) in enumerate(
        zip(batch_paths, all_erp, all_faces)
    ):
        logger.info(f"[{img_idx + 1}/{len(batch_paths)}] {erp_path.name} 처리 중")

        # 소스 후보: 같은 배치의 다른 이미지들
        source_faces_list = [
            f for i, f in enumerate(all_faces) if i != img_idx
        ]

        result_faces: dict[str, object] = {}
        filled_masks: dict[str, object] = {}

        # 2) face별 처리
        for face_name in ["front", "right", "back", "left", "top", "bottom"]:
            target_face = target_faces[face_name]

            # 2-a) 마스크 생성
            person_mask = segmentor.segment(target_face)
            if not person_mask.any():
                result_faces[face_name] = target_face
                filled_masks[face_name] = person_mask
                continue

            # 2-b) 소스에서 배경 복원
            source_face_candidates = [sf[face_name] for sf in source_faces_list]
            best_warp, coverage = matcher.best_fill(
                target_face, source_face_candidates, person_mask
            )

            if best_warp is not None:
                restored = blender.compose_from_warped(target_face, best_warp, person_mask)
                filled_mask = person_mask & (best_warp.sum(axis=2) > 0)
            else:
                restored = target_face.copy()
                filled_mask = ~person_mask  # 채워진 영역 없음

            # 2-c) 잔여 영역 LaMa inpainting
            if cfg.get("lama_enabled", True):
                restored = inpainter.inpaint_residual(
                    restored, person_mask, filled_mask
                )

            result_faces[face_name] = restored
            filled_masks[face_name] = filled_mask

        # 3) CubeMap → ERP 재합성
        h, w = target_erp.shape[:2]
        result_erp = converter.cubemap_to_erp(result_faces, erp_height=h, erp_width=w)

        # 4) 저장
        out_path = output_dir / erp_path.name
        save_erp(result_erp, out_path)

        if save_cmp:
            cmp_path = output_dir / f"cmp_{erp_path.stem}.png"
            save_comparison(target_erp, result_erp, cmp_path)

        logger.success(f"완료: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    log_level = "DEBUG" if args.debug else cfg.get("log_level", "INFO")
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level, colorize=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 초기화 ────────────────────────────────────────────────────────
    device = cfg.get("device", "cuda")

    converter = CubeMapConverter(
        face_size=cfg.get("cubemap_face_size", 1024),
        device=device,
    )
    segmentor = PersonSegmentor(
        yolo_model_path=cfg.get("yolo_model", "yolo11x-seg.pt"),
        sam2_model_path=cfg.get("sam2_model"),
        sam2_config=cfg.get("sam2_config", "sam2_hiera_l.yaml"),
        person_class_id=cfg.get("person_class_id", 0),
        mask_dilate_px=cfg.get("mask_dilate_px", 15),
        device=device,
        use_sam2=bool(cfg.get("sam2_model")),
    )
    matcher = BackgroundMatcher(
        matcher_type=cfg.get("feature_matcher", "loftr"),
        min_coverage_ratio=cfg.get("min_coverage_ratio", 0.85),
        device=device,
    )
    inpainter = LamaInpainter(device=device)
    blender = PatchBlender(mode="poisson")

    # ── 배치 처리 ──────────────────────────────────────────────────────────
    batches = batch_collect(args.input, batch_size=cfg.get("batch_size", 4))
    logger.info(f"총 {sum(len(b) for b in batches)}장 / {len(batches)}개 배치")

    for batch in tqdm(batches, desc="배치 처리"):
        process_batch(
            batch_paths=batch,
            cfg=cfg,
            converter=converter,
            segmentor=segmentor,
            matcher=matcher,
            inpainter=inpainter,
            blender=blender,
            output_dir=output_dir,
            save_cmp=args.save_comparison,
        )

    logger.success("전체 처리 완료.")


if __name__ == "__main__":
    main()
