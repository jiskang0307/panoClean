"""
batch_runner.py — 360° 이미지 사람 제거 파이프라인 배치 실행 진입점.

사용법:
    # 배치 처리
    python batch_runner.py --input ./input --output ./output

    # 단일 이미지 (인접 파일 자동 선택)
    python batch_runner.py --single input/target.jpg --source-dir ./input

    # 소스 수동 지정
    python batch_runner.py --single target.jpg --sources a.jpg b.jpg c.jpg

    # 이어서 처리 (이미 완료된 파일 스킵)
    python batch_runner.py --input ./input --output ./output --resume
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from loguru import logger
from tqdm import tqdm

from pipeline.cubemap import FACE_NAMES, CubeMapConverter
from pipeline.cubemap import load_erp as load_erp_tensor
from pipeline.cubemap import save_erp as save_erp_tensor
from pipeline.inpainting import LamaInpainter
from pipeline.segmentation import FaceMosaicker, PersonSegmenter
from utils.image_io import collect_images


# ── Ctrl+C 핸들러 ─────────────────────────────────────────────────────────

_SHUTDOWN = False


def _handle_sigint(sig, frame):
    global _SHUTDOWN
    logger.warning("Ctrl+C 감지 — 현재 이미지 완료 후 종료합니다.")
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_sigint)


# ── LRU 큐브맵 캐시 ───────────────────────────────────────────────────────

class _CubeMapCache:
    """최근 N개 이미지의 cubemap을 CPU 텐서로 캐시 (OrderedDict LRU)."""

    def __init__(self, converter: CubeMapConverter, max_size: int = 16) -> None:
        self._conv = converter
        self._max  = max_size
        self._data: OrderedDict[str, tuple[dict[str, torch.Tensor], int, int]] = OrderedDict()

    def get(self, path: Path) -> tuple[dict[str, torch.Tensor], int, int]:
        key = str(path)
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]

        erp = load_erp_tensor(str(path))
        _, erp_h, erp_w = erp.shape
        faces = self._conv.erp_to_cubemap(erp)
        # CPU에 옮겨 저장 (VRAM 절약)
        faces_cpu = {k: v.cpu() for k, v in faces.items()}
        val = (faces_cpu, erp_h, erp_w)

        self._data[key] = val
        if len(self._data) > self._max:
            self._data.popitem(last=False)
        return val


# ── Pipeline360 ───────────────────────────────────────────────────────────

class Pipeline360:
    """360° ERP 이미지 사람 제거 파이프라인."""

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, encoding="utf-8") as f:
            self.cfg: dict = yaml.safe_load(f)

        cfg    = self.cfg
        device = cfg.get("device", "cuda")

        self.converter = CubeMapConverter(
            face_size=cfg.get("cubemap_face_size", 1024),
            device=device,
        )
        self.segmenter = PersonSegmenter(cfg)
        self.inpainter = LamaInpainter(
            device=device,
            down_blur_kernel=cfg.get("down_blur_kernel", 251),
            down_blur_feather=cfg.get("down_blur_feather", 101),
            down_blur_passes=cfg.get("down_blur_passes", 2),
        )
        self.mosaicker = FaceMosaicker(
            mosaic_block_size=cfg.get("mosaic_block_size", 20),
            feather_px=cfg.get("mosaic_feather_px", 8),
        )

        cache_size = cfg.get("n_adjacent", 10) + 6
        self._cache = _CubeMapCache(self.converter, max_size=cache_size)
        self._device = device

    # ── 공개 API ──────────────────────────────────────────────────────────

    def process_single(
        self,
        target_path: str | Path,
        source_paths: list[str | Path] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        단일 이미지 처리 — blur 통일 모드.

        PHOTOGRAPHER 마스크 영역을 face별 Gaussian blur로 처리.
        BackgroundMatcher / LaMa 미사용.

        반환: (result_erp_tensor (3,H,W) float32, stats_dict)
        """
        t_start     = time.monotonic()
        target_path = Path(target_path)

        # 1. target 로드
        target_faces_cpu, erp_h, erp_w = self._cache.get(target_path)
        target_faces = {k: v.to(self._device) for k, v in target_faces_cpu.items()}

        # 2. segmentation
        seg_results = self.segmenter.segment_all_faces(target_faces, erp_h, erp_w)

        # 3. face별 처리
        result_faces: dict[str, torch.Tensor] = {}
        face_stats:   dict[str, dict]         = {}
        total_bg_persons = 0

        for face_name in FACE_NAMES:
            target_face = target_faces[face_name]
            seg         = seg_results[face_name]
            photo_mask  = seg["photographer_mask"].to(self._device)

            total_bg_persons += len(seg.get("background_masks", []))

            # 배경 인물 모자이크
            face_img = self.mosaicker.apply_background_mosaics(target_face, seg)

            has_photo = bool(photo_mask.any())
            if has_photo:
                face_img = self.inpainter.blur_face(face_img, photo_mask, face_name)
                logger.info(f"[{face_name}] blur 적용 완료")

            result_faces[face_name]   = face_img
            face_stats[face_name]     = {"photographer": has_photo}

        # 4. ERP 재합성
        result_erp = self.converter.cubemap_to_erp(result_faces, erp_h, erp_w)

        elapsed = time.monotonic() - t_start
        logger.info(f"{target_path.name} 완료 — {elapsed:.1f}s")

        return result_erp, {
            "target":             target_path.name,
            "faces":              face_stats,
            "background_persons": total_bg_persons,
            "elapsed_sec":        elapsed,
            "coverage_avg":       0.0,
        }

    def _select_sources(
        self,
        all_paths: list[Path],
        target_path: Path,
        n_adjacent: int = 10,
    ) -> list[Path]:
        """target 앞뒤 n_adjacent//2 장씩 인접 파일 선택."""
        sorted_paths = sorted(all_paths)
        idx = next(
            (i for i, p in enumerate(sorted_paths) if p == target_path or p.name == target_path.name),
            -1,
        )
        if idx == -1:
            logger.warning(f"_select_sources: {target_path.name}를 all_paths에서 찾지 못함")
            return []

        half = n_adjacent // 2
        lo   = max(0, idx - half)
        hi   = min(len(sorted_paths), idx + half + 1)
        return [p for p in sorted_paths[lo:hi] if p != target_path and p.exists()]

    def process_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        resume: bool = True,
        save_comparison: bool = False,
    ) -> list[dict]:
        """
        배치 처리 메인 루프.

        - resume=True: 이미 결과 파일이 있으면 스킵
        - 실패한 이미지는 output_dir/failed.txt에 기록
        - Ctrl+C 시 현재까지 처리된 결과 보존
        - 각 이미지 처리 후 torch.cuda.empty_cache()

        반환: 성공한 이미지들의 stats_dict 리스트
        """
        global _SHUTDOWN

        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_paths = collect_images(str(input_dir))

        # resume 필터
        targets: list[Path] = []
        for p in all_paths:
            out_p = output_dir / (p.stem + "_removed" + p.suffix)
            if resume and out_p.exists():
                logger.debug(f"스킵 (이미 처리됨): {p.name}")
            else:
                targets.append(p)

        logger.info(f"처리 대상: {len(targets)}장 / 전체 {len(all_paths)}장")
        if not targets:
            logger.success("처리할 이미지가 없습니다.")
            return []

        results: list[dict] = []
        failed:  list[str]  = []

        pbar = tqdm(targets, desc="처리 중", unit="img")
        for target_path in pbar:
            if _SHUTDOWN:
                logger.warning("중단 요청 — 여기까지 처리된 결과를 보존합니다.")
                break

            pbar.set_description(target_path.name[:40])

            try:
                result_erp, stats = self.process_single(target_path)
                out_path = output_dir / (target_path.stem + "_removed" + target_path.suffix)
                save_erp_tensor(result_erp, str(out_path))

                if save_comparison:
                    _save_comparison(
                        load_erp_tensor(str(target_path)),
                        result_erp,
                        output_dir / f"cmp_{target_path.stem}.jpg",
                    )

                results.append(stats)
                pbar.set_postfix(
                    cov=f"{stats['coverage_avg']:.2f}",
                    t=f"{stats['elapsed_sec']:.1f}s",
                )

            except Exception as exc:
                logger.error(f"{target_path.name} 처리 실패: {exc} — 스킵")
                import traceback
                logger.debug(traceback.format_exc())
                failed.append(target_path.name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if failed:
            failed_path = output_dir / "failed.txt"
            failed_path.write_text("\n".join(failed), encoding="utf-8")
            logger.warning(f"실패 {len(failed)}장 → {failed_path}")

        logger.success(f"배치 완료: {len(results)}장 성공, {len(failed)}장 실패")
        return results


# ── 공용 헬퍼 ─────────────────────────────────────────────────────────────

def _save_comparison(
    src: torch.Tensor,
    dst: torch.Tensor,
    path: Path,
) -> None:
    import cv2
    import numpy as np

    def t2bgr(t: torch.Tensor):
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    panel = np.hstack([t2bgr(src), t2bgr(dst)])
    cv2.imwrite(str(path), panel)


# ── 로깅 초기화 ────────────────────────────────────────────────────────────

def _setup_logging(debug: bool, log_level: str) -> None:
    level = "DEBUG" if debug else log_level
    logger.remove()
    # 콘솔 — 색상 활성화
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> {message}")
    # 파일 — 타임스탬프 포함
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{ts}.log"
    logger.add(str(log_file), level="DEBUG",
               format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.info(f"로그 파일: {log_file}")


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="360° ERP 이미지에서 사람을 배경으로 대체합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",    default="config/default.yaml", help="설정 파일 경로")
    p.add_argument("--resume",    action="store_true",           help="이미 처리된 파일 스킵")
    p.add_argument("--save-comparison", action="store_true",     help="원본/결과 나란히 저장")
    p.add_argument("--debug",     action="store_true",           help="디버그 로그 활성화")
    p.add_argument("--report",    action="store_true",           help="완료 후 HTML 리포트 생성")

    mode = p.add_mutually_exclusive_group(required=True)

    # 배치 처리 모드
    mode.add_argument("--input",  metavar="DIR", help="입력 디렉토리 (배치 처리)")

    # 단일 이미지 모드
    mode.add_argument("--single", metavar="IMG", help="단일 이미지 경로")

    p.add_argument("--output",      default="./output",  help="출력 디렉토리")
    p.add_argument("--source-dir",  metavar="DIR",       help="소스 디렉토리 (--single 전용)")
    p.add_argument("--sources",     nargs="+", metavar="IMG",
                   help="소스 이미지 경로 목록 (--single 전용, --source-dir 대체)")
    p.add_argument("--n-adjacent",  type=int, default=None,
                   help="인접 소스 수 (기본값: config n_adjacent)")
    return p


def main() -> None:
    args   = _build_parser().parse_args()
    cfg_path = args.config

    # 설정 로드 (로깅 레벨 파악용)
    with open(cfg_path, encoding="utf-8") as f:
        cfg_raw: dict = yaml.safe_load(f)

    _setup_logging(args.debug, cfg_raw.get("log_level", "INFO"))

    pipeline = Pipeline360(config_path=cfg_path)

    # --n-adjacent CLI 오버라이드
    if args.n_adjacent is not None:
        pipeline.cfg["n_adjacent"] = args.n_adjacent

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        # ── 배치 처리 ──────────────────────────────────────────────────────
        results = pipeline.process_batch(
            input_dir=args.input,
            output_dir=output_dir,
            resume=args.resume,
            save_comparison=args.save_comparison,
        )

        if args.report and results:
            from evaluate import generate_batch_report, compute_batch_stats
            stats_summary = compute_batch_stats(results)
            report_path   = output_dir / "report.html"
            generate_batch_report(results, str(report_path), stats_summary)
            logger.success(f"리포트: {report_path}")

    else:
        # ── 단일 이미지 처리 ────────────────────────────────────────────────
        target_path = Path(args.single)
        if not target_path.exists():
            logger.error(f"파일 없음: {target_path}")
            sys.exit(1)

        result_erp, stats = pipeline.process_single(target_path)
        out_path = output_dir / (target_path.stem + "_removed" + target_path.suffix)
        save_erp_tensor(result_erp, str(out_path))
        logger.success(f"저장: {out_path}")

        # debug_output/result_erp_final.jpg — nadir 크롭 포함 저장
        import cv2, numpy as np
        from pathlib import Path as _Path
        debug_dir = _Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        erp_bgr = cv2.imread(str(out_path))
        if erp_bgr is not None:
            h = erp_bgr.shape[0]
            cv2.imwrite(str(debug_dir / "result_erp_final.jpg"), erp_bgr)
            cv2.imwrite(str(debug_dir / "result_erp_final_nadir.jpg"),
                        erp_bgr[int(h * 0.75):, :])
            logger.info("debug_output/result_erp_final.jpg 저장 완료")

        if args.save_comparison:
            _save_comparison(
                load_erp_tensor(str(target_path)),
                result_erp,
                output_dir / f"cmp_{target_path.stem}.jpg",
            )


if __name__ == "__main__":
    main()
