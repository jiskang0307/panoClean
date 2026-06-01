"""
image_io.py — ERP 이미지 로드/저장 및 배치 입출력 유틸리티.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_erp(path: str | Path) -> np.ndarray:
    """
    ERP 이미지를 BGR ndarray로 로드.

    Args:
        path: 이미지 파일 경로.

    Returns:
        HxWx3 uint8 BGR ndarray.

    Raises:
        FileNotFoundError: 파일이 없는 경우.
        ValueError: 디코딩 실패 또는 2:1 종횡비가 아닌 경우.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"이미지 없음: {p}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 디코딩 실패: {p}")

    h, w = img.shape[:2]
    if abs(w / h - 2.0) > 0.1:
        logger.warning(f"ERP 종횡비 비정상 ({w}x{h}) — 계속 진행합니다: {p.name}")

    return img


def save_erp(image: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """
    ERP 이미지를 파일로 저장.

    Args:
        image  : HxWx3 uint8 BGR ndarray.
        path   : 저장 경로 (디렉토리가 없으면 자동 생성).
        quality: JPEG 품질 (0~100).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - quality) // 10))]
    else:
        params = []

    success = cv2.imwrite(str(p), image, params)
    if not success:
        raise IOError(f"이미지 저장 실패: {p}")

    logger.debug(f"저장 완료: {p}")


def collect_images(directory: str | Path) -> list[Path]:
    """디렉토리에서 지원 포맷 이미지 경로 목록을 반환 (정렬)."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"유효하지 않은 디렉토리: {d}")

    paths = sorted(
        p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    logger.info(f"이미지 {len(paths)}장 발견: {d}")
    return paths


def batch_collect(
    directory: str | Path, batch_size: int = 4
) -> list[list[Path]]:
    """이미지 경로 목록을 batch_size 단위 청크로 분할."""
    paths = collect_images(directory)
    return [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
