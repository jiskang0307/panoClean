"""
visualization.py — 마스크 오버레이, 매칭 결과 시각화 유틸리티.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    이미지에 마스크를 반투명 색상으로 오버레이.

    Args:
        image: HxWx3 uint8 BGR ndarray.
        mask : HxW bool ndarray.
        color: 오버레이 BGR 색상.
        alpha: 투명도 (0=완전 투명, 1=불투명).

    Returns:
        오버레이 결과 이미지.
    """
    overlay = image.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def draw_cubemap_grid(
    faces: dict[str, np.ndarray],
    title: str = "CubeMap Faces",
) -> np.ndarray:
    """
    6개 CubeMap face를 2x3 그리드로 배치한 시각화 이미지 반환.

    배치 순서:
        [front] [right] [back]
        [left ] [top  ] [bottom]
    """
    order = ["front", "right", "back", "left", "top", "bottom"]
    rows: list[np.ndarray] = []

    for i in range(0, 6, 3):
        cols = [faces[name] for name in order[i : i + 3] if name in faces]
        rows.append(np.hstack(cols))

    grid = np.vstack(rows)

    cv2.putText(
        grid, title, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    )
    return grid


def save_comparison(
    original: np.ndarray,
    result: np.ndarray,
    path: str | Path,
    title_left: str = "Original",
    title_right: str = "Cleaned",
) -> None:
    """원본과 결과를 좌우로 나란히 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    for ax, img, title in zip(
        axes, [original, result], [title_left, title_right]
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    max_display: int = 100,
) -> np.ndarray:
    """두 이미지의 feature match를 선으로 연결해 반환."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    n = min(len(pts1), max_display)
    for p1, p2 in zip(pts1[:n], pts2[:n]):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]) + w1, int(p2[1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 3, (255, 0, 0), -1)

    return canvas
