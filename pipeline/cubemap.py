"""
cubemap.py — ERP(Equirectangular Projection) ↔ CubeMap 양방향 변환.

equilib 라이브러리를 이용해 CUDA 가속 변환을 수행한다.
6개 face 순서: front, right, back, left, top, bottom
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from equilib import equi2cube, cube2equi
from loguru import logger


FaceName = Literal["front", "right", "back", "left", "top", "bottom"]
FACE_NAMES: list[FaceName] = ["front", "right", "back", "left", "top", "bottom"]


class CubeMapConverter:
    """ERP 이미지를 CubeMap 6-face 텐서로 변환하고 역변환한다."""

    def __init__(self, face_size: int = 1024, device: str = "cuda") -> None:
        self.face_size = face_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if str(self.device) != device:
            logger.warning(f"CUDA 불가 — CPU로 fallback: device={self.device}")

    # ── ERP → CubeMap ──────────────────────────────────────────────────────

    def erp_to_cubemap(
        self, erp: np.ndarray | torch.Tensor
    ) -> dict[FaceName, np.ndarray]:
        """
        ERP 이미지를 6개 CubeMap face로 변환.

        Args:
            erp: HxWx3 uint8 ndarray 또는 3xHxW float32 tensor (0~1).

        Returns:
            face_size x face_size x 3 uint8 ndarray 6개를 담은 딕셔너리.
        """
        tensor = self._to_tensor(erp)  # 1x3xHxW float32 on device

        cubemap_tensor = equi2cube(
            src=tensor,
            rots={"roll": 0, "pitch": 0, "yaw": 0},
            w_face=self.face_size,
            cube_format="dict",
        )

        faces: dict[FaceName, np.ndarray] = {}
        for name in FACE_NAMES:
            face_t = cubemap_tensor[name]  # 3 x H x W
            if face_t.dim() == 4:
                face_t = face_t.squeeze(0)
            faces[name] = self._tensor_to_uint8(face_t)

        return faces

    # ── CubeMap → ERP ──────────────────────────────────────────────────────

    def cubemap_to_erp(
        self,
        faces: dict[FaceName, np.ndarray],
        erp_height: int | None = None,
        erp_width: int | None = None,
    ) -> np.ndarray:
        """
        6개 CubeMap face를 ERP 이미지로 재합성.

        Args:
            faces    : erp_to_cubemap 반환값과 동일한 구조.
            erp_height: 출력 ERP 높이 (기본 face_size * 2).
            erp_width : 출력 ERP 너비 (기본 face_size * 4).

        Returns:
            HxWx3 uint8 ERP ndarray.
        """
        h = erp_height or self.face_size * 2
        w = erp_width or self.face_size * 4

        cube_dict = {
            name: self._uint8_to_tensor(faces[name]).to(self.device)
            for name in FACE_NAMES
        }

        erp_tensor = cube2equi(
            cubemap=cube_dict,
            cube_format="dict",
            height=h,
            width=w,
        )

        if erp_tensor.dim() == 4:
            erp_tensor = erp_tensor.squeeze(0)

        return self._tensor_to_uint8(erp_tensor)

    # ── 헬퍼 ───────────────────────────────────────────────────────────────

    def _to_tensor(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).float() / 255.0
            if t.dim() == 3:          # HxWxC → CxHxW
                t = t.permute(2, 0, 1)
            t = t.unsqueeze(0)        # 1xCxHxW
        else:
            t = img.float()
            if t.dim() == 3:
                t = t.unsqueeze(0)
        return t.to(self.device)

    def _uint8_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float() / 255.0
        if t.dim() == 3:
            t = t.permute(2, 0, 1)
        return t.unsqueeze(0)

    @staticmethod
    def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().clamp(0, 1).numpy()
        if arr.ndim == 3:             # CxHxW → HxWxC
            arr = arr.transpose(1, 2, 0)
        return (arr * 255).astype(np.uint8)
