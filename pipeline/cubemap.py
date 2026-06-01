"""
cubemap.py — ERP(Equirectangular Projection) ↔ CubeMap 양방향 변환.

equilib 우선, 없으면 py360convert로 fallback.
모든 공개 메서드는 numpy/PIL/torch 혼용 입력을 수용한다.
"""

from __future__ import annotations

import math
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from tqdm import tqdm

# ── 백엔드 선택 ───────────────────────────────────────────────────────────
try:
    from equilib import equi2cube, cube2equi
    _BACKEND = "equilib"
except ImportError:  # pragma: no cover
    try:
        import py360convert as p360
        _BACKEND = "py360convert"
        logger.warning("equilib 미설치 — py360convert fallback 사용")
    except ImportError:
        raise ImportError("equilib 또는 py360convert 중 하나를 설치하세요.")

logger.debug(f"cubemap backend: {_BACKEND}")

# ── 타입 ──────────────────────────────────────────────────────────────────
FaceName = Literal["front", "right", "back", "left", "up", "down"]
FACE_NAMES: list[FaceName] = ["front", "right", "back", "left", "up", "down"]

# equilib은 "top"/"bottom", py360convert는 "up"/"down" 키 사용
# 내부적으로 "up"/"down"을 표준으로 삼고 equilib 호출 시 변환
_EQUI_KEY: dict[str, str] = {
    "front": "front", "right": "right", "back": "back",
    "left": "left", "up": "top", "down": "bottom",
}
_EQUI_KEY_INV: dict[str, str] = {v: k for k, v in _EQUI_KEY.items()}

ImageLike = Union[np.ndarray, torch.Tensor, Image.Image]


# ── 유틸 함수 (모듈 레벨) ─────────────────────────────────────────────────

def load_erp(path: str) -> torch.Tensor:
    """
    ERP 이미지 파일을 (3, H, W) float32 CUDA tensor로 로드.

    Returns:
        torch.Tensor: 값 범위 [0, 1], device=cuda(가능한 경우).
    """
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.array(img)).float() / 255.0  # HxWx3
    t = t.permute(2, 0, 1)                               # 3xHxW
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return t.to(device)


def save_erp(tensor: torch.Tensor, path: str) -> None:
    """
    (3, H, W) float32 tensor를 ERP 이미지 파일로 저장.
    확장자에 따라 PNG/JPEG 자동 선택.
    """
    import pathlib
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    logger.debug(f"ERP 저장: {path}")


# ── 메인 클래스 ───────────────────────────────────────────────────────────

class CubeMapConverter:
    """
    ERP ↔ CubeMap 양방향 변환기.

    모든 공개 메서드 I/O:
      - 입력: numpy HxWx3 uint8 | PIL Image | torch (3,H,W) float32
      - 출력: torch (3, face_size, face_size) float32, device 유지
    """

    SEAM_WIDTH = 8  # face 경계 블렌딩 픽셀 폭

    def __init__(self, face_size: int = 1024, device: str = "cuda") -> None:
        self.face_size = face_size
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 불가 — CPU fallback")
            device = "cpu"
        self.device = torch.device(device)
        logger.info(f"CubeMapConverter: face_size={face_size}, device={self.device}, backend={_BACKEND}")

    # ── ERP → CubeMap ─────────────────────────────────────────────────────

    def erp_to_cubemap(self, erp_img: ImageLike) -> dict[FaceName, torch.Tensor]:
        """
        ERP 이미지를 6개 face dict로 변환.

        Args:
            erp_img: HxWx3 uint8 ndarray / PIL Image / (3,H,W) float32 tensor.

        Returns:
            {"front","right","back","left","up","down"} 각각 (3, face_size, face_size) float32.
        """
        src = self._to_chw_tensor(erp_img).to(self.device)  # 3xHxW
        logger.debug(f"erp_to_cubemap: src={tuple(src.shape)}")

        if _BACKEND == "equilib":
            faces = self._equi2cube_equilib(src)
        else:
            faces = self._equi2cube_py360(src)

        return faces

    def cubemap_to_erp(
        self,
        faces: dict[FaceName, torch.Tensor],
        erp_height: int,
        erp_width: int,
        no_blend_faces: tuple[str, ...] | list[str] = ("down",),
    ) -> torch.Tensor:
        """
        6개 face dict를 ERP tensor로 합성.

        면 경계는 8px gaussian blend로 처리.
        no_blend_faces에 포함된 face는 seam blend 없이 원본 그대로 합성.

        Args:
            faces          : erp_to_cubemap 반환값과 동일한 구조.
            erp_height     : 출력 ERP 높이.
            erp_width      : 출력 ERP 너비.
            no_blend_faces : seam blend를 생략할 face 이름 목록.
                             기본값 ("down",) — 넓은 inpainting 영역 번짐 방지.

        Returns:
            (3, erp_height, erp_width) float32 tensor (device 유지).
        """
        # no_blend_faces는 blend_seams에 넣지 않고 원본 그대로 사용
        blend_input  = {k: v for k, v in faces.items() if k not in no_blend_faces}
        pass_through = {k: v for k, v in faces.items() if k in no_blend_faces}

        blended = self.blend_seams(blend_input)
        blended.update(pass_through)   # down face 원본 그대로 삽입
        logger.debug(f"cubemap_to_erp: → ({erp_height}, {erp_width})")

        if _BACKEND == "equilib":
            erp = self._cube2equi_equilib(blended, erp_height, erp_width)
        else:
            erp = self._cube2equi_py360(blended, erp_height, erp_width)

        return erp

    # ── 배치 처리 ─────────────────────────────────────────────────────────

    def erp_to_cubemap_batch(
        self, erp_imgs: list[ImageLike]
    ) -> list[dict[FaceName, torch.Tensor]]:
        """
        ERP 이미지 리스트를 배치 처리.

        Returns:
            erp_to_cubemap 결과 리스트 (순서 유지).
        """
        results: list[dict[FaceName, torch.Tensor]] = []
        for img in tqdm(erp_imgs, desc="ERP→CubeMap", unit="img"):
            results.append(self.erp_to_cubemap(img))
        return results

    # ── UV 맵 ─────────────────────────────────────────────────────────────

    def get_face_uv_map(
        self, face_name: FaceName, erp_h: int, erp_w: int
    ) -> torch.Tensor:
        """
        특정 face의 각 픽셀이 ERP에서 어느 좌표인지 UV 맵 반환.

        Returns:
            (2, face_size, face_size) float32 tensor, 값 범위 [0, 1].
            [0] = u (x 방향), [1] = v (y 방향).

        배경 대체 시 source ERP에서 픽셀을 역추적할 때 사용한다.
        """
        fs = self.face_size
        # face 로컬 좌표 [-1, 1] 그리드
        lin = torch.linspace(-1 + 1 / fs, 1 - 1 / fs, fs, device=self.device)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")  # (fs, fs)

        lon, lat = self._face_xy_to_lonlat(face_name, grid_x, grid_y)

        # ERP UV: lon[-π,π]→u[0,1], lat[-π/2,π/2]→v[0,1]
        u = (lon / (2 * math.pi) + 0.5).clamp(0, 1)
        v = (0.5 - lat / math.pi).clamp(0, 1)

        return torch.stack([u, v], dim=0)  # (2, fs, fs)

    # ── Seam Blending ─────────────────────────────────────────────────────

    def blend_seams(
        self, faces: dict[FaceName, torch.Tensor]
    ) -> dict[FaceName, torch.Tensor]:
        """
        인접한 face 경계를 gaussian weighted average로 부드럽게 blend.

        각 face 모서리 SEAM_WIDTH px를 인접 face에서 샘플링해 혼합한다.

        Returns:
            동일 구조의 새 dict (원본 불변).
        """
        w = self.SEAM_WIDTH
        result = {k: v.clone() for k, v in faces.items()}

        # 1D 감쇠 weight: 경계에서 1→내부에서 0
        decay = torch.linspace(1.0, 0.0, w, device=self.device)  # (w,)
        decay = (decay ** 2).reshape(1, 1, -1)  # (1, 1, w) — 브로드캐스트용

        # 인접 face 쌍: (dst_face, dst_edge, src_face, src_edge)
        # edge: 'top','bottom','left','right'
        adjacency = [
            # front의 오른쪽 ↔ right의 왼쪽
            ("front", "right_edge", "right", "left_edge"),
            # right의 오른쪽 ↔ back의 왼쪽
            ("right", "right_edge", "back", "left_edge"),
            # back의 오른쪽 ↔ left의 왼쪽
            ("back", "right_edge", "left", "left_edge"),
            # left의 오른쪽 ↔ front의 왼쪽
            ("left", "right_edge", "front", "left_edge"),
        ]

        for dst, dst_edge, src, src_edge in adjacency:
            if dst not in result or src not in result:
                continue
            d = result[dst]   # (3, fs, fs)
            s = result[src]

            # 오른쪽 경계: d[:, :, -w:] blends with s[:, :, :w] (flipped)
            blend_w = decay                          # (1,1,w): 경계=1, 내부=0
            d_edge = d[:, :, -w:]                   # (3, fs, w)
            s_edge = s[:, :, :w].flip(-1)           # (3, fs, w)
            result[dst][:, :, -w:] = d_edge * (1 - blend_w) + s_edge * blend_w

            # 반대편: src 왼쪽 경계도 대칭 blend
            s_e2 = s[:, :, :w]
            d_e2 = d[:, :, -w:].flip(-1)
            result[src][:, :, :w] = s_e2 * (1 - blend_w) + d_e2 * blend_w

        return result

    # ── 내부: equilib 호출 ────────────────────────────────────────────────

    def _equi2cube_equilib(self, src: torch.Tensor) -> dict[FaceName, torch.Tensor]:
        # equilib은 배치 차원을 요구하는 버전이 있으므로 1xCxHxW로 전달
        batch = src.unsqueeze(0)  # (1, 3, H, W)
        raw = equi2cube(
            src=batch,
            rots={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            w_face=self.face_size,
            cube_format="dict",
        )
        faces: dict[FaceName, torch.Tensor] = {}
        for equi_key, std_key in _EQUI_KEY_INV.items():
            t = raw[equi_key]
            if t.dim() == 4:
                t = t.squeeze(0)  # (3, fs, fs)
            faces[std_key] = t.to(self.device).float().clamp(0, 1)
        return faces

    def _cube2equi_equilib(
        self,
        faces: dict[FaceName, torch.Tensor],
        h: int,
        w: int,
    ) -> torch.Tensor:
        cube_dict = {
            _EQUI_KEY[std_key]: faces[std_key].unsqueeze(0).to(self.device)
            for std_key in FACE_NAMES
        }
        out = cube2equi(
            cubemap=cube_dict,
            cube_format="dict",
            height=h,
            width=w,
        )
        if out.dim() == 4:
            out = out.squeeze(0)
        return out.to(self.device).float().clamp(0, 1)

    # ── 내부: py360convert fallback ───────────────────────────────────────

    def _equi2cube_py360(self, src: torch.Tensor) -> dict[FaceName, torch.Tensor]:
        arr = src.cpu().permute(1, 2, 0).numpy()  # HxWx3 float32
        faces: dict[FaceName, torch.Tensor] = {}
        # py360convert face order: F R B L U D
        for i, name in enumerate(FACE_NAMES):
            face_arr = p360.e2c(arr, face_w=self.face_size, cube_format="list")[i]
            t = torch.from_numpy(face_arr).float().permute(2, 0, 1).to(self.device)
            faces[name] = t.clamp(0, 1)
        return faces

    def _cube2equi_py360(
        self,
        faces: dict[FaceName, torch.Tensor],
        h: int,
        w: int,
    ) -> torch.Tensor:
        face_list = [
            faces[n].cpu().permute(1, 2, 0).numpy() for n in FACE_NAMES
        ]
        arr = p360.c2e(face_list, h=h, w=w, cube_format="list")
        return torch.from_numpy(arr).float().permute(2, 0, 1).to(self.device).clamp(0, 1)

    # ── 내부: 좌표 변환 ───────────────────────────────────────────────────

    @staticmethod
    def _face_xy_to_lonlat(
        face: FaceName,
        x: torch.Tensor,  # (fs, fs) in [-1, 1]
        y: torch.Tensor,  # (fs, fs) in [-1, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """face 로컬 xy → 구면 경도/위도 (라디안)."""
        one = torch.ones_like(x)

        face_dirs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            "front":  ( one,  x,  -y),
            "back":   (-one, -x,  -y),
            "right":  (   x, -one, -y),  # local coords vary by convention
            "left":   (  -x,  one, -y),
            "up":     (   y,   x,  one),
            "down":   (  -y,   x, -one),
        }
        # Use a consistent projection: map face to 3D unit vector
        # Standard cubemap: front=+Z, right=+X, up=+Y
        _dirs: dict[str, tuple] = {
            "front":  ( x,  -y,  one),
            "back":   (-x,  -y, -one),
            "right":  ( one, -y,  -x),
            "left":   (-one, -y,   x),
            "up":     ( x,   one,  y),
            "down":   ( x,  -one, -y),
        }
        vx, vy, vz = _dirs[face]
        norm = torch.sqrt(vx**2 + vy**2 + vz**2).clamp(min=1e-8)
        vx, vy, vz = vx / norm, vy / norm, vz / norm

        lat = torch.asin(vy.clamp(-1, 1))
        lon = torch.atan2(vx, vz)
        return lon, lat

    # ── 내부: 타입 변환 ───────────────────────────────────────────────────

    def _to_chw_tensor(self, img: ImageLike) -> torch.Tensor:
        """임의 타입 입력을 (3, H, W) float32 tensor로 변환."""
        if isinstance(img, torch.Tensor):
            t = img.float()
            if t.max() > 1.5:
                t = t / 255.0
            if t.dim() == 3 and t.shape[0] not in (1, 3, 4):
                t = t.permute(2, 0, 1)
            return t.clamp(0, 1)
        if isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))
        # numpy HxWxC uint8 or float
        arr = np.asarray(img)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        if t.dim() == 3 and t.shape[2] in (3, 4):
            t = t[:, :, :3].permute(2, 0, 1)
        return t.clamp(0, 1)
