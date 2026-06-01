"""
inpainting.py — LaMa 기반 마스크 영역 복원.

simple-lama-inpainting 래핑:
  - inpaint_face()       : 일반 face, 원본 해상도 (down face는 512px 다운샘플)
  - inpaint_down_face()  : down face 전용, 256px 다운샘플, 30px feathering
  - inpaint_residual()   : 배경 교체 후 잔여 영역 처리 (down → inpaint_down_face)
  - inpaint_all_faces()  : 6개 face 전부 (down 먼저)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


class LamaInpainter:
    """LaMa(Large Mask inpainting) 기반 이미지 복원기."""

    def __init__(
        self,
        device: str = "cuda",
        debug_dir: Optional[Path] = None,
        residual_threshold: float = 0.05,
        down_face_size: int = 256,
        feather_px: int = 8,
        down_feather_px: int = 30,
    ) -> None:
        self.device = device
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.residual_threshold = residual_threshold
        self.down_face_size = down_face_size
        self.feather_px = feather_px
        self.down_feather_px = down_feather_px
        self.lama = None
        self.available = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            from simple_lama_inpainting import SimpleLama
            logger.info("LaMa 모델 로드 중...")
            self.lama = SimpleLama()
            self.available = True
            logger.info("LaMa 로드 완료")
        except Exception as e:
            logger.warning(f"LaMa 로드 실패: {e} — inpainting 비활성화")
            self.lama = None
            self.available = False

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def inpaint_face(
        self,
        face_img: torch.Tensor,
        mask: torch.Tensor,
        face_name: str = "",
    ) -> torch.Tensor:
        """
        입력:
          face_img : (3, H, W) float32 [0,1]
          mask     : (H, W) bool — True = inpainting 대상
        출력:
          (3, H, W) float32, mask 영역이 채워진 이미지

        down face는 512px 다운샘플 후 inpainting → 업샘플.
        나머지 face는 원본 해상도 유지.
        경계 feathering: sigma=feather_px (기본 8px).
        """
        if not mask.any():
            return face_img
        if not self.available:
            logger.warning(f"[{face_name}] LaMa 비활성화 — 원본 반환")
            return face_img

        _, H, W = face_img.shape

        if face_name == "down":
            proc_size = 512
            proc_img = F.interpolate(
                face_img.unsqueeze(0), size=(proc_size, proc_size),
                mode="bilinear", align_corners=False
            ).squeeze(0)
            proc_mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0), size=(proc_size, proc_size),
                mode="nearest"
            ).squeeze(0).squeeze(0).bool()
        else:
            proc_img = face_img
            proc_mask = mask

        pil_img, pil_mask = self._to_pil(proc_img, proc_mask)
        result_pil = self._run_lama(pil_img, pil_mask)
        result_proc = self._from_pil(result_pil, proc_img)

        if face_name == "down":
            result_t = F.interpolate(
                result_proc.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze(0).to(face_img.device)
        else:
            result_t = result_proc

        # feathering
        mask_np = mask.cpu().numpy().astype(np.uint8)
        alpha = self._feather_mask(mask_np, sigma=self.feather_px)
        alpha_t = torch.from_numpy(alpha).to(face_img.device).unsqueeze(0)
        blended = (face_img * (1.0 - alpha_t) + result_t * alpha_t).clamp(0, 1)

        if self.debug_dir and face_name:
            self._save_diff_debug(face_img, blended, mask, face_name)

        return blended

    def inpaint_down_face(
        self,
        face_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        down face 전용.
        256px 다운샘플 → LaMa → 업샘플 → 30px feathering.
        """
        if not self.available:
            logger.warning("[down] LaMa 비활성화 — 원본 반환")
            return face_img

        _, H, W = face_img.shape
        target_size = self.down_face_size

        small_img = F.interpolate(
            face_img.unsqueeze(0), size=(target_size, target_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        small_mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0), size=(target_size, target_size),
            mode="nearest"
        ).squeeze(0).squeeze(0).bool()

        pil_img, pil_mask = self._to_pil(small_img, small_mask)
        result_pil = self._run_lama(pil_img, pil_mask)
        result_small = self._from_pil(result_pil, small_img)

        result_t = F.interpolate(
            result_small.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze(0).to(face_img.device)

        mask_np = mask.cpu().numpy().astype(np.uint8)
        alpha = self._feather_mask(mask_np, sigma=self.down_feather_px)
        alpha_t = torch.from_numpy(alpha).to(face_img.device).unsqueeze(0)
        blended = (face_img * (1.0 - alpha_t) + result_t * alpha_t).clamp(0, 1)

        if self.debug_dir:
            self._save_diff_debug(face_img, blended, mask, "down")

        return blended

    def inpaint_residual(
        self,
        face_img: torch.Tensor,
        original_mask: torch.Tensor,
        filled_mask,  # (H,W) bool tensor OR scalar bool
        face_name: str = "",
    ) -> tuple[torch.Tensor, bool]:
        """
        배경 교체 후 잔여 영역만 inpainting.

        face_name == "down": inpaint_down_face() 호출 (filled_mask 무시).
        나머지: residual = original_mask & ~filled_mask, 5% 미만이면 스킵.

        반환: (처리된 이미지, inpainting 실행 여부)
        """
        if face_name == "down":
            if not original_mask.any():
                return face_img, False
            result = self.inpaint_down_face(face_img, original_mask)
            return result, self.available

        # 잔여 마스크 계산
        if isinstance(filled_mask, bool) or (
            isinstance(filled_mask, torch.Tensor) and filled_mask.ndim == 0
        ):
            residual = torch.zeros_like(original_mask) if bool(filled_mask) else original_mask
        else:
            residual = original_mask & ~filled_mask.bool()

        total = int(original_mask.sum().item())
        if total == 0:
            return face_img, False

        ratio = int(residual.sum().item()) / total
        if ratio < self.residual_threshold:
            logger.debug(f"[{face_name}] 잔여 {ratio:.1%} < {self.residual_threshold:.0%} — LaMa 스킵")
            return face_img, False

        if not self.available:
            logger.warning(f"[{face_name}] LaMa 비활성화 — 원본 반환")
            return face_img, False

        if self.debug_dir and face_name:
            self._save_residual_mask_debug(residual, face_name)

        logger.info(f"[{face_name}] LaMa inpainting 시작 (잔여={ratio:.1%})")
        result = self.inpaint_face(face_img, residual, face_name=face_name)
        return result, True

    def inpaint_all_faces(
        self,
        faces: dict[str, torch.Tensor],
        original_masks: dict[str, torch.Tensor],
        filled_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """6개 face 전부 처리. down face 먼저."""
        results: dict[str, torch.Tensor] = {}
        order = ["down"] + [f for f in faces if f != "down"]

        for face_name in order:
            if face_name not in faces:
                continue

            t0 = time.monotonic()
            orig_mask = original_masks.get(face_name, torch.zeros(1, dtype=torch.bool))
            fill_mask = filled_masks.get(face_name, torch.zeros_like(orig_mask))

            result, did_inpaint = self.inpaint_residual(
                faces[face_name], orig_mask, fill_mask, face_name=face_name
            )
            results[face_name] = result

            elapsed_ms = (time.monotonic() - t0) * 1000
            total = int(orig_mask.sum().item())
            if total > 0 and isinstance(fill_mask, torch.Tensor) and fill_mask.ndim > 0:
                filled_px = int(fill_mask.sum().item())
                residual_ratio = max(0.0, (total - filled_px) / total)
            else:
                residual_ratio = 1.0 if total > 0 else 0.0

            logger.info(
                f"[{face_name}] inpainting={did_inpaint}, "
                f"residual_ratio={residual_ratio:.2f}, "
                f"elapsed={elapsed_ms:.1f}ms"
            )

        return results

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _to_pil(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> tuple:
        """(3,H,W) float32 + (H,W) bool → (PIL Image RGB, PIL Image L)"""
        from PIL import Image
        arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_u8 = (arr * 255).astype(np.uint8)
        mask_u8 = mask.cpu().numpy().astype(np.uint8) * 255
        return Image.fromarray(img_u8, "RGB"), Image.fromarray(mask_u8, "L")

    def _from_pil(self, pil_img, original: torch.Tensor) -> torch.Tensor:
        """PIL Image RGB → (3,H,W) float32 tensor (original과 동일 device)"""
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).to(original.device)

    def _feather_mask(self, mask: np.ndarray, sigma: int = 8) -> np.ndarray:
        """binary uint8 mask → gaussian-blurred float32 [0,1] alpha"""
        ksize = max(3, int(sigma * 4 + 1) | 1)  # 홀수 커널
        mask_f = mask.astype(np.float32)
        feathered = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
        return np.clip(feathered, 0, 1).astype(np.float32)

    def _run_lama(self, pil_img, pil_mask):
        """GPU OOM 발생 시 모델을 CPU로 이동 후 재시도."""
        try:
            return self.lama(pil_img, pil_mask)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            logger.warning("GPU OOM — CPU fallback 시도")
            torch.cuda.empty_cache()
            model = getattr(self.lama, "model", None) or getattr(self.lama, "net", None)
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
                try:
                    result = self.lama(pil_img, pil_mask)
                finally:
                    model.to(self.device)
                return result
            raise

    def _save_diff_debug(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        mask: torch.Tensor,
        face_name: str,
    ) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(exist_ok=True)
        vis = visualize_inpainting_diff(before, after, mask, face_name)
        path = self.debug_dir / f"inpaint_before_after_{face_name}.jpg"
        cv2.imwrite(str(path), vis)
        logger.debug(f"디버그 저장: {path.name}")

    def _save_residual_mask_debug(
        self, residual: torch.Tensor, face_name: str
    ) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(exist_ok=True)
        mask_u8 = residual.cpu().numpy().astype(np.uint8) * 255
        path = self.debug_dir / f"inpaint_residual_mask_{face_name}.jpg"
        cv2.imwrite(str(path), mask_u8)


def visualize_inpainting_diff(
    before: torch.Tensor,
    after: torch.Tensor,
    mask: torch.Tensor,
    face_name: str = "",
) -> np.ndarray:
    """
    전/후 비교 이미지 생성.
    좌=before, 우=after. mask 경계를 노란색 contour로 표시.
    """
    def t2bgr(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    b = t2bgr(before)
    a = t2bgr(after)
    mask_u8 = mask.cpu().numpy().astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(b, cnts, -1, (0, 255, 255), 2)
    cv2.drawContours(a, cnts, -1, (0, 255, 255), 2)
    if face_name:
        for img in (b, a):
            cv2.putText(img, face_name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return np.hstack([b, a])
