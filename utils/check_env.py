"""
check_env.py — CUDA 환경 및 핵심 라이브러리 설치 상태 검증 스크립트.

실행:
    python utils/check_env.py
"""

from __future__ import annotations

import sys


# ANSI 색상 코드
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✘{RESET}  {RED}{msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET}  {YELLOW}{msg}{RESET}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


# ── Python 버전 ────────────────────────────────────────────────────────────
section("Python 환경")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 10:
    ok(f"Python {major}.{minor}.{sys.version_info[2]}")
else:
    fail(f"Python {major}.{minor} — 3.10 이상 필요")


# ── PyTorch / CUDA ─────────────────────────────────────────────────────────
section("PyTorch / CUDA")
try:
    import torch

    ok(f"torch {torch.__version__}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"CUDA 사용 가능 — GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")

        cuda_ver = torch.version.cuda
        ok(f"CUDA 버전: {cuda_ver}")
    else:
        warn("CUDA 불가 — CPU 모드로만 실행됩니다.")
except ImportError:
    fail("torch 미설치 — pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


# ── 핵심 라이브러리 ────────────────────────────────────────────────────────
section("핵심 라이브러리")

libraries = [
    ("ultralytics", "ultralytics", "YOLO11-seg"),
    ("sam2", "sam2", "SAM2 (Meta)"),
    ("equilib", "equilib", "ERP/CubeMap 변환"),
    ("kornia", "kornia", "Feature matching"),
    ("simple_lama_inpainting", "simple_lama_inpainting", "LaMa Inpainting"),
    ("cv2", "cv2", "OpenCV"),
    ("numpy", "numpy", "NumPy"),
    ("PIL", "Pillow", "Pillow"),
    ("scipy", "scipy", "SciPy"),
    ("tqdm", "tqdm", "tqdm"),
    ("yaml", "pyyaml", "PyYAML"),
    ("loguru", "loguru", "loguru"),
    ("matplotlib", "matplotlib", "matplotlib"),
]

all_ok = True
for import_name, pkg_name, display_name in libraries:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        ok(f"{display_name:<30} ({import_name} {ver})")
    except ImportError:
        fail(f"{display_name:<30} 미설치  →  pip install {pkg_name}")
        all_ok = False


# ── 최종 요약 ──────────────────────────────────────────────────────────────
section("요약")
if all_ok:
    ok("모든 라이브러리 정상 설치됨 — 파이프라인 실행 준비 완료")
else:
    fail("일부 라이브러리 누락 — 위 항목을 설치 후 재실행하세요.")

print()
