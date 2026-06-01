"""
conftest.py — pytest 공통 설정.

핵심 의존성이 미설치된 환경에서도 테스트 컬렉션이 실패하지 않도록
import 오류를 graceful하게 처리한다.
"""

from __future__ import annotations

import importlib
import pytest


def _available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


# 각 테스트 파일 최상단에서 사용할 수 있는 skip 마커
requires_torch = pytest.mark.skipif(
    not _available("torch"),
    reason="torch 미설치 — pip install torch",
)
requires_equilib = pytest.mark.skipif(
    not (_available("equilib") or _available("py360convert")),
    reason="equilib / py360convert 미설치",
)


def pytest_collection_modifyitems(items):
    """torch 또는 equilib 없이는 cubemap 관련 테스트 전체 skip."""
    missing_torch = not _available("torch")
    missing_backend = not (_available("equilib") or _available("py360convert"))

    skip_torch = pytest.mark.skip(reason="torch 미설치")
    skip_backend = pytest.mark.skip(reason="equilib / py360convert 미설치")

    for item in items:
        if missing_torch:
            item.add_marker(skip_torch)
        elif missing_backend and "cubemap" in item.nodeid:
            item.add_marker(skip_backend)
