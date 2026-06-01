"""
pipeline 패키지 — 360° 이미지 사람 제거 파이프라인의 핵심 모듈 모음.

Modules:
    cubemap     : ERP ↔ CubeMap 변환
    segmentation: YOLO11-seg / SAM2 기반 사람 마스크 생성
    matching    : 멀티-뷰 feature matching 및 호모그래피 추정
    inpainting  : LaMa 기반 영역 복원
    blending    : 다중 소스 이미지 합성 및 경계 블렌딩
"""
