# panoClean — 360° 이미지 사람 제거 파이프라인

같은 공간을 촬영한 다수의 360° Equirectangular(ERP) 이미지에서  
사람을 완전히 제거하고 실제 배경으로 자연스럽게 복원합니다.

---

## 파이프라인 개요

```
ERP 이미지 (다수)
       │
       ▼
 [1] CubeMap 변환 (6-face)
       │
       ▼
 [2] 사람 Segmentation
     YOLO11-seg → (선택) SAM2 정밀화 → 마스크 팽창
       │
       ▼
 [3] 멀티-뷰 Feature Matching
     LoFTR / SuperPoint / SIFT
     다른 이미지의 동일 face에서 호모그래피 추정 → 워핑
       │
       ▼
 [4] 배경 합성 + Poisson 블렌딩
       │
       ▼
 [5] 잔여 영역 LaMa Inpainting (선택)
       │
       ▼
 [6] CubeMap → ERP 재합성
       │
       ▼
    결과 저장
```

---

## 요구 사항

- Python 3.10 이상
- CUDA 11.8 이상 (GPU 권장)
- VRAM 8 GB 이상 (face_size=1024 기준)

---

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/jiskang0307/panoClean.git
cd panoClean
```

### 2. PyTorch 설치 (CUDA 11.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. SAM2 설치 (Meta 공식)

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2
```

### 5. 환경 검증

```bash
python utils/check_env.py
```

---

## 실행

### 기본 실행

```bash
python batch_runner.py --input ./input --output ./output
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | `./input` | 입력 ERP 이미지 디렉토리 |
| `--output` | `./output` | 결과 저장 디렉토리 |
| `--config` | `config/default.yaml` | 설정 파일 경로 |
| `--save-comparison` | 비활성 | 원본/결과 나란히 비교 이미지 저장 |
| `--debug` | 비활성 | 디버그 로그 출력 |

### 비교 이미지 포함 실행

```bash
python batch_runner.py --input ./input --output ./output --save-comparison
```

---

## 설정 파일 (`config/default.yaml`)

주요 항목:

| 키 | 기본값 | 설명 |
|----|--------|------|
| `device` | `cuda` | 실행 장치 (`cuda` / `cpu`) |
| `cubemap_face_size` | `1024` | CubeMap 각 face 크기 (px) |
| `yolo_model` | `yolo11x-seg.pt` | YOLO11 세그멘테이션 모델 |
| `sam2_model` | `sam2_hiera_large.pt` | SAM2 체크포인트 (미설정 시 YOLO만 사용) |
| `mask_dilate_px` | `15` | 마스크 팽창 픽셀 수 |
| `feature_matcher` | `loftr` | 매처 종류 (`loftr` / `superpoint` / `sift`) |
| `min_coverage_ratio` | `0.85` | 이 비율 이상 채워지면 inpainting 생략 |
| `lama_enabled` | `true` | LaMa inpainting 사용 여부 |
| `batch_size` | `4` | 배치당 이미지 수 |

---

## 테스트

```bash
pytest tests/ -v
```

---

## 디렉토리 구조

```
panoClean/
├── config/
│   └── default.yaml        # 기본 설정
├── pipeline/
│   ├── cubemap.py          # ERP ↔ CubeMap 변환
│   ├── segmentation.py     # YOLO11-seg / SAM2 마스크 생성
│   ├── matching.py         # LoFTR / SIFT feature matching
│   ├── inpainting.py       # LaMa inpainting
│   └── blending.py         # Poisson / feather 블렌딩
├── utils/
│   ├── image_io.py         # 이미지 로드/저장
│   ├── visualization.py    # 시각화 유틸
│   └── check_env.py        # 환경 검증 스크립트
├── tests/
│   └── test_cubemap.py     # 단위 테스트
├── input/                  # 입력 ERP 이미지 (사용자 제공)
├── output/                 # 결과 저장
├── batch_runner.py         # 배치 실행 진입점
├── requirements.txt
└── README.md
```

---

## 라이선스

MIT License
