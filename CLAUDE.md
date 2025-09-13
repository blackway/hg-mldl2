# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 코드 작업을 할 때 지침을 제공합니다.

## 프로젝트 개요

이 저장소는 "혼자 공부하는 머신러닝+딥러닝(개정판)" 책의 코드 예제를 담고 있습니다. Keras와 PyTorch 구현을 모두 사용하여 머신러닝과 딥러닝 개념을 보여주는 포괄적인 Jupyter 노트북 모음입니다.

## 주요 개발 명령어

### 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 가상환경 활성화 (.venv 사용 시)
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 노트북 실행
```bash
# Jupyter Lab 시작하여 노트북 실행
jupyter lab

# Jupyter Notebook 시작 (대안)
jupyter notebook
```

### Python 환경
- Python 버전: 3.12+ (pyproject.toml에 명시)
- 주요 의존성: TensorFlow, PyTorch, Keras, scikit-learn, matplotlib, numpy

## 저장소 구조 및 아키텍처

### 노트북 구성
저장소는 각 노트북이 책의 특정 섹션에 해당하는 장 기반 구조를 따릅니다:

- **01-3.ipynb**: k-최근접 이웃을 사용한 생선 분류로 머신러닝 소개
- **02-x.ipynb**: 데이터 전처리 및 선형 회귀
- **03-x.ipynb**: 분류 알고리즘 (k-최근접 이웃, 로지스틱 회귀)
- **04-x.ipynb**: 트리 기반 알고리즘 (결정 트리, 랜덤 포레스트)
- **05-x.ipynb**: 교차 검증 및 하이퍼파라미터 튜닝
- **06-x.ipynb**: 비지도 학습 (클러스터링, PCA)
- **07-x.ipynb**: 인공 신경망을 이용한 딥러닝 기초
- **08-x.ipynb**: 이미지 분류를 위한 합성곱 신경망 (CNN)
- **09-x.ipynb**: 시퀀스 처리를 위한 순환 신경망 (RNN, LSTM, GRU)
- **10-x.ipynb**: 트랜스포머 및 대규모 언어 모델을 포함한 고급 주제

### 이중 프레임워크 구현
저장소는 Keras와 PyTorch 구현을 모두 제공하는 독특한 특징을 가집니다:
- **표준 .ipynb 파일**: Keras/TensorFlow 구현
- **.pytorch.ipynb로 끝나는 파일**: 동일한 개념의 PyTorch 구현
- 이를 통해 동일한 예제로 두 프레임워크를 모두 학습할 수 있습니다

### 모델 파일
사전 훈련된 모델이 다양한 형식으로 저장됩니다:
- **best-*.keras**: Keras 모델 파일
- **best-*.pt**: PyTorch 모델 파일
- 이러한 모델들은 추론이나 추가 훈련을 위해 로드할 수 있습니다

### 문서 파일
- **README.md**: 일반적인 프로젝트 소개
- **GEMINI.md**: 추가 프로젝트 개요 및 사용 지침
- **분석 파일 (09-2_분석.md, 10-2_분석.md 등)**: 특정 장에 대한 상세 분석 및 발표 자료
- **발표 파일 (*_발표자료.md, *_상세요약.md)**: 장 요약 및 발표 자료

## 노트북 작업하기

### 일반적인 패턴
1. **데이터 로딩**: 대부분의 노트북은 scikit-learn 또는 사용자 정의 데이터 파일에서 데이터셋을 로드하는 것으로 시작
2. **전처리**: scikit-learn의 전처리 도구를 사용한 데이터 전처리
3. **모델 훈련**: 고전적인 ML과 딥러닝 접근법 모두를 사용한 훈련
4. **평가**: 적절한 메트릭을 사용한 성능 평가
5. **시각화**: matplotlib을 사용한 결과 및 모델 동작 시각화

### 사용되는 주요 라이브러리
- **TensorFlow/Keras**: 주요 딥러닝 프레임워크
- **PyTorch**: 대안 딥러닝 프레임워크 (.pytorch.ipynb 파일에서)
- **scikit-learn**: 고전적인 머신러닝 알고리즘
- **matplotlib**: 플롯 및 시각화
- **numpy**: 수치 계산
- **transformers**: 대규모 언어 모델 예제 (10장)

### 노트북 실행 패턴 예제
```python
# 일반적인 노트북 구조:
1. 라이브러리 임포트 및 재현성을 위한 랜덤 시드 설정
2. 데이터 로드 및 탐색
3. 데이터 전처리 (스케일링, 인코딩 등)
4. 적절한 하이퍼파라미터로 모델 훈련
5. 모델 성능 평가
6. 결과 시각화 및 필요시 모델 저장
```

## 개발 가이드라인

### 랜덤 시드 관리
대부분의 노트북은 재현 가능한 결과를 위해 일관된 랜덤 시드를 사용합니다:
```python
# Keras/TensorFlow
keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# PyTorch
torch.manual_seed(42)
```

### 모델 저장 규칙
- `best-lstm-model.keras` 또는 `best-cnn-model.pt`와 같은 설명적인 이름 사용
- 조기 종료 및 최적 모델 체크포인트를 위한 검증 콜백 포함
- 쉬운 접근을 위해 루트 디렉토리에 모델 저장

### 코드 스타일
- 일관성을 위해 책의 코딩 패턴 따르기
- 수학적 개념과 일치하는 명확한 변수명 사용
- 복잡한 ML/DL 개념을 설명하는 주석 포함
- CPU와 GPU 실행 모두와의 호환성 유지

## 특별 고려사항

### 대규모 언어 모델 (10장)
- EXAONE 및 OpenAI API를 사용하는 예제 포함
- OpenAI 통합을 위해 API 키 필요
- 다양한 샘플링 전략을 통한 텍스트 생성 시연

### GPU 지원
- TensorFlow 및 PyTorch 코드는 CPU와 GPU 모두에서 작동하도록 설계됨
- CUDA 가용성이 자동으로 감지됨
- GPU 드라이버 누락에 대한 성능 경고는 CPU 전용 시스템에서 무시 가능

### 가상 환경
- 저장소에는 격리된 Python 환경을 위한 .venv 디렉토리 포함
- 재현 가능한 빌드를 위해 requirements.txt에 모든 의존성이 고정됨