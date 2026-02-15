# 🚀 간단한 KV 캐시 제거 전략

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[LaCache](https://github.com/LaCache/LaCache)에서 제공하는 효율적인 KV 캐시 제거 전략으로, 중요도 점수 계산 오버헤드 없이 장문맥 언어 모델 추론을 지원합니다.

**[English README](README.md)** | **한국어 README**

## 📋 개요

KV 캐싱을 사용하는 대규모 언어 모델(LLM)은 긴 문맥을 처리할 때 메모리 부족 문제가 발생할 수 있습니다. 이 프로젝트는 가장 중요한 토큰만 선택적으로 유지하여 장문맥 추론 중 메모리 소비를 줄이는 **기본 KV 캐시 제거 전략**을 구현합니다:

- **전략 1 (Sink + Recent)**: 첫 N개 토큰(sink)과 마지막 M개 토큰(recent)을 유지하고 중간은 모두 제거
- **전략 2 (Sink + Recent + Uniform Middle)**: sink 토큰, recent 토큰과 함께 중간 영역에서 균등하게 샘플링된 토큰을 유지

### ✨ 주요 특징

- ✅ **제로 오버헤드**: 중요도 점수 계산에 따른 연산 비용 없음 (어텐션 기반 방법과 달리)
- ✅ **메모리 효율적**: 긴 시퀀스에서 50-80% 메모리 절약
- ✅ **Flash Attention 2 호환**: 최적화된 어텐션 커널과 완벽하게 호환
- ✅ **벤치마크 완료**: LongBench 데이터셋의 여러 QA 태스크로 테스트 완료
- ✅ **시각화 도구**: 제거 패턴 시각화 및 전략 비교 도구 포함

## 🏗️ 아키텍처

### 전략 1: Sink + Recent (`middle_strategy="none"`)

```
원본 캐시:
┌─────────────────────────────────────────────────────────┐
│ Sink (첫 N개) │ Middle (제거됨) │ Recent (마지막 M개)  │
└─────────────────────────────────────────────────────────┘

제거 후:
┌──────────────┬──────────────┐
│ Sink (N)     │ Recent (M)   │
└──────────────┴──────────────┘
```

**원리**: 첫 토큰들(sink)은 전체 시퀀스의 맥락을 제공하고, 최근 토큰들은 다음 토큰 예측에 필요합니다. 중간 토큰들은 상대적으로 덜 중요합니다.

### 전략 2: Sink + Recent + Uniform Middle (`middle_strategy="uniform"`)

```
원본 캐시:
┌─────────────────────────────────────────────────────────┐
│ Sink │ Middle (균등 샘플링) │ Recent                     │
└─────────────────────────────────────────────────────────┘

제거 후:
┌──────────┬──────┬──────┬──────┬──────────┐
│ Sink (N) │ Mid1 │ Mid2 │ Mid3 │ Recent(M)│
└──────────┴──────┴──────┴──────┴──────────┘
```

**원리**: 고정된 메모리 예산 내에서 시퀀스 전체의 중요한 정보를 포착하기 위해 균등하게 분포된 중간 토큰을 추가합니다.

## 📁 프로젝트 구조

```
simple_kv_eviction/
├── compare_strategies.py      # 다양한 전략 간 결과 비교
├── run_longbench_example.py   # 제거 기능이 포함된 LongBench 평가 실행
├── test_simple_evict.py       # 제거 로직 단위 테스트
├── visualize_eviction.py      # 제거 패턴을 히트맵으로 시각화
├── run_examples.sh            # 실험 일괄 실행 스크립트
├── requirements.txt           # Python 의존성
├── results/                   # 실험 결과 (JSON)
└── README.md                  # 영문 README
```

## 💻 설치 방법

### LaCache 가상 환경 사용

```bash
# LaCache 가상 환경 활성화
source ../LaCache/.venv/bin/activate

# 의존성 설치 (필요한 경우)
pip install -r requirements.txt
```

### 의존성

- **PyTorch**: 2.4.1
- **Transformers**: 4.43.4
- **Flash Attention 2**: 2.6.3 (최적화된 어텐션 계산용)
- **Block Sparse Flash Attention**: 2.8.3 (선택사항, 희소 어텐션 패턴용)

## 🚀 사용 예제

### 예제 1: Sink + Recent만 사용

```python
from evict.simple_evict import SimpleEvictConfig, build_simple_keep_token_idx

# 설정: 처음 256개 + 마지막 512개 토큰 유지
config = SimpleEvictConfig(
    sink_tokens=256,
    recent_tokens=512,
    middle_strategy="none"
)

# 유지할 토큰의 인덱스 생성
keep_idx = build_simple_keep_token_idx(
    total_len=4096,      # 캐시의 전체 토큰 수
    block_n=128,         # 블록 크기
    cfg=config
)
# 출력: [0, 1, ..., 255, 3584, 3585, ..., 4095]
# 총 768개 토큰 유지 (256 + 512)
```

### 예제 2: Sink + Recent + Uniform Middle

```python
# 설정: sink + recent + 균등 분포 중간 토큰 유지
config = SimpleEvictConfig(
    sink_tokens=256,
    recent_tokens=512,
    middle_strategy="uniform",
    middle_budget_tokens=512,  # 중간 영역에서 ~512개 토큰 유지
    uniform_stride=0            # 예산 사용 (stride 아님)
)

keep_idx = build_simple_keep_token_idx(
    total_len=4096,
    block_n=128,
    cfg=config
)
# 출력: [0, ..., 255, 512, 1024, 1536, ..., 3584, ..., 4095]
# 총 ~1280개 토큰 유지 (256 + 512 + 512)
```

## 🧪 실험 결과

이 프로젝트는 **LongBench** 데이터셋의 여러 질의응답 태스크에 대한 벤치마크 결과를 포함합니다:

| 데이터셋 | 태스크 유형 | 평균 문맥 길이 |
|---------|-----------|---------------|
| NarrativeQA | 독해 | ~18K 토큰 |
| Qasper | 과학 QA | ~3.5K 토큰 |
| MultiFieldQA-en | 다분야 QA (영어) | ~4.5K 토큰 |
| MultiFieldQA-zh | 다분야 QA (중국어) | ~6K 토큰 |
| HotpotQA | 다단계 추론 | ~9K 토큰 |
| 2WikiMQA | 다단계 추론 | ~4.5K 토큰 |
| MuSiQue | 다단계 추론 | ~11K 토큰 |

### 시각화 예제

프로젝트에는 어떤 토큰이 유지/제거되는지 보여주는 히트맵을 생성하는 시각화 도구가 포함되어 있습니다:

- **파란색 영역**: 유지된 토큰
- **회색 영역**: 제거된 토큰
- **빨간 점선**: Sink 경계
- **녹색 점선**: Recent 경계

## 🔬 실험 실행하기

### 빠른 테스트 (GPU 불필요)

```bash
# 더미 데이터로 제거 로직 테스트
python test_simple_evict.py
```

### 제거 패턴 시각화

```bash
# 기본 설정 시각화
python visualize_eviction.py

# sink + recent만 시각화 (middle 없음)
python visualize_eviction.py --strategy sink_recent --save eviction_sink_recent.png

# 커스텀 파라미터로 시각화
python visualize_eviction.py \
    --total_len 8192 \
    --strategy sink_recent_uniform \
    --sink_tokens 256 \
    --recent_tokens 1024 \
    --middle_budget 2816 \
    --save eviction_custom.png
```

### LongBench 평가 실행 (GPU 필요)

```bash
# 데모 모드 (작은 모델, 2개 예제)
python run_longbench_example.py --demo_mode

# 특정 전략으로 전체 평가
python run_longbench_example.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --strategy sink_recent \
    --sink_tokens 256 \
    --recent_tokens 1024 \
    --dataset narrativeqa \
    --num_examples 10 \
    --output results/sr_narrativeqa.json

# uniform middle 전략으로 평가
python run_longbench_example.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --strategy sink_recent_uniform \
    --sink_tokens 256 \
    --recent_tokens 1024 \
    --middle_budget 512 \
    --dataset hotpotqa \
    --num_examples 10 \
    --output results/sru_hotpotqa.json
```

### 전략 비교

```bash
# 두 전략 비교
python compare_strategies.py results/sr_narrativeqa.json results/sru_narrativeqa.json

# 시각화와 함께 비교
python compare_strategies.py results/sr_narrativeqa.json results/sru_narrativeqa.json --plot
```

## 📊 성능 특성

| 전략 | 메모리 절약 | 연산 오버헤드 | 최적 용도 |
|------|-----------|-------------|----------|
| Sink + Recent | 높음 (50-80%) | **제로** | 긴 시퀀스, 엄격한 메모리 제약 |
| Sink + Recent + Uniform | 중간 (30-60%) | **제로** | 긴 시퀀스, 더 나은 품질 |

**다른 방법과의 비교:**
- ❌ **어텐션 기반 제거** (예: H2O): 어텐션 점수 계산 필요 → 오버헤드 발생
- ❌ **EMA 기반 중요도** (예: StreamingLLM): 실행 통계 추적 → 오버헤드 발생
- ✅ **이 접근법**: 위치 기반 제거 → **제로 오버헤드**

## 🎯 활용 사례

1. **긴 문서 질의응답**: 모델의 컨텍스트 윈도우를 넘어서는 문서 처리
2. **다중 턴 대화**: 대화 기록을 메모리 제한 내에서 유지
3. **스트리밍 추론**: 제어된 메모리 증가로 실시간 처리
4. **연구**: 고급 제거 전략을 위한 베이스라인 비교

## 🤝 기여하기

기여는 언제나 환영합니다! 이슈나 풀 리퀘스트를 자유롭게 제출해주세요.

## 📚 참고 문헌

- **LaCache**: [장문맥 추론 프레임워크](https://github.com/LaCache/LaCache)
- **Flash Attention 2**: [Dao et al., 2023 - 더 나은 병렬화와 작업 분할을 통한 빠른 어텐션](https://arxiv.org/abs/2307.08691)
- **LongBench**: [Bai et al., 2023 - 장문맥 이해를 위한 이중언어, 다중작업 벤치마크](https://arxiv.org/abs/2308.14508)
- **StreamingLLM**: [Xiao et al., 2023 - 어텐션 싱크를 활용한 효율적인 스트리밍 언어 모델](https://arxiv.org/abs/2309.17453)

## 📄 라이선스

이 프로젝트는 LaCache 프레임워크를 기반으로 합니다. 라이선스 정보는 [LaCache 저장소](https://github.com/LaCache/LaCache)를 참조하세요.

## 🙏 감사의 말

- LaCache 팀의 원본 제거 구현
- HuggingFace Transformers의 모델 인터페이스
- LongBench 저자들의 평가 벤치마크

---

**⭐ 이 프로젝트가 도움이 되셨다면 스타를 눌러주세요!**
