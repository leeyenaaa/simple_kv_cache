# KV Cache Eviction Implementation Plan

## 목표
LongBench 태스크에서 chunk 단위로 프롬프트를 입력하며 KV cache를 관리하는 시스템 구현

## 요구사항

### 1. 두 가지 Eviction 전략
1. **Strategy 1: Sink + Recent Only (중간 버리기)**
   - 맨 앞 sink 토큰들 유지
   - 최근 recent 토큰들 유지
   - 중간(mid) 부분은 모두 버림

2. **Strategy 2: Sink + Recent + Uniform Mid (규칙적 분산)**
   - 맨 앞 sink 토큰들 유지
   - 최근 recent 토큰들 유지
   - 중간(mid) 부분에서 규칙적인 간격으로 토큰들을 선택하여 유지

### 2. 파라미터
- `kv_cache_size`: 최대 캐시 크기 (토큰 수)
- `sink_tokens`: 앞에서 유지할 토큰 수
- `recent_tokens`: 뒤에서 유지할 토큰 수
- `middle_budget_tokens`: (Strategy 2용) 중간에서 유지할 토큰 수
- `uniform_stride`: (Strategy 2용) 선택 간격

### 3. 기술 스택
- Flash Attention 2 사용
- 가상환경: `../LaCache/.venv`
- 기존 LaCache 코드베이스 활용

## 기존 코드 분석

### LaCache/LongBench/evict/simple_evict.py
이미 두 가지 전략이 완벽하게 구현되어 있음:

```python
@dataclass
class SimpleEvictConfig:
    sink_tokens: int = 256
    recent_tokens: int = 512
    middle_strategy: str = "none"  # "none" or "uniform"
    middle_budget_tokens: int = 0
    uniform_stride: int = 0
```

- `middle_strategy="none"`: Sink + Recent만 (Strategy 1)
- `middle_strategy="uniform"`: Sink + Recent + Uniform Mid (Strategy 2)

### LaCache/LongBench/gated_pred.py
- LongBench 데이터셋 평가 스크립트
- Streaming prefill 지원 (chunk 단위 입력)
- Flash Attention 지원
- Simple eviction 통합되어 있음

## 구현 계획

### Task 1: 프로젝트 구조 생성
- README.md: 사용법, 예제, 설명
- 디렉토리 구조 설정

### Task 2: 간단한 테스트 스크립트 작성
`test_simple_evict.py`: 두 전략의 동작을 검증하는 미니멀 예제
- 작은 입력 시퀀스로 테스트
- 어떤 토큰들이 유지되는지 시각화
- 메모리 사용량 측정

### Task 3: LongBench 통합 예제
`run_longbench_example.py`: LongBench 데이터셋에서 실제 테스트
- Strategy 1과 Strategy 2 비교
- 성능 메트릭 수집
- Flash Attention 활성화

### Task 4: 비교 분석 스크립트
`compare_strategies.py`: 두 전략의 상세 비교
- Accuracy 비교
- Latency 비교
- Memory usage 비교
- 시각화 (kept tokens heatmap)

### Task 5: 실행 스크립트
`run_examples.sh`: 모든 예제를 순차 실행하는 쉘 스크립트

### Task 6: 문서화
- README.md 완성
- 코드 주석 추가
- 사용 예제 추가

## 실행 순서

1. ✅ 프로젝트 구조 생성
2. ⏳ 미니멀 테스트 스크립트 작성 및 검증
3. ⏳ LongBench 통합 예제 작성
4. ⏳ 비교 분석 스크립트 작성
5. ⏳ 실행 스크립트 및 문서 작성
6. ⏳ 전체 검증

## 검증 체크리스트

- [ ] Strategy 1 (Sink+Recent) 정상 동작
- [ ] Strategy 2 (Sink+Recent+Uniform) 정상 동작
- [ ] Flash Attention 활성화 확인
- [ ] Chunk 단위 입력 정상 동작
- [ ] 메모리 사용량 측정 가능
- [ ] LongBench 데이터셋에서 테스트 성공
- [ ] 두 전략 비교 결과 생성

## 참고사항

- LaCache 가상환경 경로: `../LaCache/.venv`
- 기존 코드 재사용: `../LaCache/LongBench/evict/simple_evict.py`
- Flash Attention 2.6.3 설치됨
- Transformers 4.43.4 설치됨
