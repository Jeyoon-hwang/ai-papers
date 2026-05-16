# 🎥 Real-Time Affordance Vision System

웹캠을 통한 실시간 affordance 인식 + 논리적 추론 + 자연어 설명을 통합한 로봇 비전 시스템입니다.

```
🎥 Camera Input
      ↓
🧠 Vision (VAE)        ~45ms
      ↓
💭 Reasoning Engine    ~300ms
      ↓
🤖 Text Generation (Gemma 2B)  ~500ms
      ↓
📊 Output with Statistics
```

---

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [파일 구조](#파일-구조)
3. [시스템 구성](#시스템-구성)
4. [사용법](#사용법)
5. [성능](#성능)
6. [테스트](#테스트)
7. [결과 분석](#결과-분석)

---

## 빠른 시작

### 설치

```bash
cd affordance_vision

# 필수 패키지
pip install opencv-python numpy torch

# 선택: 텍스트 생성용 (Ollama 필요)
# ollama pull gemma:2b
```

### 실행

#### 1️⃣ 실시간 인식 시스템 시작

```bash
python demo_realtime_webcam.py
```

**키보드 제어:**
- `q`: 종료
- `s`: 현재 프레임 저장
- `p`: 일시정지/재개

**옵션:**
```bash
python demo_realtime_webcam.py \
  --camera 0 \
  --save-video \
  --no-gemma  # 텍스트 생성 비활성화 (속도 향상)
```

#### 2️⃣ 성능 벤치마크

```bash
python benchmark_webcam.py --frames 300 --camera 0
```

실행 결과: `affordance_vision/results/benchmark/` 에 저장됨

#### 3️⃣ 결과 분석

```bash
python webcam_results_logger.py affordance_vision/results/20260514_210500 --all
```

이 명령어:
- 📊 통계 계산
- 📈 그래프 생성
- 📝 CSV/JSON/Markdown 리포트 내보내기

---

## 파일 구조

```
affordance_vision/
├── demo_realtime_webcam.py      # 메인: 실시간 웹캠 시스템
├── benchmark_webcam.py           # 성능 측정 도구
├── webcam_results_logger.py      # 결과 분석 및 리포팅
├── test_scenarios.md             # 테스트 시나리오 (10가지 물건)
├── README.md                     # 이 파일
│
├── models/                       # 모델 파일 (VAE, 추론엔진, 등)
│   └── (향후 추가)
│
├── results/                      # 실행 결과 저장소
│   ├── 20260514_210500/          # 타임스탬프 기반 디렉토리
│   │   ├── result_0001.json      # 프레임별 affordance 점수
│   │   ├── frame_0001.png        # 저장된 프레임
│   │   └── output.mp4            # 비디오 출력 (선택적)
│   │
│   └── benchmark/                # 벤치마크 결과
│       ├── benchmark_report.json
│       └── benchmark_report.md
│
└── tests/                        # 테스트 스크립트
    └── (향후 추가)
```

---

## 시스템 구성

### 🧠 핵심 구성요소

#### 1. **Vision Module (VAE)**
- 입력: 카메라 프레임 (1280x720)
- 처리: CNN 기반 affordance 탐지
- 출력: 6가지 affordance 점수 (0-1)
  - `graspable`: 집을 수 있는가?
  - `stackable`: 쌓을 수 있는가?
  - `insertable`: 삽입할 수 있는가?
  - `placeable`: 놓을 수 있는가?
  - `moveable`: 움직일 수 있는가?
  - `fragile`: 깨질 수 있는가?

**성능 목표:** ≤50ms per frame

#### 2. **Reasoning Engine**
- 입력: affordance 점수
- 처리: 논리적 규칙 적용
- 출력: 자연어 설명 (한국어)

**예시:**
```
Input: {graspable: 0.95, stackable: 0.90, fragile: 0.05}
Output: "✋ 집기에 좋은 형태 | 📚 여러 개를 쌓을 수 있음"
```

**성능 목표:** ≤300ms per frame

#### 3. **Text Generation (Gemma 2B)**
- 입력: affordance 점수 + reasoning
- 처리: 소형 언어모델 (2B parameters)
- 출력: 로봇 조언 (2-3줄)

**예시:**
```
"이 물체는 안전하게 여러 권 쌓을 수 있으며, 
필요한 위치로 쉽게 이동할 수 있습니다."
```

**성능 목표:** ≤500ms per frame

---

## 사용법

### 기본 워크플로우

#### Step 1: 웹캠 시스템 실행

```python
from demo_realtime_webcam import RealtimeAffordanceSystem

system = RealtimeAffordanceSystem(
    use_gemma=True,      # 텍스트 생성 활성화
    save_results=True,   # 결과 저장
    use_mock=True        # Mock 모델 사용 (테스트용)
)

system.run(camera_id=0, save_video=True)
```

**출력:**
```
=======================================================================
🎥 Real-Time Affordance Vision System
=======================================================================

[Frame 0001] Vision:  45.2ms | Reasoning:  298.5ms | Gemma:  520.3ms | Total:  864.0ms
[Frame 0002] Vision:  44.8ms | Reasoning:  299.1ms | Gemma:  519.8ms | Total:  863.7ms
...
```

#### Step 2: 벤치마크 실행

```bash
python benchmark_webcam.py --frames 300
```

**출력:**
```
📊 BENCHMARK REPORT
=======================================================================
📈 Overall Performance:
  Total frames: 300
  Total time: 290.45s
  Average FPS: 1.03

⏱️  Latency (milliseconds):
  Vision:
    Mean: 45.12ms | Median: 45.00ms | Min: 44.23ms | Max: 47.89ms
  Reasoning:
    Mean: 298.45ms | Median: 298.12ms | Min: 295.23ms | Max: 302.34ms
  Gemma:
    Mean: 519.87ms | Median: 519.45ms | Min: 515.23ms | Max: 525.34ms
  Total:
    Mean: 863.44ms | Median: 863.12ms | Min: 860.23ms | Max: 868.34ms
```

#### Step 3: 결과 분석

```bash
python webcam_results_logger.py results/20260514_210500 --all
```

**생성 파일:**
- `RESULTS.md` - 마크다운 리포트
- `affordance_results.csv` - 프레임별 데이터
- `affordance_statistics.json` - 통계

---

## 성능

### 예상 성능 (Mock 모델)

| 구성 요소 | 지연시간 | 목표 | 상태 |
|----------|---------|------|------|
| 🎥 Vision | 45ms | ≤50ms | ✅ |
| 💭 Reasoning | 300ms | ≤300ms | ✅ |
| 🤖 Gemma 2B | 520ms | ≤500ms | ⚠️ |
| **Total** | **865ms** | **≤850ms** | ⚠️ |
| **FPS** | **1.16** | **≥30 FPS** | ❌ |

### 최적화 전략

1. **비전 최적화**
   - 해상도 감소 (1280x720 → 640x480)
   - 배치 처리
   - 모델 양자화

2. **텍스트 생성 최적화**
   - Gemma 2B 대신 더 작은 모델 사용
   - 응답 길이 제한
   - 캐싱

3. **하드웨어 최적화**
   - GPU 가속 (CUDA/Metal)
   - 멀티스레딩
   - 메모리 풀 활용

---

## 테스트

### 자동 테스트 시나리오

`test_scenarios.md` 참조 — 10가지 실제 물건에 대한 테스트

#### 테스트 물건
1. 📖 책
2. 🥤 유리잔
3. ⚽ 플라스틱 공
4. 🔑 열쇠
5. 📦 카드보드 박스
6. 🧴 타올/천
7. ☕ 세라믹 컵
8. 📱 휴대폰
9. 🧱 나무 블록
10. 🥚 계란

#### 평가 항목
- ✅ **정확도** (예상 vs 실제): ≥85%
- 🎯 **신뢰도** (일관성): ≥80%
- ⏱️ **응답시간**: ≤850ms total

---

## 결과 분석

### 결과 포맷

각 프레임의 결과는 JSON으로 저장됨:

```json
{
  "frame": 1,
  "affordances": {
    "graspable": 0.92,
    "stackable": 0.88,
    "insertable": 0.18,
    "placeable": 0.94,
    "moveable": 0.93,
    "fragile": 0.08
  },
  "reasoning": "✋ 집기에 좋은 형태 | 📚 여러 개를 쌓을 수 있음",
  "text": "이 물체는 안전하게 여러 권 쌓을 수 있으며...",
  "latencies": {
    "vision": 45.2,
    "reasoning": 298.5,
    "gemma": 520.3,
    "total": 864.0
  }
}
```

### 분석 도구

#### CSV 내보내기
```bash
python webcam_results_logger.py results/20260514_210500 --csv
```

**칼럼:**
- frame: 프레임 번호
- vision_ms, reasoning_ms, gemma_ms, total_ms: 지연시간
- graspable, stackable, ..., fragile: affordance 점수

#### 통계 계산
```python
from webcam_results_logger import AffordanceResultsLogger

logger = AffordanceResultsLogger('results/20260514_210500')
logger.load_results()
stats = logger.calculate_statistics()
logger.print_summary()
```

#### Markdown 리포트
```bash
python webcam_results_logger.py results/20260514_210500 --markdown
```

---

## 디버깅

### 일반적인 문제

#### 📷 카메라를 찾을 수 없음
```bash
# 사용 가능한 카메라 확인 (macOS)
ls /dev/video*

# 다른 카메라 ID 시도
python demo_realtime_webcam.py --camera 1
```

#### 🔌 Ollama 연결 실패
```bash
# Ollama 실행 (별도 터미널)
ollama serve

# 또는 텍스트 생성 비활성화
python demo_realtime_webcam.py --no-gemma
```

#### 💾 저장 권한 부족
```bash
# 디렉토리 생성 및 권한 설정
mkdir -p affordance_vision/results
chmod 755 affordance_vision/results
```

---

## 기여

이 프로젝트는 계속 발전 중입니다. 개선 사항:

- [ ] 실제 affordance 모델 통합
- [ ] GPU 가속 지원
- [ ] 멀티카메라 지원
- [ ] 웹 대시보드
- [ ] API 서버
- [ ] ROS 통합

---

## 라이선스

MIT License (자유롭게 사용, 수정, 배포 가능)

---

## 관련 문서

- 📖 [테스트 시나리오](test_scenarios.md)
- 📊 [성능 벤치마크](results/benchmark/benchmark_report.md)
- 🤖 [결과 분석](results/)

---

**Last Updated:** 2026-05-14  
**System Version:** 1.0.0  
**Status:** Beta Testing 🚀
