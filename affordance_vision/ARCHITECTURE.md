# 🏗️ System Architecture - Affordance Vision System

웹캠 기반 실시간 affordance 인식 시스템의 상세 아키텍처

---

## 📐 고수준 구조도

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│                   (Webcam / Video File)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Frame Capture Module  │
         │  (OpenCV VideoCapture) │
         │  Resolution: 1280x720  │
         │  FPS: 30 (target)      │
         └──────────┬─────────────┘
                    │ 1280x720 frames @ 30fps
                    ▼
    ┌───────────────────────────────────────┐
    │   VISION PIPELINE (demo_realtime.py)  │
    │                                       │
    │  ┌─────────────────────────────────┐  │
    │  │  1. VAE Vision Model            │  │
    │  │  ├─ Input: RGB Image (1280x720)│  │
    │  │  ├─ Processing: CNN Encoder    │  │
    │  │  ├─ Latency: ~45ms             │  │
    │  │  └─ Output: 6 affordance scores│  │
    │  │    {graspable, stackable, ...} │  │
    │  └──────────────┬──────────────────┘  │
    │                 │                      │
    │                 │ affordance vector   │
    │                 ▼                      │
    │  ┌─────────────────────────────────┐  │
    │  │  2. Reasoning Engine            │  │
    │  │  ├─ Input: Affordance scores   │  │
    │  │  ├─ Processing: Rule-based     │  │
    │  │  ├─ Latency: ~300ms            │  │
    │  │  └─ Output: Korean explanations│  │
    │  │    (e.g., "집기 쉬운 형태")     │  │
    │  └──────────────┬──────────────────┘  │
    │                 │                      │
    │                 │ reasoning text      │
    │                 ▼                      │
    │  ┌─────────────────────────────────┐  │
    │  │  3. Text Generation (Gemma 2B)  │  │
    │  │  ├─ Input: affordance + reason  │  │
    │  │  ├─ Processing: LLM inference   │  │
    │  │  ├─ Latency: ~500ms            │  │
    │  │  └─ Output: Natural language    │  │
    │  │    advice (2-3 lines)           │  │
    │  └──────────────┬──────────────────┘  │
    └─────────────────┼─────────────────────┘
                      │
                      │ full result
                      ▼
    ┌───────────────────────────────────────┐
    │   OUTPUT & STORAGE                    │
    │                                       │
    │  ├─ Visualization (OpenCV)           │
    │  │  └─ Overlays: bars, text, stats   │
    │  │                                   │
    │  ├─ Frame Saving (PNG)               │
    │  │  └─ Results/[TIMESTAMP]/frame_*.png│
    │  │                                   │
    │  ├─ Result Logging (JSON)            │
    │  │  └─ Results/[TIMESTAMP]/result_*. │
    │  │     json                          │
    │  │                                   │
    │  └─ Video Recording (MP4)            │
    │     └─ Results/[TIMESTAMP]/output.mp4│
    └───────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────┐
    │   ANALYSIS & REPORTING                │
    │   (webcam_results_logger.py)          │
    │                                       │
    │  ├─ Load Results (JSON)              │
    │  ├─ Calculate Statistics            │
    │  ├─ Export Formats:                 │
    │  │  ├─ CSV (spreadsheet)            │
    │  │  ├─ JSON (detailed stats)        │
    │  │  └─ Markdown (report)            │
    │  └─ Generate Graphs & Trends        │
    └───────────────────────────────────────┘
```

---

## 🔧 모듈 상세 설명

### 1. Vision Module (VAE)

**파일:** `demo_realtime_webcam.py` - `MockVAEModel` 클래스

**책임:**
- 웹캠 프레임을 affordance 점수로 변환
- 6가지 affordance 감지

**입력:**
```python
frame: np.ndarray  # BGR 이미지, 1280x720, uint8
```

**출력:**
```python
affordances: Dict[str, float]
{
    'graspable': 0.95,      # 0-1 범위
    'stackable': 0.90,
    'insertable': 0.20,
    'placeable': 0.95,
    'moveable': 0.93,
    'fragile': 0.08,
}
```

**처리 단계:**
```
RGB Image
    ↓
Grayscale Conversion
    ↓
Edge Detection (Canny)
    ↓
Feature Extraction
    ↓
6 Affordance Scores
```

**성능:**
- 지연: ~45ms
- 처리량: ~22 FPS (단독)

**확장 가능성:**
- 실제 CNN 모델로 교체
- 배치 처리 추가
- GPU 가속

---

### 2. Reasoning Engine

**파일:** `demo_realtime_webcam.py` - `MockReasoningEngine` 클래스

**책임:**
- affordance 점수에서 논리적 규칙 적용
- 한국어 설명 생성

**입력:**
```python
affordances: Dict[str, float]
```

**출력:**
```python
reasoning: str
# 예시:
# "✋ 집기에 좋은 형태 | 📚 여러 개를 쌓을 수 있음"
```

**규칙 시스템:**
```
if graspable > 0.7:
    reasons.append("✋ 집기에 좋은 형태")

if stackable > 0.7:
    reasons.append("📚 여러 개를 쌓을 수 있음")

if fragile > 0.5:
    reasons.append("⚠️ 깨질 수 있으니 주의 필요")

if moveable > 0.7:
    reasons.append("🚚 움직이기 좋음")
```

**성능:**
- 지연: <1ms (계산량 적음)
- 처리량: 매우 빠름

---

### 3. Text Generation (Gemma 2B)

**파일:** `demo_realtime_webcam.py` - `MockOllamaClient` 클래스

**책임:**
- affordance + reasoning에서 자연어 조언 생성
- Ollama를 통해 Gemma 2B 모델 실행

**입력:**
```python
prompt: str
# """당신은 로봇 비전 분석 전문가입니다.
# Affordance 점수:
# - Graspable: 95%
# - ...
# 이 물체를 로봇이 어떻게 다뤄야 하는지 
# 2-3줄로 설명하세요."""
```

**출력:**
```python
response: str
# "이 물체는 안전하게 여러 권 쌓을 수 있으며,
#  필요한 위치로 쉽게 이동할 수 있습니다."
```

**성능:**
- 지연: ~500ms (토큰 생성)
- 모델 크기: 2B parameters
- 메모리: ~4GB (양자화 시 ~2GB)

**실행 환경:**
```bash
ollama pull gemma:2b
ollama serve  # 별도 터미널에서 실행
```

---

### 4. Result Logger (분석 도구)

**파일:** `webcam_results_logger.py` - `AffordanceResultsLogger` 클래스

**책임:**
- JSON 결과 파일 로드
- 통계 계산
- 다양한 형식으로 내보내기

**입력:**
```python
results_dir: Path
# 디렉토리 내 result_*.json 파일들
```

**출력 형식:**

1. **CSV**
   ```
   frame,vision_ms,reasoning_ms,gemma_ms,total_ms,graspable,stackable,...
   1,45.2,298.5,520.3,864.0,0.95,0.90,...
   2,44.8,299.1,519.8,863.7,0.92,0.88,...
   ```

2. **JSON**
   ```json
   {
     "metadata": {
       "total_frames": 300,
       "timestamp": "2026-05-14T21:07:22"
     },
     "latency_stats": {
       "vision_ms": {"mean": 45.2, "median": 45.0, ...},
       ...
     },
     "affordance_stats": {...},
     "confidence_stats": {...}
   }
   ```

3. **Markdown**
   ```markdown
   # Affordance Vision Results
   
   | Component | Mean | Median | Min | Max |
   |-----------|------|--------|-----|-----|
   | Vision | 45.2 | 45.0 | 44.2 | 47.9 |
   | ...
   ```

---

## 🔄 데이터 흐름

### 실시간 실행 (demo_realtime_webcam.py)

```
┌─────────────────────────────────────┐
│ Main Loop (run method)              │
│                                     │
│ while True:                         │
│   frame = cap.read()                │
│   ↓                                 │
│   result = process_frame(frame)     │
│   ↓                                 │
│   vis = visualize_result(frame)     │
│   ↓                                 │
│   cv2.imshow(vis)                   │
│   ↓                                 │
│   save_result_json()                │
│   ↓                                 │
│   print stats                       │
│                                     │
└─────────────────────────────────────┘
```

### 결과 처리 (webcam_results_logger.py)

```
┌──────────────────────────────────────┐
│ Analysis Workflow                    │
│                                      │
│ load_results()                       │
│   └─ Read all result_*.json files    │
│      └─ Store in self.results list   │
│                                      │
│ calculate_statistics()               │
│   ├─ latency_stats()                 │
│   ├─ affordance_stats()              │
│   ├─ confidence_stats()              │
│   └─ temporal_trends()               │
│                                      │
│ export_csv/json/markdown()           │
│   └─ Generate reports                │
│                                      │
└──────────────────────────────────────┘
```

---

## 📊 데이터 구조

### Frame Result JSON

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

### Statistics JSON

```json
{
  "metadata": {
    "total_frames": 300,
    "timestamp": "2026-05-14T21:07:22"
  },
  "latency_stats": {
    "vision_ms": {
      "mean": 45.2,
      "median": 45.0,
      "min": 44.2,
      "max": 47.9,
      "stdev": 1.2,
      "p95": 47.3,
      "p99": 47.8
    },
    "reasoning_ms": {...},
    "gemma_ms": {...},
    "total_ms": {...}
  },
  "affordance_stats": {
    "graspable": {...},
    "stackable": {...},
    ...
  }
}
```

---

## 🎨 시각화 레이아웃

```
┌────────────────────────────────────────────────┐
│  Real-Time Affordance Vision                   │
├────────────────────────────────────────────────┤
│ Graspable:    ████████████████░░░░░░ 95%       │
│ Stackable:    ███████████████░░░░░░░ 90%       │
│ Insertable:   ████░░░░░░░░░░░░░░░░░░ 20%       │
│ Placeable:    ███████████████░░░░░░░ 95%       │
│ Moveable:     ███████████████░░░░░░░ 93%       │
│ Fragile:      ██░░░░░░░░░░░░░░░░░░░░ 8%        │
│                                                 │
│ 💭 분석:                                       │
│ ✋ 집기에 좋은 형태 | 📚 여러 개 쌓기 가능    │
│                                                 │
│ 🤖 로봇 조언:                                  │
│ 이 물체는 안전하게 여러 권 쌓을 수 있으며,   │
│ 필요한 위치로 쉽게 이동할 수 있습니다.       │
│                                                 │
│ Vision: 45ms | Reasoning: 299ms | Total: 864ms│
└────────────────────────────────────────────────┘
```

---

## ⚙️ 설정 및 파라미터

### 카메라 설정

```python
# demo_realtime_webcam.py
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 해상도
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)              # 목표 FPS
```

### Affordance 임계값 (Reasoning)

```python
# MockReasoningEngine.think()
if affordances['graspable'] > 0.7:         # 70% 이상이면 "집기 가능"
    reasons.append("✋ 집기에 좋은 형태")
```

### 텍스트 생성 프롬프트

```python
# demo_realtime_webcam.py._create_prompt()
prompt = f"""당신은 로봇 비전 분석 전문가입니다.
Affordance 점수:
- Graspable: {affordances['graspable']:.0%}
...
이 물체를 로봇이 어떻게 다뤄야 하는지 2-3줄로 설명하세요."""
```

---

## 🔌 확장 포인트

### 1. 실제 Vision 모델 연결

```python
# demo_realtime_webcam.py의 MockVAEModel을 교체
class RealVAEModel:
    def predict(self, frame):
        # 실제 CNN 모델 호출
        return real_model.infer(frame)
```

### 2. 커스텀 Reasoning 규칙

```python
class CustomReasoningEngine:
    def think(self, affordances):
        # 도메인별 규칙 추가
        # e.g., 의료용, 산업용, 등
        pass
```

### 3. 다른 LLM 모델

```python
class CustomLLMClient:
    def generate(self, model, prompt):
        # LLaMA, Mistral, 등 다른 모델 사용
        pass
```

### 4. 데이터베이스 연결

```python
def save_result_db(result):
    # SQLite/PostgreSQL로 저장
    # 장기 분석/트렌드 추적용
    pass
```

---

## 📈 성능 최적화 전략

### 1단계: Profiling (현재)
- 각 컴포넌트 지연시간 측정
- 병목 지점 식별

### 2단계: 모델 최적화
- 양자화 (FP32 → INT8)
- 프루닝 (불필요한 가중치 제거)
- 증류 (큰 모델 → 작은 모델)

### 3단계: 구현 최적화
- SIMD 명령어 활용
- 멀티스레딩
- 배치 처리

### 4단계: 하드웨어 가속
- GPU (CUDA/Metal)
- TPU (Google)
- 특화 하드웨어 (Vision Transformer 칩)

---

## 🧪 테스트 아키텍처

```
┌──────────────────────────────────────┐
│ Test Scenarios (test_scenarios.md)   │
│                                      │
│ 10개 물건 테스트                     │
│ ├─ 책                                │
│ ├─ 유리잔                            │
│ ├─ 공                                │
│ ├─ 열쇠                              │
│ ├─ 박스                              │
│ ├─ 타올                              │
│ ├─ 컵                                │
│ ├─ 휴대폰                            │
│ ├─ 나무 블록                         │
│ └─ 계란                              │
│                                      │
│ 각 물건에 대해:                      │
│ ├─ 예상 affordance 점수              │
│ ├─ 정확도 평가                       │
│ ├─ 신뢰도 평가                       │
│ └─ 응답시간 측정                     │
│                                      │
└──────────────────────────────────────┘
```

---

## 🚀 배포 시나리오

### 단계 1: 로컬 개발
```bash
python demo_realtime_webcam.py --camera 0
```

### 단계 2: 성능 검증
```bash
python benchmark_webcam.py --frames 300
```

### 단계 3: 결과 분석
```bash
python webcam_results_logger.py results/[TIMESTAMP] --all
```

### 단계 4: 로봇 시스템 통합
```python
from demo_realtime_webcam import RealtimeAffordanceSystem

system = RealtimeAffordanceSystem()
result = system.process_frame(camera_frame)

# 로봇 제어 명령어 생성
command = generate_robot_command(result['affordances'])
robot.execute(command)
```

---

**Last Updated:** 2026-05-14  
**Version:** 1.0.0  
**Status:** Architecture Finalized ✅
