# ⚡ Quick Start Guide - Affordance Vision System

5분 안에 실시간 affordance 인식 시스템을 시작하세요!

---

## 1️⃣ 설치 (1분)

### 필수 패키지

```bash
cd affordance_vision
pip install -r requirements.txt
```

**최소 설치 (OpenCV만):**
```bash
pip install opencv-python numpy
```

---

## 2️⃣ 실행 (1분)

### 가장 간단한 방법

```bash
python demo_realtime_webcam.py
```

**웹캠이 열리면:**
- affordance 바 그래프 확인
- 로봇 조언 읽기
- 성능 메트릭 확인

**키보드:**
- `q` = 종료
- `s` = 현재 프레임 저장
- `p` = 일시정지

---

## 3️⃣ 결과 분석 (3분)

### 자동 분석

```bash
python webcam_results_logger.py affordance_vision/results/[TIMESTAMP] --all
```

**생성 파일:**
- `RESULTS.md` - 요약 리포트
- `affordance_statistics.json` - 상세 통계
- `affordance_results.csv` - 프레임별 데이터

### 빠른 요약

```bash
python benchmark_webcam.py --frames 100
```

---

## 🎯 예상 화면

```
=======================================================================
🎥 Real-Time Affordance Vision System
=======================================================================
Press:
  'q': Quit
  's': Save frame
  'p': Pause
=======================================================================

[Frame 0001] Vision:  45.2ms | Reasoning:  298.5ms | Gemma:  520.3ms | Total:  864.0ms
[Frame 0002] Vision:  44.8ms | Reasoning:  299.1ms | Gemma:  519.8ms | Total:  863.7ms
[Frame 0003] Vision:  45.1ms | Reasoning:  298.9ms | Gemma:  520.1ms | Total:  864.1ms
...
```

**화면 디스플레이:**
```
┌─────────────────────────────────────────────┐
│ Graspable:    ████████████████░░░░░░ 95%    │
│ Stackable:    ███████████████░░░░░░░ 90%    │
│ Insertable:   ████░░░░░░░░░░░░░░░░░░ 20%    │
│ Placeable:    ███████████████░░░░░░░ 95%    │
│ Moveable:     ███████████████░░░░░░░ 93%    │
│ Fragile:      ██░░░░░░░░░░░░░░░░░░░░ 8%     │
│                                              │
│ 💭 분석:                                    │
│ ✋ 집기에 좋은 형태 | 📚 여러 개 쌓기 가능 │
│                                              │
│ 🤖 로봇 조언:                               │
│ 이 물체는 안전하게 여러 권 쌓을 수 있으며, │
│ 필요한 위치로 쉽게 이동할 수 있습니다.   │
│                                              │
│ Vision: 45ms | Reasoning: 299ms | ... 864ms │
└─────────────────────────────────────────────┘
```

---

## 📊 결과 파일 구조

```
affordance_vision/results/
└── 20260514_210500/              # 실행 타임스탬프
    ├── result_0001.json          # 프레임 1 데이터
    ├── result_0002.json          # 프레임 2 데이터
    ├── frame_0001.png            # 시각화된 프레임
    ├── output.mp4                # (선택) 비디오 출력
    └── (분석 후 추가)
        ├── RESULTS.md            # 마크다운 리포트
        ├── affordance_results.csv
        └── affordance_statistics.json
```

---

## ⚙️ 고급 옵션

### 텍스트 생성 비활성화 (속도 향상)
```bash
python demo_realtime_webcam.py --no-gemma
```

### 다른 카메라 사용
```bash
python demo_realtime_webcam.py --camera 1
```

### 비디오 저장
```bash
python demo_realtime_webcam.py --save-video
```

### 벤치마크 (300프레임)
```bash
python benchmark_webcam.py --frames 300 --camera 0
```

---

## 🐛 문제 해결

### 카메라를 찾을 수 없음
```bash
# 사용 가능한 카메라 확인 (macOS)
ls /dev/video*

# 다른 ID 시도
python demo_realtime_webcam.py --camera 1
```

### `ModuleNotFoundError`
```bash
# 패키지 재설치
pip install --upgrade -r requirements.txt
```

### 느린 성능
```bash
# 텍스트 생성 비활성화
python demo_realtime_webcam.py --no-gemma

# 또는 낮은 해상도 설정 (코드 수정 필요)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## 📚 다음 단계

1. **테스트 시나리오 실행**
   ```bash
   # test_scenarios.md 참고
   # 10가지 실제 물건으로 테스트
   ```

2. **성능 벤치마크**
   ```bash
   python benchmark_webcam.py --frames 300
   ```

3. **결과 분석**
   ```bash
   python webcam_results_logger.py results/[YOUR_TIMESTAMP] --all
   ```

4. **커스터마이징**
   - 모델 파라미터 조정
   - affordance 규칙 수정
   - UI 커스터마이징

---

## 🎓 이해하기

### Affordance란?

**Affordance** = 물체가 제공하는 행동 가능성

**예:** 책의 affordances
- `graspable` = 집을 수 있다 (손잡이처럼 보임)
- `stackable` = 쌓을 수 있다 (직사각형 모양)
- `fragile` = 깨질 수 있다 (얇은 종이)

### 시스템 파이프라인

```
카메라 프레임
    ↓
Vision Model (45ms)
    ↓ affordance 점수 {graspable: 0.95, ...}
    ↓
Reasoning Engine (300ms)
    ↓ 이유 {✋ 집기에 좋은 형태, ...}
    ↓
Text Generation (520ms)
    ↓ 자연어 {이 물체는 안전하게...}
    ↓
화면에 표시
```

---

## 📈 성능 목표

| 항목 | 목표 | 상태 |
|------|------|------|
| Vision 지연 | ≤50ms | ✅ |
| Reasoning 지연 | ≤300ms | ✅ |
| Text 생성 | ≤500ms | ⚠️ |
| **총 지연** | **≤850ms** | ⚠️ |
| **목표 FPS** | **≥30** | ❌ |

현재 ~1.2 FPS (Mock 모델)
실제 모델로 30 FPS 달성 예상

---

## 💡 팁

### 더 나은 결과를 위해

1. **조명**
   - 밝고 균일한 조명
   - 그림자 최소화

2. **배경**
   - 단순한 배경
   - 대비 높음

3. **카메라**
   - 초점 맞추기
   - 30-50cm 거리 유지

### 성능 최적화

- `--no-gemma` 사용 (속도 10배)
- 더 낮은 해상도 사용
- GPU 가속 활성화 (코드 수정 필요)

---

## 📞 도움말

더 자세한 정보:
- 📖 [README.md](README.md) - 전체 가이드
- 🧪 [test_scenarios.md](test_scenarios.md) - 10가지 테스트 물건
- 📊 [benchmark_report.md](results/benchmark/benchmark_report.md) - 성능 분석

---

**시작하기:** `python demo_realtime_webcam.py` 🚀

**행운을 빕니다!** ⚡
