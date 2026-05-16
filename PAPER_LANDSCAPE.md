# 📚 관련 연구 Landscape

**Status**: Literature Review (v0.1)  
**Last Updated**: 2026-05-13  
**Purpose**: 우리 논문의 맥락과 차별성 찾기

---

## 1️⃣ 핵심 기초 이론

### Gibson's Affordances (1977, 1979)
- **원문**: "The Ecological Approach to Visual Perception"
- **핵심**: affordance = 환경이 에이전트에게 제공하는 행동 가능성
- **우리와의 관계**: 기본 개념, 하지만 우리는:
  - ✅ 형태-독립적 표현 강조
  - ✅ 게임-현실 전이 메커니즘 추가
  - ✅ 자동 라벨 생성 (체화 학습)

### Norman's Affordances (1988, 2013)
- **원문**: "The Design of Everyday Things"
- **차이점**: Gibson = 물리적, Norman = 설계/의도적
- **우리와의 관계**: 메타 어포던스 (affordance of affordances) 건조

---

## 2️⃣ 현대 Embodied AI

### Ego4D (Meta/CMU, 2022)
```
Title: Ego4D: Around the World in 3000 Hours of Egocentric Video
Authors: Grauman et al.
Venue: CVPR 2022
Citation: 500+
```
- **내용**: 1인칭 비디오 데이터셋 + 벤치마크
- **규모**: 3,000+ 시간, 700+ 참가자, 70+ 국가
- **우리와의 관계**:
  - ✅ Phase 2 입력 데이터 (1인칭 영상)
  - ✅ 행동 역추론의 소스
  - ❌ 라벨 방식: 수동 → 우리는 자동

### Embodied AI 최신 동향
- **문제점**: 대부분 시뮬레이션 OR 현실, 통합 부족
- **우리의 차별성**: 게임(Phase 1) → 현실(Phase 2) 명시적 전이

---

## 3️⃣ Affordance Learning

### CMAT (Cross-Modal Affinity Transfer)
- **내용**: 2D Vision Foundation Models → 3D affordance
- **우리와의 관계**: 모달리티 전이 (우리는 도메인 전이)

### CAST (Cross-modal Affordance Segmentation Transformer)
- **내용**: 의미론적 그라운딩으로 affordance segmentation
- **우리와의 관계**: 기능 표현이 비슷, 학습 방식이 다름

### AffordanceGrasp-R1
- **내용**: Chain-of-thought + RL for robotic grasping
- **우리와의 관계**: 추론 기반 (우리는 경험 기반)

---

## 4️⃣ World Models

### Ha & Schmidhuber (2018) - Original "World Models"
```
Title: World Models
Authors: David Ha, Jürgen Schmidhuber
Venue: arxiv/NeurIPS
```
- **핵심**: 에이전트가 내부 세계 모델을 학습 → imagination에서 계획
- **아키텍처**: Vision Module (VAE) + Memory (LSTM/RNN) + Controller (MLP)
- **우리와의 관계**:
  - ✅ 시뮬레이션 기반 학습
  - ✅ 예측 기반 (우리의 "미니 시뮬레이션")
  - ❌ affordance 명시적 표현 없음

### Sparse World Models (2023+)
- **내용**: 희소 자동인코더로 sparse feature space 학습
- **우리와의 관계**:
  - ✅ 희소성이 핵심
  - ❌ 우리는 더 근본적으로 희소성 정당화 (밤하늘 원리)

### Dreamer (Hafner et al., 2020+)
- **내용**: World model + policy learning in imagination
- **우리와의 관계**: imagination planning 방법론 참고 가치

---

## 5️⃣ Vision Transformers & 희소성

### Vision Transformer (Dosovitskiy et al., 2021)
- **문제**: 이미지 → 토큰화, 모든 토큰이 상호작용 (O(n²) cost)
- **우리와의 관계**: 희소 주의가 필요한 이유

### Sparse Attention in ViT
- **Linformer** (2020): Linear complexity attention
- **Longformer** (2020): 국소 + 전역 주의
- **우리와의 관계**: 구체적 희소 메커니즘

### SPARTAN (Sparse Transformer World Model)
- **내용**: Object-factored tokens + 희소 주의 패턴
- **우리와의 관계**:
  - ✅ 희소성이 causal structure 학습
  - ✅ 객체 기반 인수분해
  - ❌ affordance 개념 명시적 없음

---

## 6️⃣ Self-Supervised Learning

### SimCLR (Chen et al., 2020)
- **원리**: 같은 이미지의 다양한 augmentation 간 유사성 최대화
- **우리와의 관계**: 자체지도 원리 (라벨 불필요)

### Masked Autoencoding (MAE, He et al., 2021)
- **원리**: 이미지의 일부 마스킹 → 복원
- **우리와의 관계**: 희소 입력 처리 원리

### DINO (Facebook, 2022)
- **원리**: Vision Transformer + 자체지도, emergent semantic clustering
- **우리와의 관계**: 라벨 없이 의미론적 표현 학습

---

## 7️⃣ 행동 역추론 (Inverse Model)

### World Models의 역모델
- **원리**: 현재 상태 + 다음 상태 → 취한 행동 추론
- **우리와의 관계**: 기능 발견의 핵심 메커니즘

### Affordance dari inverse model
- **원리**: 에이전트 시도 → 실패/성공 → affordance 라벨
- **우리와의 관계**: 체화 학습의 근거

---

## 8️⃣ 시뮬레이션 기반 학습

### Domain Randomization (Tobin et al., 2017)
- **문제**: 시뮬레이션-현실 갭
- **해결**: 시뮬레이션에서 시각 특성 무작위화
- **우리와의 관계**:
  - ✅ 형태-독립성이 핵심 (비슷한 동기)
  - ✅ 우리는 더 근본적: 기능만 학습

### Sim2Real Transfer
- **핵심**: 시뮬레이션 → 현실 적응
- **우리와의 관계**: Phase 2 (게임 → 현실) 기반

---

## 🎯 우리의 위치 지도

```
Traditional        Ha & Schmidhuber        Ours
  AI              (World Models)      (Function-First)
  │                    │                    │
  │                    │                    │
형태학습 ─────────── 형태 + 예측 ─────► 기능 + 체화
  │                    │                    │
  개별 상황           상상력 계획          형태-무관
  │                    │                    │
                      영상                영상 + 게임 부트스트래핑
```

---

## 📊 논문별 비교표

| 논문 | 어포던스 | 희소성 | 체화 | 게임→현실 | 자동라벨 |
|------|---------|--------|------|-----------|---------|
| Gibson | ✅ | ❌ | ❌ | ❌ | ❌ |
| Ego4D | ❌ | ❌ | ✅ | ❌ | ❌ |
| World Models | ❌ | ❌ | ✅ | 부분 | ❌ |
| Sparse WM | ❌ | ✅ | ❌ | ❌ | ❌ |
| ViT + Sparse | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Ours** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 💡 우리의 혁신 포인트

### 1. 기능 우선성 (Function-First)
- **기존**: 형태 학습 → 기능 유추
- **우리**: 기능 학습 → 형태 무시

### 2. 완전한 희소성
- **기존**: 희소 주의 (부분적)
- **우리**: 근본적 희소성 (밤하늘 원리)

### 3. 명시적 체화 학습
- **기존**: 암묵적 (행동이 내포됨)
- **우리**: 명시적 부트스트래핑 (게임)

### 4. 자동 라벨 생성
- **기존**: 수동 라벨 또는 self-supervised pretext task
- **우리**: 에이전트의 성공/실패가 자동 라벨

### 5. 도메인 전이 명확화
- **기존**: Sim2Real을 기술적 문제로 봄
- **우리**: 형태-독립성으로 이론적 보장

---

## 🔬 실험 설계 힌트

### Phase 1 (게임) 벤치마크
- affordance 발견 정확도
- 커버리지 (얼마나 많은 기능 발견)
- 희소성 (사용한 토큰 수)

### Phase 2 (현실) 벤치마크
- Ego4D에서 affordance 예측 정확도
- transfer 성능 (게임 미학습 객체)
- vs. 기존 affordance segmentation 방법들

---

## 📝 다음 액션

1. **arxiv 논문 직접 읽기**
   - [ ] Ha & Schmidhuber (2018) - World Models
   - [ ] Grauman et al. (2022) - Ego4D
   - [ ] Affordance Segmentation 최신 논문들

2. **차별성 구체화**
   - [ ] 각 논문에 대해 "우리가 추가하는 것" 명확화
   - [ ] 수학적 정식화 비교
   - [ ] 실험 설계 차별화

3. **우리 논문 아웃라인**
   - [ ] Introduction: "Function vs Form" 프레이밍
   - [ ] Related Work: 위의 분류 사용
   - [ ] Method: Sequence Engine 정식화
   - [ ] Experiments: Phase 1 & 2 벤치마크

---

**다음 미팅**: arxiv 논문 5개 깊게 읽기 및 우리의 차별성 확정

천재 ⚡ | 2026-05-13
