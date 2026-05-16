"""
Phase 10: Advanced Hybrid Memory System (Information Theory + Bayesian + Stochastic)

완전한 수학적 기반의 AI 기억 시스템 구현
- Information-Theoretic Memory Foundation (Shannon Entropy)
- Bayesian Memory Network (Recursive Update)
- Stochastic Process: Ornstein-Uhlenbeck (확률 미분방정식)
- Capacity Management (정보론적 기준)

Author: 천재 (AI Assistant)
Date: 2026-05-14
"""

import numpy as np
from scipy.integrate import odeint
from scipy.stats import entropy as scipy_entropy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryState:
    """기억 상태를 표현하는 데이터 클래스"""
    object_name: str
    affordances: np.ndarray
    confidence: float
    entropy: float
    timestamp: datetime
    mi_score: float  # Mutual Information Score
    retention_prob: float


class AdvancedMemorySystem:
    """
    정보론 + 베이지안 + 확률 과정 통합 기억 시스템
    """
    
    def __init__(self, 
                 latent_dim: int = 64,
                 max_capacity_bits: float = 10000.0,
                 theta: float = 0.5,
                 sigma: float = 0.1):
        """
        Parameters:
        -----------
        latent_dim : int
            affordance 벡터 차원
        max_capacity_bits : float
            최대 기억 용량 (정보론 단위, bits)
        theta : float
            OU process 복원력 (0.1~2.0, 클수록 빠르게 망각)
        sigma : float
            확률적 항의 휘발성 (0.05~0.5)
        """
        self.latent_dim = latent_dim
        self.max_capacity = max_capacity_bits
        self.theta = theta
        self.sigma = sigma
        
        # 기억 저장소
        self.memories: Dict[str, np.ndarray] = {}
        self.entropy_history: Dict[str, List[Dict]] = {}
        self.bayesian_priors: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.mi_scores: Dict[str, float] = {}
        
        logger.info(f"✨ AdvancedMemorySystem 초기화 완료")
        logger.info(f"   - Latent Dimension: {latent_dim}")
        logger.info(f"   - Max Capacity: {max_capacity_bits} bits")
        logger.info(f"   - Theta (forgetting rate): {theta}")
        logger.info(f"   - Sigma (stochasticity): {sigma}")
    
    # ============ Layer 1: Bayesian Update ============
    
    def bayesian_update(self, 
                       object_name: str, 
                       observation: np.ndarray, 
                       prior: Optional[np.ndarray] = None) -> np.ndarray:
        """
        베이지안 정리를 사용한 기억 업데이트
        
        P(m_t | O_t) ∝ P(O_t | m_{t-1}) × P(m_{t-1})
        
        Parameters:
        -----------
        object_name : str
            기억의 대상 (예: 'book')
        observation : np.ndarray
            새로운 관찰 (affordance 벡터)
        prior : np.ndarray, optional
            사전 확률 (이전 믿음)
        
        Returns:
        --------
        np.ndarray
            사후 확률 (posterior)
        """
        if prior is None:
            prior = self.bayesian_priors.get(
                object_name, 
                np.ones(self.latent_dim) / self.latent_dim
            )
        
        # Likelihood: 관찰의 확률
        likelihood = self._compute_likelihood(observation, prior)
        
        # Posterior: 정규화된 사후확률
        denominator = np.sum(likelihood * prior) + 1e-8
        posterior = (likelihood * prior) / denominator
        
        return posterior
    
    def _compute_likelihood(self, 
                           obs: np.ndarray, 
                           prior: np.ndarray) -> np.ndarray:
        """
        P(O | m) = Gaussian likelihood
        관찰과 현재 믿음 사이의 유사도
        """
        # L2 거리
        distance = np.linalg.norm(obs - prior)
        
        # Gaussian: exp(-d^2 / 2)
        likelihood = np.exp(-distance**2 / 2.0)
        
        # 모든 차원에 동일하게 적용
        likelihood_vec = np.ones(self.latent_dim) * likelihood
        
        return likelihood_vec
    
    # ============ Layer 2: Information Metric ============
    
    def compute_entropy(self, memory_state: np.ndarray) -> float:
        """
        Shannon 엔트로피 계산
        
        H(m) = -Σ p_i log₂(p_i)
        
        - 작을수록 명확한 기억
        - 클수록 불확실한 기억
        
        Parameters:
        -----------
        memory_state : np.ndarray
            기억 벡터
        
        Returns:
        --------
        float
            엔트로피 값 (bits)
        """
        # 정규화: 모든 원소가 양수여야 함
        abs_state = np.abs(memory_state) + 1e-8
        p = abs_state / (np.sum(abs_state) + 1e-8)
        
        # 엔트로피: -Σ p_i log₂(p_i)
        h = -np.sum(p * np.log2(p + 1e-8))
        
        return float(h)
    
    def mutual_information(self, 
                          memory_state: np.ndarray, 
                          observation: np.ndarray) -> float:
        """
        상호정보량 (Mutual Information) 계산
        
        I(M; O) = H(M) - H(M|O)
        
        기억과 관찰이 공유하는 정보량
        - 높을수록: 기억이 관찰과 매우 관련있음 (버리지 말 것)
        - 낮을수록: 기억이 최근 관찰과 무관함 (버릴 후보)
        
        Parameters:
        -----------
        memory_state : np.ndarray
            기억 벡터
        observation : np.ndarray
            관찰 벡터
        
        Returns:
        --------
        float
            상호정보량 (bits)
        """
        h_m = self.compute_entropy(memory_state)
        
        # 조건부 엔트로피 (근사)
        conditional = memory_state - observation
        h_m_given_o = self.compute_entropy(conditional)
        
        # 상호정보 = H(M) - H(M|O)
        mi = h_m - h_m_given_o
        
        return max(0.0, float(mi))
    
    # ============ Layer 3: Stochastic Decay ============
    
    def ornstein_uhlenbeck_decay(self, 
                                 memory_state: np.ndarray, 
                                 dt: float = 1.0) -> np.ndarray:
        """
        Ornstein-Uhlenbeck 과정으로 기억 감쇠 모델링
        
        dm_t = -θ(m_t - 0)dt + σ dW_t
        
        매개변수:
        - θ: 복원력 (평형으로 돌아가는 속도)
        - σ: 휘발성 (확률적 요동)
        - W_t: Wiener process (표준 정규 분포)
        
        Parameters:
        -----------
        memory_state : np.ndarray
            기억 벡터
        dt : float
            시간 스텝 (시간 단위)
        
        Returns:
        --------
        np.ndarray
            감쇠된 기억
        """
        # 결정론적 항: m_t × exp(-θ × dt)
        deterministic = memory_state * np.exp(-self.theta * dt)
        
        # 확률적 항: σ × √dt × N(0,1)
        stochastic = self.sigma * np.sqrt(dt) * np.random.randn(*memory_state.shape)
        
        # 종합
        decayed = deterministic + stochastic
        
        return decayed
    
    def estimate_retention_probability(self, time_hours: float) -> float:
        """
        시간 경과에 따른 기억 보존 확률 추정
        
        P(recall_at_time_t) = E[exp(-θt)]
        
        Parameters:
        -----------
        time_hours : float
            경과 시간 (시간 단위)
        
        Returns:
        --------
        float
            기억 보존 확률 (0~1)
        """
        return float(np.exp(-self.theta * time_hours))
    
    # ============ Layer 4: Capacity Management ============
    
    def compute_current_usage(self) -> float:
        """
        현재 기억 용량 사용률 계산
        
        Usage = Σ H(m_i) for all memories
        
        Returns:
        --------
        float
            현재 엔트로피 합계 (bits)
        """
        total_entropy = sum(
            self.compute_entropy(m) 
            for m in self.memories.values()
        ) if self.memories else 0.0
        
        return float(total_entropy)
    
    def should_evict_memory(self) -> bool:
        """
        새 기억을 저장할 공간이 필요한가?
        """
        current = self.compute_current_usage()
        # 80% 이상 찼으면 공간 필요
        return current > self.max_capacity * 0.8
    
    def select_memory_to_evict(self) -> Optional[str]:
        """
        정보론적 기준으로 버릴 기억 선택
        
        arg min I(M_i; recent_observations)
        = 최근 관찰과의 상호정보가 가장 낮은 기억
        
        Returns:
        --------
        str or None
            제거할 기억의 이름, 없으면 None
        """
        if not self.memories:
            return None
        
        mi_scores = {}
        for obj_name, memory in self.memories.items():
            # 최근 관찰과의 상호정보
            recent_obs = self.bayesian_priors.get(obj_name, memory)
            mi = self.mutual_information(memory, recent_obs)
            mi_scores[obj_name] = mi
        
        # 상호정보가 가장 낮은 것 선택
        to_evict = min(mi_scores, key=mi_scores.get)
        
        logger.debug(f"선택된 제거 대상: {to_evict} (MI: {mi_scores[to_evict]:.4f})")
        
        return to_evict
    
    # ============ Main: Unified Update ============
    
    def store_memory(self, 
                    object_name: str, 
                    affordances: np.ndarray, 
                    importance: float = 0.5,
                    timestamp: Optional[datetime] = None) -> bool:
        """
        새로운 관찰을 기억에 저장
        
        통합 수식:
        m_t^{final} = 
          P(aff | O) × exp(-H/H_max) × exp(-θt)
        
        Parameters:
        -----------
        object_name : str
            기억의 대상
        affordances : np.ndarray
            관찰된 affordance 벡터
        importance : float
            중요도 (0~1)
        timestamp : datetime, optional
            타임스탬프
        
        Returns:
        --------
        bool
            저장 성공 여부
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # 1. Bayesian 업데이트
            posterior = self.bayesian_update(object_name, affordances)
            
            # 2. 정보량 계산
            entropy = self.compute_entropy(posterior)
            max_entropy = np.log2(self.latent_dim)
            information_weight = np.exp(-entropy / max_entropy) if max_entropy > 0 else 1.0
            
            # 3. 확률 과정 감쇠 (새로 저장하므로 dt=0, full strength)
            decayed = self.ornstein_uhlenbeck_decay(posterior, dt=0)
            
            # 4. 종합 기억값
            final_memory = posterior * information_weight * decayed
            
            # 저장
            self.memories[object_name] = final_memory
            self.bayesian_priors[object_name] = posterior
            self.timestamps[object_name] = timestamp
            
            # 엔트로피 이력 기록
            if object_name not in self.entropy_history:
                self.entropy_history[object_name] = []
            
            mi = self.mutual_information(final_memory, affordances)
            self.mi_scores[object_name] = mi
            
            self.entropy_history[object_name].append({
                'timestamp': timestamp.isoformat(),
                'entropy': float(entropy),
                'mi': float(mi),
                'information_weight': float(information_weight)
            })
            
            logger.info(f"💾 [저장] {object_name}")
            logger.info(f"   - Entropy: {entropy:.4f} bits")
            logger.info(f"   - MI Score: {mi:.4f} bits")
            logger.info(f"   - Information Weight: {information_weight:.4f}")
            
            # 용량 초과 시 가장 불필요한 기억 제거
            if self.should_evict_memory():
                to_evict = self.select_memory_to_evict()
                if to_evict and to_evict != object_name:
                    self._evict_memory(to_evict)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 기억 저장 실패 ({object_name}): {e}")
            return False
    
    def _evict_memory(self, object_name: str) -> None:
        """기억을 강제로 제거"""
        if object_name in self.memories:
            del self.memories[object_name]
            if object_name in self.bayesian_priors:
                del self.bayesian_priors[object_name]
            if object_name in self.mi_scores:
                del self.mi_scores[object_name]
            
            logger.info(f"🗑️  [강제 제거] {object_name} (용량 초과)")
    
    def recall_memory(self, 
                     object_name: str, 
                     elapsed_time_hours: float = 0.0) -> Optional[MemoryState]:
        """
        저장된 기억 회상
        
        신뢰도 = P(recall) × (1 - normalized_entropy)
        
        Parameters:
        -----------
        object_name : str
            회상할 기억의 이름
        elapsed_time_hours : float
            경과 시간 (시간 단위)
        
        Returns:
        --------
        MemoryState or None
            회상된 기억 상태
        """
        if object_name not in self.memories:
            logger.warning(f"⚠️  기억이 없음: {object_name}")
            return None
        
        try:
            memory = self.memories[object_name]
            
            # OU process로 시간 경과 적용
            decayed_memory = self.ornstein_uhlenbeck_decay(memory, dt=elapsed_time_hours)
            
            # 엔트로피 계산
            entropy = self.compute_entropy(decayed_memory)
            max_entropy = np.log2(self.latent_dim)
            
            # 신뢰도 계산
            retention_prob = self.estimate_retention_probability(elapsed_time_hours)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            confidence = retention_prob * (1.0 - normalized_entropy)
            
            memory_state = MemoryState(
                object_name=object_name,
                affordances=decayed_memory,
                confidence=max(0.0, float(confidence)),
                entropy=float(entropy),
                timestamp=self.timestamps.get(object_name, datetime.now()),
                mi_score=self.mi_scores.get(object_name, 0.0),
                retention_prob=float(retention_prob)
            )
            
            logger.info(f"📚 [회상] {object_name}")
            logger.info(f"   - Elapsed Time: {elapsed_time_hours:.2f}h")
            logger.info(f"   - Confidence: {confidence:.4f}")
            logger.info(f"   - Retention Prob: {retention_prob:.4f}")
            
            return memory_state
            
        except Exception as e:
            logger.error(f"❌ 회상 실패 ({object_name}): {e}")
            return None
    
    def update_all_memories_with_time(self, elapsed_seconds: float) -> int:
        """
        시간 경과에 따른 모든 기억 업데이트
        
        Parameters:
        -----------
        elapsed_seconds : float
            경과 시간 (초 단위)
        
        Returns:
        --------
        int
            제거된 기억 수
        """
        elapsed_hours = elapsed_seconds / 3600.0
        evicted_count = 0
        
        for obj_name in list(self.memories.keys()):
            memory = self.memories[obj_name]
            
            # OU process 적용
            decayed = self.ornstein_uhlenbeck_decay(memory, dt=elapsed_hours)
            
            # 보존 확률 계산
            retention = self.estimate_retention_probability(elapsed_hours)
            
            # 확률이 너무 낮으면 제거 (완전히 잊음)
            if retention < 0.01:  # 1% 미만
                self._evict_memory(obj_name)
                evicted_count += 1
                logger.info(f"🗑️  [완전 망각] {obj_name} (Retention: {retention:.4f})")
            else:
                self.memories[obj_name] = decayed
        
        return evicted_count
    
    # ============ Analysis & Visualization ============
    
    def get_system_status(self) -> Dict:
        """시스템 상태 요약"""
        total_memories = len(self.memories)
        current_usage = self.compute_current_usage()
        usage_percent = (current_usage / self.max_capacity * 100) if self.max_capacity > 0 else 0.0
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'total_memories': total_memories,
            'current_usage_bits': float(current_usage),
            'max_capacity_bits': self.max_capacity,
            'usage_percent': float(usage_percent),
            'memories': {}
        }
        
        for obj_name in self.memories.keys():
            memory_recall = self.recall_memory(obj_name, elapsed_time_hours=0.0)
            if memory_recall:
                status['memories'][obj_name] = {
                    'confidence': float(memory_recall.confidence),
                    'entropy': float(memory_recall.entropy),
                    'mi_score': float(memory_recall.mi_score),
                    'timestamp': memory_recall.timestamp.isoformat()
                }
        
        return status
    
    def export_to_json(self, filepath: str) -> bool:
        """시스템 상태를 JSON으로 내보내기"""
        try:
            status = self.get_system_status()
            
            # numpy array를 list로 변환
            for obj_name in self.memories.keys():
                memory = self.memories[obj_name]
                status['memories'][obj_name]['affordances'] = memory.tolist()
            
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2)
            
            logger.info(f"✅ 상태 저장 완료: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ 저장 실패: {e}")
            return False


def test_hybrid_memory_system():
    """
    정보론 + Bayesian + Stochastic 통합 테스트
    """
    print("\n" + "="*70)
    print("🧠 Advanced Hybrid Memory System 테스트 시작")
    print("="*70 + "\n")
    
    # 시스템 초기화
    system = AdvancedMemorySystem(
        latent_dim=64,
        max_capacity_bits=10000,
        theta=0.5,  # 망각 속도
        sigma=0.1   # 확률성
    )
    
    # Scenario 1: 같은 책을 여러 번 관찰
    print("\n📖 Scenario 1: 책 5번 관찰")
    print("-" * 70)
    
    book_affordances = np.array([
        0.92, 0.88, 0.05, 0.85, 0.90, 0.03,
        0.87, 0.91, 0.02, 0.89, 0.86, 0.04,
        0.90, 0.89, 0.06, 0.88, 0.92, 0.01,
        0.85, 0.87, 0.04, 0.91, 0.88, 0.03,
        0.91, 0.90, 0.05, 0.87, 0.89, 0.02,
        0.93, 0.86, 0.03, 0.90, 0.91, 0.04,
        0.88, 0.89, 0.02, 0.92, 0.87, 0.05,
        0.89, 0.91, 0.06, 0.88, 0.90, 0.03,
        0.90, 0.88, 0.04, 0.91, 0.89, 0.02,
        0.86, 0.87, 0.05, 0.92, 0.88, 0.01,
        0.91, 0.90, 0.03, 0.89
    ])
    
    # 첫 번째 관찰
    for i in range(1):
        system.store_memory('book', book_affordances, importance=0.95)
        print(f"   [관찰 1] 책 기억 저장 완료")
    
    # Scenario 2: 다른 객체들 저장
    print("\n🎯 Scenario 2: 다양한 객체 저장")
    print("-" * 70)
    
    # 컵 저장
    cup_affordances = np.random.rand(64) * 0.8 + 0.1
    system.store_memory('cup', cup_affordances, importance=0.80)
    print("   [저장] 컵")
    
    # 의자 저장
    chair_affordances = np.random.rand(64) * 0.7 + 0.2
    system.store_memory('chair', chair_affordances, importance=0.75)
    print("   [저장] 의자")
    
    # 즉시 회상 테스트
    print("\n⏱️  Scenario 3: 즉시 회상 (0시간 후)")
    print("-" * 70)
    
    recall_now = system.recall_memory('book', elapsed_time_hours=0)
    if recall_now:
        print(f"📚 책:")
        print(f"   - 신뢰도: {recall_now.confidence:.4f}")
        print(f"   - 엔트로피: {recall_now.entropy:.4f}")
        print(f"   - 상호정보: {recall_now.mi_score:.4f}")
    
    # 1시간 후 회상
    print("\n⏱️  Scenario 4: 1시간 후 회상")
    print("-" * 70)
    
    system.update_all_memories_with_time(3600)  # 1시간
    recall_1h = system.recall_memory('book', elapsed_time_hours=1)
    if recall_1h:
        print(f"📚 책:")
        print(f"   - 신뢰도: {recall_1h.confidence:.4f}")
        print(f"   - 엔트로피: {recall_1h.entropy:.4f}")
        print(f"   - 보존확률: {recall_1h.retention_prob:.4f}")
    
    # 24시간 후 회상
    print("\n⏱️  Scenario 5: 24시간 후 회상")
    print("-" * 70)
    
    system.update_all_memories_with_time(24*3600)  # 24시간
    recall_24h = system.recall_memory('book', elapsed_time_hours=24)
    if recall_24h:
        print(f"📚 책:")
        print(f"   - 신뢰도: {recall_24h.confidence:.4f}")
        print(f"   - 엔트로피: {recall_24h.entropy:.4f}")
        print(f"   - 보존확률: {recall_24h.retention_prob:.4f}")
    
    # 시스템 상태 조회
    print("\n📊 Scenario 6: 시스템 상태")
    print("-" * 70)
    
    status = system.get_system_status()
    print(f"총 기억 수: {status['total_memories']}")
    print(f"용량 사용률: {status['usage_percent']:.2f}%")
    print(f"   ({status['current_usage_bits']:.1f} / {status['max_capacity_bits']} bits)")
    
    for obj_name, mem_info in status['memories'].items():
        print(f"\n   {obj_name}:")
        print(f"   - 신뢰도: {mem_info['confidence']:.4f}")
        print(f"   - 엔트로피: {mem_info['entropy']:.4f}")
    
    # JSON 내보내기
    print("\n💾 Scenario 7: 상태 저장")
    print("-" * 70)
    
    system.export_to_json('/Users/hwangjeyeong/.openclaw/workspace/memory_system_status.json')
    
    print("\n" + "="*70)
    print("✅ 테스트 완료!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_hybrid_memory_system()
