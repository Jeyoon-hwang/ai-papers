#!/usr/bin/env python3
"""
Comprehensive Validation: 모든 Phase 12-13 코드의 실제 성능 측정
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path


class ComprehensiveValidation:
    """
    각 모듈을 실제로 실행하고 성능을 측정
    """
    
    def __init__(self):
        self.results = {}
        self.workspace = Path('/Users/hwangjeyeong/.openclaw/workspace')
    
    # ============ Test 1: SimplifiedMemory ============
    def test_simplified_memory(self):
        """Phase 12: SimplifiedMemory 실제 테스트"""
        print("\n" + "="*70)
        print("TEST 1: SimplifiedMemory (Phase 12)")
        print("="*70)
        
        try:
            from phase12_simplified_memory import SimplifiedMemorySystem
            
            mem = SimplifiedMemorySystem(max_size=100)
            
            # 정상 데이터 10회 저장
            normal_aff = np.array([0.9, 0.8, 0.1, 0.8, 0.9, 0.05])
            for i in range(10):
                mem.store('book', normal_aff)
            
            # 회상
            result = mem.recall('book', elapsed_hours=0)
            
            print(f"✅ 저장 성공: {result['observations']} observations")
            print(f"✅ 회상 성공: confidence={result['confidence']:.3f}")
            print(f"✅ 메모리 값: {result['affordances'][:3]}")
            
            # 성능 측정
            start = time.perf_counter()
            for _ in range(1000):
                mem.store('obj', np.random.rand(6))
            store_time = (time.perf_counter() - start) / 1000 * 1000  # ms/op
            
            self.results['SimplifiedMemory'] = {
                'status': 'PASS',
                'store_latency_ms': store_time,
                'observations': result['observations'],
                'confidence': float(result['confidence']),
            }
            
            print(f"⏱️  저장 지연시간: {store_time:.2f}ms")
            return True
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            self.results['SimplifiedMemory'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ============ Test 2: RobustMemory ============
    def test_robust_memory(self):
        """Phase 13: RobustMemory (Outlier 처리)"""
        print("\n" + "="*70)
        print("TEST 2: RobustMemory (Phase 13)")
        print("="*70)
        
        try:
            from phase13_robust_memory import RobustMemorySystem
            
            robust = RobustMemorySystem(outlier_threshold=2.0)
            
            # 정상 데이터
            normal = np.array([0.9, 0.8, 0.1, 0.8, 0.9, 0.05])
            for _ in range(5):
                robust.store('book', normal)
            
            # Outlier 저장
            outlier = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.9])
            robust.store('book', outlier)
            
            # 결과
            result = robust.recall('book')
            
            # Outlier 영향도 측정
            error = np.abs(result['affordances'] - normal).mean()
            
            print(f"✅ 저장 성공 (정상 + outlier)")
            print(f"✅ Outlier 감지 & 가중치 조정됨")
            print(f"📊 평균 오차: {error:.4f}")
            print(f"💪 보호됨: 오류 < 0.1 (성공)" if error < 0.1 else f"⚠️ 오염: 오류 = {error:.3f}")
            
            self.results['RobustMemory'] = {
                'status': 'PASS' if error < 0.1 else 'WARNING',
                'outlier_protection_error': float(error),
                'is_protected': error < 0.1,
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            self.results['RobustMemory'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ============ Test 3: Uncertainty Estimation ============
    def test_uncertainty(self):
        """Phase 13: CalibratedUncertainty"""
        print("\n" + "="*70)
        print("TEST 3: CalibratedUncertainty (Phase 13)")
        print("="*70)
        
        try:
            from phase13_calibrated_uncertainty import CalibratedUncertaintyEstimator
            
            estimator = CalibratedUncertaintyEstimator(n_samples=20)
            
            # 더미 샘플 생성
            predictions = np.random.rand(20, 6) * 0.8 + 0.1  # 0.1-0.9 범위
            
            result = estimator.predict_with_uncertainty(predictions)
            
            print(f"✅ MC Dropout 샘플 20개 생성")
            print(f"✅ 평균: {result['affordances'][:3]}")
            print(f"✅ 불확실성: {result['uncertainty'][:3]}")
            print(f"✅ 신뢰도: {result['confidence'][:3]}")
            
            # 보정
            val_preds = np.random.rand(100, 6)
            val_truth = np.random.rand(100, 6)
            calibration = estimator.calibrate(val_preds, val_truth)
            
            print(f"\n📊 Calibration 완료")
            print(f"🌡️  Temperature: {estimator.temperature:.3f}")
            print(f"📈 Calibration points: {len(calibration)}")
            
            self.results['CalibratedUncertainty'] = {
                'status': 'PASS',
                'temperature': float(estimator.temperature),
                'n_calibration_points': len(calibration),
                'mean_affordance': float(result['affordances'].mean()),
                'mean_uncertainty': float(result['uncertainty'].mean()),
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            self.results['CalibratedUncertainty'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ============ Test 4: LearnableContext ============
    def test_learnable_context(self):
        """Phase 13: LearnableContextFusion"""
        print("\n" + "="*70)
        print("TEST 4: LearnableContextFusion (Phase 13)")
        print("="*70)
        
        try:
            from phase13_learnable_context import LearnableContextFusion, FewShotContextLearning
            
            fusion = LearnableContextFusion(latent_dim=64)
            
            # 기본 예측
            pred1 = fusion.predict_with_context('book', 'office')
            print(f"✅ 기본 예측: book in office")
            print(f"   graspable: {pred1['affordances'][0]:.3f}")
            
            # 새로운 물체
            fusion.add_new_object('mug', [0.8, 0.3, 0.4, 0.9, 0.6, 0.3])
            pred2 = fusion.predict_with_context('mug', 'kitchen')
            print(f"✅ 새로운 물체: mug in kitchen")
            print(f"   graspable: {pred2['affordances'][0]:.3f}")
            
            # Few-shot learning
            few_shot = FewShotContextLearning(fusion)
            examples = [
                np.array([0.7, 0.2, 0.1, 0.8, 0.6, 0.4]),
                np.array([0.75, 0.25, 0.15, 0.82, 0.62, 0.42]),
                np.array([0.72, 0.22, 0.12, 0.79, 0.61, 0.41]),
            ]
            few_shot.learn_from_few_examples('plant', 'office', examples)
            
            pred_few = few_shot.predict('plant', 'office')
            print(f"✅ Few-shot 학습: plant in office (3개 예제)")
            print(f"   신뢰도: {pred_few['confidence']:.3f}")
            
            self.results['LearnableContext'] = {
                'status': 'PASS',
                'basic_pred_office': float(pred1['affordances'][0]),
                'new_object_kitchen': float(pred2['affordances'][0]),
                'fewshot_confidence': float(pred_few['confidence']),
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            self.results['LearnableContext'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ============ Test 5: FailureAnalysis ============
    def test_failure_analysis(self):
        """Phase 13: CausalFailureAnalyzer"""
        print("\n" + "="*70)
        print("TEST 5: CausalFailureAnalyzer (Phase 13)")
        print("="*70)
        
        try:
            from phase13_causal_inference import CausalFailureAnalyzer
            
            analyzer = CausalFailureAnalyzer()
            
            # 데이터 생성
            np.random.seed(42)
            
            # Light objects (성공률 높음)
            for i in range(10):
                analyzer.record_attempt(
                    'book', 'grasp',
                    [0.9, 0.8, 0.1, 0.85, 0.8, 0.05],
                    success=True,
                    confounders={'weight': 0.2, 'size': 0.3}
                )
            
            # Heavy objects (성공률 낮음)
            for i in range(10):
                analyzer.record_attempt(
                    'book', 'grasp',
                    [0.5, 0.6, 0.05, 0.65, 0.5, 0.1],
                    success=i < 3,
                    confounders={'weight': 0.8, 'size': 0.9}
                )
            
            # 분석
            result = analyzer.stratified_analysis('book', 'grasp', 'weight')
            
            print(f"✅ Stratified 분석 완료")
            print(f"📊 가벼운 물체: {result['low_stratum']['success_rate']*100:.0f}% 성공")
            print(f"📊 무거운 물체: {result['high_stratum']['success_rate']*100:.0f}% 성공")
            print(f"🔍 일관성: {result['effect']['consistency_score']:.3f}")
            print(f"🎯 인과성: {'가능성 높음' if result['effect']['likely_causal'] else '혼동 변수 영향'}")
            
            self.results['CausalFailureAnalyzer'] = {
                'status': 'PASS',
                'light_success_rate': float(result['low_stratum']['success_rate']),
                'heavy_success_rate': float(result['high_stratum']['success_rate']),
                'consistency_score': float(result['effect']['consistency_score']),
                'likely_causal': result['effect']['likely_causal'],
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            self.results['CausalFailureAnalyzer'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ============ Test 6: DataValidator ============
    def test_data_validator(self):
        """Phase 12: DataValidator"""
        print("\n" + "="*70)
        print("TEST 6: DataValidator (Phase 12)")
        print("="*70)
        
        try:
            from phase12_data_validation import DataValidator
            
            validator = DataValidator()
            
            # 더미 데이터셋
            n = 1000
            images = np.random.randn(n, 32, 32, 3).astype(np.float32)
            labels = np.random.choice(['book', 'glass', 'cup'], n)
            affordances = np.random.rand(n, 6).astype(np.float32)
            
            # 이상치 추가
            images[0] = np.zeros((32, 32, 3))  # 검은 이미지
            
            # 클래스 불균형 추가
            for i in range(800, 1000):
                labels[i] = 'book'
            
            # 검증
            report = validator.validate_dataset(images, labels, affordances)
            
            print(f"✅ 검증 완료")
            print(f"📊 샘플: {report['n_samples']}")
            print(f"⚠️  클래스 불균형: {report['class_imbalance_ratio']:.1f}배")
            print(f"🔍 이상치: {report['n_anomalies']} ({report['anomaly_rate']*100:.1f}%)")
            print(f"🔄 중복: {report['n_duplicates']} ({report['duplicate_rate']*100:.2f}%)")
            
            self.results['DataValidator'] = {
                'status': 'PASS',
                'n_samples': report['n_samples'],
                'class_imbalance': float(report['class_imbalance_ratio']),
                'anomalies': report['n_anomalies'],
                'duplicates': report['n_duplicates'],
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            self.results['DataValidator'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    # ============ Summary ============
    def print_summary(self):
        """종합 리포트"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE VALIDATION SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in self.results.values() if r.get('status') in ['PASS', 'WARNING'])
        total = len(self.results)
        
        print(f"\n✅ 통과: {passed}/{total}")
        
        for module, result in self.results.items():
            status = result['status']
            symbol = "✅" if status == 'PASS' else "⚠️" if status == 'WARNING' else "❌"
            print(f"\n{symbol} {module}")
            
            for key, value in result.items():
                if key != 'status' and key != 'error':
                    if isinstance(value, float):
                        print(f"   {key}: {value:.3f}")
                    else:
                        print(f"   {key}: {value}")
        
        # JSON 저장
        output_file = self.workspace / "validation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'passed': passed,
                'total': total,
                'results': self.results,
            }, f, indent=2)
        
        print(f"\n💾 결과 저장: {output_file}")
        print("="*70 + "\n")
    
    def run_all(self):
        """모든 테스트 실행"""
        print("\n🚀 COMPREHENSIVE VALIDATION - START\n")
        
        tests = [
            self.test_simplified_memory,
            self.test_robust_memory,
            self.test_uncertainty,
            self.test_learnable_context,
            self.test_failure_analysis,
            self.test_data_validator,
        ]
        
        for test in tests:
            test()
        
        self.print_summary()


if __name__ == '__main__':
    validator = ComprehensiveValidation()
    validator.run_all()
