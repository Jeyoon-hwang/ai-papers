#!/usr/bin/env python3
"""
Industrial Robot Affordance System - Main Demo
산업용 로봇 affordance 시스템 통합 데모
"""

import numpy as np
import torch
from pathlib import Path
import json

# Import modules
from sim_env import IndustrialRobotEnv
from generate_industrial_data import IndustrialDataGenerator
from affordance_model import IndustrialAffordanceModel, AffordanceTrainer
from affordance_policy import AffordancePolicy, RobotActionExecutor
from ros_controller import RobotController, DemoPickAndPlace
from safety_check import SafetyValidator
from evaluation import AffordanceEvaluator


def demo_1_environment_setup():
    \"\"\"
    Demo 1: UR10e + Robotiq 환경 구축
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 1: Industrial Robot Environment Setup\")
    print(\"=\" * 70)
    
    # 환경 생성
    env = IndustrialRobotEnv(use_gui=False)
    
    # 산업 물건 생성
    print(\"\\nSpawning industrial objects...\")
    
    # 박스
    env.spawn_industrial_object(
        'box',
        [0.3, 0, 0.2],
        {'graspable': 0.9, 'stackable': 0.8, 'placeable': 0.85, 'moveable': 0.85, 'insertable': 0.2, 'fragile': 0.1}
    )
    print(\"  ✓ Box spawned\")
    
    # 원통
    env.spawn_industrial_object(
        'cylinder',
        [0.6, 0, 0.2],
        {'graspable': 0.7, 'stackable': 0.3, 'placeable': 0.8, 'moveable': 0.7, 'insertable': 0.8, 'fragile': 0.3}
    )
    print(\"  ✓ Cylinder spawned\")
    
    # 구
    env.spawn_industrial_object(
        'sphere',
        [0.9, 0, 0.2],
        {'graspable': 0.6, 'stackable': 0.1, 'placeable': 0.9, 'moveable': 0.8, 'insertable': 0.2, 'fragile': 0.5}
    )
    print(\"  ✓ Sphere spawned\")
    
    # 이미지 캡처
    print(\"\\nCapturing RGB-D image...\")
    rgb, depth = env.capture_image()
    print(f\"  RGB shape: {rgb.shape}\")
    print(f\"  Depth shape: {depth.shape}\")
    
    # Affordance 감지
    print(\"\\nDetecting affordances from physics...\")
    for obj_name, obj_data in env.objects.items():
        affordances = env.detect_affordances_from_physics(obj_name)
        print(f\"\\n  {obj_name}:\")
        for aff_name, aff_value in affordances.items():
            bar_length = int(aff_value * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f\"    {aff_name:12s}: [{bar}] {aff_value:.3f}\")
    
    env.disconnect()
    print(\"\\n✓ Demo 1 complete!\")


def demo_2_dataset_generation():
    \"\"\"
    Demo 2: 시뮬레이션 데이터셋 생성
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 2: Industrial Dataset Generation\")
    print(\"=\" * 70)
    
    # 데이터 생성기 생성
    generator = IndustrialDataGenerator(
        output_dir=\"./industrial_robot_affordance/dataset\"
    )
    
    print(\"\\nGenerating dataset (1000 samples)...\")
    print(\"  This may take a minute...\")
    
    # 데이터셋 생성
    generator.generate_dataset(num_samples=1000, use_synthetic=True)
    
    # 데이터셋 저장
    print(\"\\nSaving dataset...\")
    generator.save_dataset(format='npz')
    
    print(\"\\n✓ Demo 2 complete!\")


def demo_3_model_training():
    \"\"\"
    Demo 3: Affordance 모델 학습
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 3: Affordance Model Training\")
    print(\"=\" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f\"\\nUsing device: {device}\")
    
    # 모델 생성
    model = IndustrialAffordanceModel(latent_dim=64)
    model.to(device)
    
    print(\"\\nModel architecture:\")
    print(f\"  VAE: Conv encoder → latent (64d) → ConvT decoder\")
    print(f\"  Affordance head: latent (64d) → 6 affordances\")
    
    # Trainer 생성
    trainer = AffordanceTrainer(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 더미 데이터로 학습 스텝 시뮬레이션
    print(\"\\nTraining (5 iterations with dummy data)...\")
    
    for epoch in range(5):
        # 더미 배치
        batch_images = torch.randn(4, 3, 128, 128).to(device)
        batch_affordances = torch.rand(4, 6).to(device)
        
        # 학습
        losses = trainer.train_step(batch_images, batch_affordances, optimizer)
        
        # 검증
        val_metrics = trainer.validate(batch_images, batch_affordances)
        
        print(f\"  Epoch {epoch+1}: loss={losses['total_loss']:.4f}, acc={val_metrics['aff_accuracy']:.3f}\")
    
    print(\"\\n✓ Demo 3 complete!\")


def demo_4_affordance_policy():
    \"\"\"
    Demo 4: Affordance → Robot Action 매핑
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 4: Affordance to Robot Action Mapping\")
    print(\"=\" * 70)
    
    policy = AffordancePolicy()
    executor = RobotActionExecutor()
    
    # 테스트 케이스들
    test_cases = [
        {
            'name': 'Standard Box',
            'affordances': {
                'graspable': 0.9, 'stackable': 0.8,
                'insertable': 0.2, 'placeable': 0.85,
                'moveable': 0.85, 'fragile': 0.1
            },
            'object_info': {'size': [0.1, 0.1, 0.1], 'weight_estimate': 2.0}
        },
        {
            'name': 'Fragile Glass',
            'affordances': {
                'graspable': 0.85, 'stackable': 0.3,
                'insertable': 0.2, 'placeable': 0.8,
                'moveable': 0.7, 'fragile': 0.9
            },
            'object_info': {'size': [0.08, 0.08, 0.15], 'weight_estimate': 0.5}
        },
        {
            'name': 'Heavy Equipment',
            'affordances': {
                'graspable': 0.3, 'stackable': 0.1,
                'insertable': 0.1, 'placeable': 0.5,
                'moveable': 0.2, 'fragile': 0.2
            },
            'object_info': {'size': [0.3, 0.3, 0.3], 'weight_estimate': 50.0}
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f\"\\n[Test {i+1}] {test['name']}\")
        print(\"-\" * 70)
        
        action = policy.affordance_to_robot_action(
            test['affordances'],
            test['object_info']
        )
        
        executor.execute_action(action)
    
    print(\"\\n✓ Demo 4 complete!\")


def demo_5_safety_validation():
    \"\"\"
    Demo 5: 안전성 검증
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 5: Safety Validation\")
    print(\"=\" * 70)
    
    validator = SafetyValidator()
    
    # 테스트 케이스들
    test_cases = [
        {
            'name': 'Safe Grasp',
            'affordances': {
                'graspable': 0.9, 'stackable': 0.8,
                'insertable': 0.2, 'placeable': 0.85,
                'moveable': 0.85, 'fragile': 0.1
            },
            'gripper_config': {
                'gripper_force': 80,
                'gripper_width': 0.07,
                'approach_speed': 0.3
            }
        },
        {
            'name': 'Unsafe Force (Fragile)',
            'affordances': {
                'graspable': 0.85, 'stackable': 0.3,
                'insertable': 0.2, 'placeable': 0.8,
                'moveable': 0.7, 'fragile': 0.9
            },
            'gripper_config': {
                'gripper_force': 120,  # Too high!
                'gripper_width': 0.05,
                'approach_speed': 0.5
            }
        },
        {
            'name': 'Low Force (Slip Risk)',
            'affordances': {
                'graspable': 0.9, 'stackable': 0.8,
                'insertable': 0.2, 'placeable': 0.85,
                'moveable': 0.85, 'fragile': 0.1
            },
            'gripper_config': {
                'gripper_force': 15,  # Too low!
                'gripper_width': 0.07,
                'approach_speed': 0.3
            }
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f\"\\n[Test {i+1}] {test['name']}\")
        print(\"-\" * 70)
        
        is_safe, issues = validator.validate_grasp(
            obj_id=i,
            affordances=test['affordances'],
            gripper_config=test['gripper_config'],
            object_position=np.array([0.5, 0, 0.2])
        )
        
        if is_safe:
            print(\"✓ Safety check PASSED\")
        else:
            print(\"✗ Safety check FAILED\")
            for issue in issues:
                print(f\"  - {issue}\")
    
    print(\"\\n✓ Demo 5 complete!\")


def demo_6_pick_and_place():
    \"\"\"
    Demo 6: Pick and place 작업
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 6: Pick & Place with Affordance-based Control\")
    print(\"=\" * 70)
    
    demo = DemoPickAndPlace(affordance_model=None)
    demo.run_demo()
    
    print(\"\\n✓ Demo 6 complete!\")


def demo_7_evaluation():
    \"\"\"
    Demo 7: 시스템 평가
    \"\"\"
    print(\"\\n\" + \"=\" * 70)
    print(\"DEMO 7: System Evaluation\")
    print(\"=\" * 70)
    
    evaluator = AffordanceEvaluator()
    
    # 더미 데이터 생성
    num_samples = 100
    
    predictions = [
        {
            'graspable': np.random.beta(7, 2),
            'stackable': np.random.beta(4, 4),
            'insertable': np.random.beta(3, 5),
            'placeable': np.random.beta(7, 2),
            'moveable': np.random.beta(5, 3),
            'fragile': np.random.beta(3, 5)
        }
        for _ in range(num_samples)
    ]
    
    ground_truth = [
        {
            'graspable': np.random.beta(7, 2),
            'stackable': np.random.beta(4, 4),
            'insertable': np.random.beta(3, 5),
            'placeable': np.random.beta(7, 2),
            'moveable': np.random.beta(5, 3),
            'fragile': np.random.beta(3, 5),
            'success': np.random.rand() > 0.2
        }
        for _ in range(num_samples)
    ]
    
    object_metadata = [
        {
            'color': (np.random.rand(), np.random.rand(), np.random.rand()),
            'size': np.random.uniform([0.05, 0.05, 0.05], [0.15, 0.15, 0.15])
        }
        for _ in range(num_samples)
    ]
    
    # 평가 실행
    evaluator.evaluate_form_independence(predictions, object_metadata)
    evaluator.evaluate_grasp_success_rate(predictions, ground_truth)
    evaluator.evaluate_affordance_accuracy(predictions, ground_truth)
    evaluator.evaluate_simulation_transfer(0.92, 0.82)
    
    # 리포트 생성
    report = evaluator.generate_evaluation_report()
    
    print(\"\\n✓ Demo 7 complete!\")


def main():
    \"\"\"
    메인 데모
    \"\"\"
    print(\"\\n\" + \"#\" * 70)
    print(\"#\" + \" \" * 68 + \"#\")
    print(\"#\" + \"  Industrial Robot Affordance System - Comprehensive Demo\".center(68) + \"#\")
    print(\"#\" + \" \" * 68 + \"#\")
    print(\"#\" * 70)
    
    demos = [
        (\"Environment Setup\", demo_1_environment_setup),
        (\"Dataset Generation\", demo_2_dataset_generation),
        (\"Model Training\", demo_3_model_training),
        (\"Affordance Policy\", demo_4_affordance_policy),
        (\"Safety Validation\", demo_5_safety_validation),
        (\"Pick & Place\", demo_6_pick_and_place),
        (\"System Evaluation\", demo_7_evaluation)
    ]
    
    print(\"\\nAvailable demos:\")
    for i, (name, _) in enumerate(demos):
        print(f\"  {i+1}. {name}\")
    print(f\"  0. Run all demos\")
    
    choice = input(\"\\nSelect demo (0-7): \").strip()
    
    if choice == '0':
        # 모든 데모 실행
        for name, demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f\"\\n✗ Error in {name}: {e}\")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demos[idx][1]()
            else:
                print(\"Invalid choice!\")
        except Exception as e:
            print(f\"Error: {e}\")
    
    print(\"\\n\" + \"=\" * 70)
    print(\"All demos completed!\")
    print(\"=\" * 70 + \"\\n\")


if __name__ == \"__main__\":
    main()
