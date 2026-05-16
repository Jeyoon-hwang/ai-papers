"""
Industrial Dataset Generation
시뮬레이션에서 affordance 학습 데이터 생성
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import tqdm


class IndustrialDataGenerator:
    """
    UR10e 시뮬레이션 환경에서 affordance 데이터셋 생성
    """
    
    # Affordance 정의
    AFFORDANCES = [
        'graspable',
        'stackable',
        'insertable',
        'placeable',
        'moveable',
        'fragile'
    ]
    
    # 객체 타입
    OBJECT_TYPES = ['box', 'cylinder', 'sphere', 'complex']
    
    # 색상 변형
    COLORS = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (0.5, 0.5, 0.5),  # Gray
    ]
    
    def __init__(self, output_dir: str = "./industrial_dataset"):
        """
        Initialize data generator
        
        Args:
            output_dir: 데이터셋 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장소
        self.dataset = {
            'images': [],
            'affordances': [],
            'metadata': [],
            'affordance_distributions': {}
        }
        
        # Affordance 분포 초기화
        for aff in self.AFFORDANCES:
            self.dataset['affordance_distributions'][aff] = {
                'positive': 0,
                'negative': 0
            }
    
    def generate_object_config(self, seed: int = None) -> Dict:
        """
        랜덤 물체 설정 생성
        
        Returns:
            config: {
                'type': str,
                'size': [dx, dy, dz],
                'position': [x, y, z],
                'color': (r, g, b),
                'mass': float
            }
        """
        if seed is not None:
            np.random.seed(seed)
        
        obj_type = np.random.choice(self.OBJECT_TYPES)
        color = self.COLORS[np.random.randint(len(self.COLORS))]
        
        if obj_type == 'box':
            size = np.random.uniform([0.05, 0.05, 0.05], [0.15, 0.15, 0.15])
            mass = np.prod(size) * np.random.uniform(50, 150)
        
        elif obj_type == 'cylinder':
            radius = np.random.uniform(0.02, 0.08)
            height = np.random.uniform(0.05, 0.2)
            size = np.array([radius * 2, radius * 2, height])
            mass = np.pi * radius**2 * height * np.random.uniform(50, 150)
        
        elif obj_type == 'sphere':
            radius = np.random.uniform(0.02, 0.1)
            size = np.array([radius * 2, radius * 2, radius * 2])
            mass = (4/3) * np.pi * radius**3 * np.random.uniform(50, 150)
        
        else:  # complex
            size = np.random.uniform([0.08, 0.08, 0.08], [0.2, 0.2, 0.2])
            mass = np.prod(size) * np.random.uniform(50, 200)
        
        position = np.array([
            np.random.uniform(0.2, 0.8),
            np.random.uniform(-0.3, 0.3),
            0.2  # 테이블 위
        ])
        
        return {
            'type': obj_type,
            'size': size,
            'position': position,
            'color': color,
            'mass': mass
        }
    
    def compute_affordances(self, config: Dict) -> Dict[str, float]:
        """
        물체 설정에서 affordance 계산
        
        물리 휴리스틱 기반 ground truth
        """
        obj_type = config['type']
        size = config['size']
        mass = config['mass']
        
        affordances = {}
        
        # Graspable: 적당한 크기, 무게
        min_size = np.min(size)
        max_size = np.max(size)
        size_ok = (0.02 < min_size < 0.15) and (max_size < 0.3)
        weight_ok = mass < 5.0
        
        # Sigmoid 스타일로 부드럽게
        graspable_score = (1.0 if (size_ok and weight_ok) else 0.3) + np.random.normal(0, 0.1)
        affordances['graspable'] = np.clip(graspable_score, 0, 1)
        
        # Stackable: 박스 형태, 안정적
        is_box = obj_type == 'box'
        stackable_base = 0.8 if is_box else 0.2
        affordances['stackable'] = np.clip(stackable_base + np.random.normal(0, 0.15), 0, 1)
        
        # Insertable: 작은 크기, 원통형/구형
        is_small = max_size < 0.1
        is_round = obj_type in ['cylinder', 'sphere']
        insertable_base = 0.7 if (is_small and is_round) else 0.2
        affordances['insertable'] = np.clip(insertable_base + np.random.normal(0, 0.15), 0, 1)
        
        # Placeable: 기본적으로 높음 (모든 물체 배치 가능)
        affordances['placeable'] = np.clip(0.85 + np.random.normal(0, 0.1), 0, 1)
        
        # Moveable: 무게 기반
        moveable_base = 0.9 if mass < 2.0 else (0.7 if mass < 4.0 else 0.3)
        affordances['moveable'] = np.clip(moveable_base + np.random.normal(0, 0.1), 0, 1)
        
        # Fragile: 가벼움 또는 크기가 크면 깨지기 쉬움
        is_light = mass < 0.5
        is_large = max_size > 0.15
        fragile_base = 0.7 if (is_light or is_large) else 0.2
        affordances['fragile'] = np.clip(fragile_base + np.random.normal(0, 0.15), 0, 1)
        
        return affordances
    
    def generate_synthetic_image(self, config: Dict, affordances: Dict) -> np.ndarray:
        """
        물체 설정에서 합성 RGB 이미지 생성
        
        실제로는 PyBullet 렌더링을 사용하지만,
        여기서는 간단한 합성 이미지 생성
        """
        height, width = 480, 640
        
        # 배경 (테이블)
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # 물체 원의 위치 (정규화)
        pos_normalized = config['position'][:2]
        center_x = int(pos_normalized[0] * width)
        center_y = int(pos_normalized[1] * width / 2 + height / 2)
        
        # 물체 크기 (픽셀)
        size_px = int((np.min(config['size']) + np.max(config['size'])) / 2 * 1000)
        
        # 색상
        color = tuple([int(c * 255) for c in config['color']])
        
        # 원 그리기 (간단한 물체 표현)
        import cv2
        cv2.circle(image, (center_x, center_y), size_px // 2, color, -1)
        
        # Noise 추가
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_dataset(self, num_samples: int = 10000, use_synthetic: bool = True):
        """
        데이터셋 생성
        
        Args:
            num_samples: 생성할 샘플 수
            use_synthetic: True면 합성 이미지, False면 PyBullet 렌더링
        """
        print(f"Generating {num_samples} samples...")
        
        pbar = tqdm.tqdm(total=num_samples)
        
        for i in range(num_samples):
            # 물체 생성
            config = self.generate_object_config(seed=i)
            affordances = self.compute_affordances(config)
            
            # 이미지 생성
            if use_synthetic:
                image = self.generate_synthetic_image(config, affordances)
            else:
                # TODO: PyBullet 렌더링
                image = self.generate_synthetic_image(config, affordances)
            
            # 데이터 저장
            self.dataset['images'].append(image)
            self.dataset['affordances'].append(affordances)
            self.dataset['metadata'].append(config)
            
            # 분포 업데이트
            for aff_name, aff_value in affordances.items():
                if aff_value > 0.5:
                    self.dataset['affordance_distributions'][aff_name]['positive'] += 1
                else:
                    self.dataset['affordance_distributions'][aff_name]['negative'] += 1
            
            pbar.update(1)
        
        pbar.close()
    
    def save_dataset(self, format: str = 'npz'):
        """
        데이터셋 저장
        
        Args:
            format: 'npz' (numpy compressed), 'pickle', or 'json' (metadata only)
        """
        print(f"Saving dataset to {self.output_dir}...")
        
        if format == 'npz':
            # 이미지를 numpy 배열로 변환
            images_array = np.array(self.dataset['images'], dtype=np.uint8)
            affordances_array = np.array(
                [[aff[k] for k in self.AFFORDANCES] for aff in self.dataset['affordances']],
                dtype=np.float32
            )
            
            np.savez_compressed(
                self.output_dir / 'industrial_affordance_data.npz',
                images=images_array,
                affordances=affordances_array
            )
            print(f"  Images shape: {images_array.shape}")
            print(f"  Affordances shape: {affordances_array.shape}")
        
        elif format == 'pickle':
            with open(self.output_dir / 'industrial_affordance_data.pkl', 'wb') as f:
                pickle.dump(self.dataset, f)
        
        # 메타데이터와 분포 저장
        metadata = {
            'affordance_names': self.AFFORDANCES,
            'object_types': self.OBJECT_TYPES,
            'num_samples': len(self.dataset['images']),
            'affordance_distributions': self.dataset['affordance_distributions']
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print("\nAffordance distributions:")
        for aff_name, dist in self.dataset['affordance_distributions'].items():
            total = dist['positive'] + dist['negative']
            pos_rate = dist['positive'] / total * 100
            print(f"  {aff_name:15s}: {pos_rate:5.1f}% positive ({dist['positive']}/{total})")
    
    def load_dataset(self, filepath: str):
        """데이터셋 로드"""
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            return data['images'], data['affordances']
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)


# 테스트 및 실행
if __name__ == "__main__":
    # 데이터셋 생성
    generator = IndustrialDataGenerator(
        output_dir="./industrial_robot_affordance/dataset"
    )
    
    # 1000개 샘플로 시작 (빠른 테스트)
    generator.generate_dataset(num_samples=1000, use_synthetic=True)
    
    # 데이터셋 저장
    generator.save_dataset(format='npz')
    
    print("\n✓ Dataset generation complete!")
    print(f"  Output directory: {generator.output_dir}")
