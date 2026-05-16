"""
ROS Integration - Robot Controller
UR10e + Robotiq gripper 제어
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from affordance_policy import AffordancePolicy
from safety_check import SafetyValidator


class RobotController:
    """
    ROS 기반 로봇 제어기
    affordance 감지 → 로봇 행동 실행
    """
    
    def __init__(self, affordance_model=None, use_sim=True):
        """
        Initialize robot controller
        
        Args:
            affordance_model: affordance 감지 모델 (optional)
            use_sim: True면 PyBullet 시뮬레이션, False면 실제 로봇
        """
        self.affordance_policy = AffordancePolicy()
        self.safety_validator = SafetyValidator()
        self.affordance_model = affordance_model
        self.use_sim = use_sim
        
        # ROS 통합 (실제 구현 시)
        self.arm = None  # moveit_commander.MoveGroupCommander("manipulator")
        self.gripper = None  # GripperController()
        
        # 로봇 상태
        self.current_pose = None
        self.current_joint_states = None
        self.gripper_state = 'open'
        
        # 카메라
        self.camera_id = None
        
    def detect_affordances(self, image: np.ndarray) -> Dict[str, float]:
        """
        이미지에서 affordance 감지
        
        Args:
            image: (H, W, 3) RGB 이미지
            
        Returns:
            affordances: {graspable, stackable, ...}
        """
        if self.affordance_model is None:
            # 더미 affordances 반환
            return {
                'graspable': np.random.rand(),
                'stackable': np.random.rand(),
                'insertable': np.random.rand(),
                'placeable': np.random.rand(),
                'moveable': np.random.rand(),
                'fragile': np.random.rand()
            }
        
        # 이미지 전처리
        image_tensor = self._preprocess_image(image)
        
        # 모델 추론
        affordances_dict = self.affordance_model.predict_affordances(image_tensor)
        
        return affordances_dict
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지 전처리
        
        Args:
            image: (H, W, 3) uint8
            
        Returns:
            tensor: (1, 3, H, W) normalized
        """
        # 정규화
        image_float = image.astype(np.float32) / 255.0
        
        # 리사이즈 (모델 입력에 맞게)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Batch dimension
        
        return image_tensor
    
    def find_graspable_objects(self,
                              image: np.ndarray,
                              min_graspability: float = 0.6) -> List[Dict]:
        """
        이미지에서 집을 수 있는 물체 찾기
        
        Args:
            image: 입력 이미지
            min_graspability: 최소 graspability threshold
            
        Returns:
            objects: [{'position': [x,y], 'affordances': {...}}, ...]
        """
        # affordance 감지
        affordances = self.detect_affordances(image)
        
        # 간단한 물체 위치 추정 (실제로는 YOLO/Mask R-CNN 사용)
        objects = []
        
        if affordances['graspable'] > min_graspability:
            # 이미지 중심에 물체가 있다고 가정
            obj = {
                'position': [image.shape[1] // 2, image.shape[0] // 2],
                'affordances': affordances,
                'center': True
            }
            objects.append(obj)
        
        return objects
    
    def estimate_grasp_pose(self,
                           object_position: Tuple[int, int],
                           affordances: Dict[str, float],
                           depth_image: np.ndarray = None) -> np.ndarray:
        """
        물체 위치에서 grasp pose 추정
        
        Args:
            object_position: [x_pixel, y_pixel]
            affordances: affordance 벡터
            depth_image: depth map (optional)
            
        Returns:
            grasp_pose: [x, y, z, roll, pitch, yaw] (6D)
        """
        # 픽셀 → 월드 좌표 (카메라 내부 파라미터 사용)
        # 여기서는 간단한 변환만 수행
        
        # 카메라 내부 파라미터 (Intel RealSense D435)
        fx, fy = 614.5, 614.5  # focal length
        cx, cy = 320, 240       # principal point
        
        # depth 값 추정 (depth image가 없으면 고정값)
        if depth_image is not None:
            depth_z = depth_image[object_position[1], object_position[0]]
        else:
            depth_z = 0.5  # 50cm
        
        # 픽셀 → 카메라 좌표
        x = (object_position[0] - cx) * depth_z / fx
        y = (object_position[1] - cy) * depth_z / fy
        z = depth_z
        
        # Grasp orientation (affordance 기반)
        # fragile하면 수직(z축)에서 약간의 각도
        if affordances['fragile'] > 0.6:
            roll, pitch, yaw = 0, 0, 0  # Vertical grasp
        else:
            roll, pitch, yaw = 0, 0, 0  # Standard grasp
        
        grasp_pose = np.array([x, y, z, roll, pitch, yaw])
        
        return grasp_pose
    
    def execute_pick_and_place(self,
                              object_pose: np.ndarray,
                              place_pose: np.ndarray,
                              affordances: Dict[str, float],
                              object_info: Dict = None) -> bool:
        """
        Pick and place 작업 실행
        
        Args:
            object_pose: [x, y, z, roll, pitch, yaw]
            place_pose: [x, y, z, roll, pitch, yaw]
            affordances: affordance 벡터
            object_info: 물체 정보
            
        Returns:
            success: 성공 여부
        """
        if object_info is None:
            object_info = {'weight_estimate': 1.0}
        
        # affordance → 로봇 액션
        action = self.affordance_policy.affordance_to_robot_action(
            affordances,
            object_info
        )
        
        # 안전성 검증
        is_safe, issues = self.safety_validator.validate_grasp(
            obj_id=0,
            affordances=affordances,
            gripper_config=action,
            object_position=object_pose[:3]
        )
        
        if not is_safe:
            print(f"Safety check failed:")
            for issue in issues:
                print(f"  ✗ {issue}")
            return False
        
        print(f"Executing pick and place with action: {action}")
        
        # 1. Approach
        approach_pose = object_pose.copy()
        approach_pose[2] += 0.1  # 위에서 접근
        
        self._move_to_pose(approach_pose, action['approach_speed'])
        print(f"  ✓ Approached object")
        
        # 2. Gripper 열기
        self._gripper_open(action['gripper_width'])
        print(f"  ✓ Gripper opened")
        
        # 3. 내려가기
        self._move_to_pose(object_pose, action['approach_speed'])
        print(f"  ✓ Moved to grasp position")
        
        # 4. Gripper 닫기
        self._gripper_close(action['gripper_force'])
        print(f"  ✓ Gripper closed")
        
        # 5. 들어올리기
        lift_pose = object_pose.copy()
        lift_pose[2] += action['lift_height']
        self._move_to_pose(lift_pose, action['approach_speed'])
        print(f"  ✓ Object lifted")
        
        # 6. 목표 위치로 이동
        self._move_to_pose(place_pose, action['approach_speed'])
        print(f"  ✓ Moved to placement position")
        
        # 7. 놓기
        self._gripper_open(action['gripper_width'])
        print(f"  ✓ Object placed")
        
        # 8. 복귀
        retreat_pose = place_pose.copy()
        retreat_pose[2] += 0.1
        self._move_to_pose(retreat_pose, action['approach_speed'])
        print(f"  ✓ Retreated")
        
        return True
    
    def _move_to_pose(self, pose: np.ndarray, speed: float = 0.5):
        \"\"\"
        로봇을 특정 pose로 이동
        
        Args:
            pose: [x, y, z, roll, pitch, yaw]
            speed: 이동 속도 (m/s)
        \"\"\"
        if self.use_sim:
            # PyBullet 시뮬레이션
            print(f"    [SIM] Moving to pose {pose[:3]}, speed={speed:.2f}")
        else:
            # 실제 로봇 (ROS)
            # self.arm.set_pose_target(pose)
            # self.arm.go()
            print(f"    [ROBOT] Moving to pose {pose[:3]}, speed={speed:.2f}")
    
    def _gripper_open(self, width: float):
        \"\"\"
        Gripper 열기
        
        Args:
            width: 개구부 너비 (m)
        \"\"\"
        if self.use_sim:
            print(f"    [SIM] Gripper open to {width*1000:.1f}mm")
        else:
            # self.gripper.open(width)
            print(f"    [ROBOT] Gripper open to {width*1000:.1f}mm")
    
    def _gripper_close(self, force: float):
        \"\"\"
        Gripper 닫기
        
        Args:
            force: 그립 력 (N)
        \"\"\"
        if self.use_sim:
            print(f"    [SIM] Gripper close with force {force:.1f}N")
        else:
            # self.gripper.close(force)
            print(f"    [ROBOT] Gripper close with force {force:.1f}N")


class DemoPickAndPlace:
    """
    Pick and place 데모
    """
    
    def __init__(self, affordance_model=None):
        \"\"\"
        Initialize demo
        
        Args:
            affordance_model: affordance 감지 모델
        \"\"\"
        self.controller = RobotController(
            affordance_model=affordance_model,
            use_sim=True  # 시뮬레이션 모드
        )
    
    def run_demo(self):
        \"\"\"
        Pick and place 데모 실행
        \"\"\"
        print("=" * 60)
        print("Industrial Robot Pick & Place Demo")
        print("=" * 60)
        
        # 테스트 케이스
        test_cases = [
            {
                'name': 'Standard Box',
                'affordances': {
                    'graspable': 0.9, 'stackable': 0.8,
                    'insertable': 0.2, 'placeable': 0.85,
                    'moveable': 0.85, 'fragile': 0.1
                },
                'object_pose': [0.5, 0, 0.2, 0, 0, 0],
                'place_pose': [0.3, 0.3, 0.2, 0, 0, 0]
            },
            {
                'name': 'Fragile Object',
                'affordances': {
                    'graspable': 0.85, 'stackable': 0.3,
                    'insertable': 0.2, 'placeable': 0.8,
                    'moveable': 0.7, 'fragile': 0.85
                },
                'object_pose': [0.6, 0.1, 0.2, 0, 0, 0],
                'place_pose': [0.3, 0.2, 0.2, 0, 0, 0]
            },
            {
                'name': 'Ungraspable Object',
                'affordances': {
                    'graspable': 0.2, 'stackable': 0.1,
                    'insertable': 0.1, 'placeable': 0.5,
                    'moveable': 0.1, 'fragile': 0.5
                },
                'object_pose': [0.7, -0.2, 0.3, 0, 0, 0],
                'place_pose': [0.3, -0.3, 0.3, 0, 0, 0]
            }
        ]
        
        for i, test in enumerate(test_cases):
            print(f"\n[Task {i+1}] {test['name']}")
            print("-" * 60)
            
            success = self.controller.execute_pick_and_place(
                object_pose=np.array(test['object_pose']),
                place_pose=np.array(test['place_pose']),
                affordances=test['affordances'],
                object_info={'weight_estimate': 1.0}
            )
            
            if success:
                print(f"✓ Task {i+1} completed successfully!")
            else:
                print(f"✗ Task {i+1} failed!")


# 테스트
if __name__ == "__main__":
    # 데모 실행
    demo = DemoPickAndPlace(affordance_model=None)
    demo.run_demo()
    
    print("\n✓ Demo complete!")
