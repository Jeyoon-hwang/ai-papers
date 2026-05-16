"""
Industrial Robot Simulation Environment
UR10e + Robotiq 2F-140 gripper + PyBullet
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, Dict, List
import os


class IndustrialRobotEnv:
    """UR10e 로봇 + Robotiq 그리퍼 + 산업 물건"""
    
    def __init__(self, use_gui=False):
        """
        Initialize industrial robot environment
        
        Args:
            use_gui: True이면 PyBullet GUI 활성화
        """
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Plane 추가
        self.plane_id = p.loadURDF("plane.urdf")
        
        # UR10e 로드 (URDF 대체)
        self.robot_id = self._load_ur10e()
        
        # Robotiq 그리퍼 (간단한 모델)
        self.gripper_id = self._load_gripper()
        
        # 산업 물건들
        self.objects = {}
        self.affordance_ground_truth = {}
        
        # 카메라 설정
        self.camera_distance = 1.5
        self.camera_yaw = 0
        self.camera_pitch = -45
        self.camera_target = [0.5, 0, 0.5]
        
    def _load_ur10e(self) -> int:
        """
        UR10e 로봇 로드 (간단한 6축 모델)
        실제로는 Universal Robots에서 URDF 다운로드 필요
        """
        # URDF 파일 생성 (임시)
        urdf_content = """<?xml version="1.0" ?>
<robot name="ur10e">
  <link name="base_link">
    <inertial>
      <mass value="40.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
      <material name="gray"/>
    </visual>
  </link>
  
  <joint name="shoulder_pan_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.13585" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="150" velocity="2.0"/>
  </joint>
  
  <link name="shoulder_link">
    <inertial>
      <mass value="3.7"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>"""
        
        urdf_path = "/tmp/ur10e_simple.urdf"
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        # 더 간단하게: 기존 URDF 사용
        try:
            robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0])
            return robot_id
        except:
            # Fallback: 기본 object로 대체
            return p.loadURDF("plane.urdf")
    
    def _load_gripper(self) -> int:
        """Robotiq 2F-140 그리퍼 (간단한 모델)"""
        gripper_urdf = "/tmp/simple_gripper.urdf"
        
        urdf_content = """<?xml version="1.0" ?>
<robot name="gripper">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.05 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
  
  <joint name="left_finger_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="left_finger"/>
    <origin xyz="-0.025 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.04" upper="0.04" effort="100" velocity="0.5"/>
  </joint>
  
  <link name="left_finger">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.02 0.08 0.05"/>
      </geometry>
    </visual>
  </link>
  
  <joint name="right_finger_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="right_finger"/>
    <origin xyz="0.025 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit lower="-0.04" upper="0.04" effort="100" velocity="0.5"/>
  </joint>
  
  <link name="right_finger">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.02 0.08 0.05"/>
      </geometry>
    </visual>
  </link>
</robot>"""
        
        with open(gripper_urdf, 'w') as f:
            f.write(urdf_content)
        
        try:
            gripper_id = p.loadURDF(gripper_urdf, [0, 0, 0.15])
            return gripper_id
        except:
            return -1
    
    def spawn_industrial_object(self, 
                               obj_type: str, 
                               position: Tuple[float, float, float],
                               affordances: Dict[str, float]) -> int:
        """
        산업 물건 생성
        
        obj_type: 'box', 'cylinder', 'sphere', 'complex'
        affordances: {graspable, stackable, insertable, placeable, moveable, fragile}
        """
        # URDF 생성
        urdf_path = f"/tmp/{obj_type}_{len(self.objects)}.urdf"
        
        if obj_type == 'box':
            size = np.random.uniform([0.05, 0.05, 0.05], [0.15, 0.15, 0.15])
            mass = np.prod(size) * 100  # 밀도 기반
            
            urdf_content = f"""<?xml version="1.0" ?>
<robot name="box">
  <link name="base_link">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
    </collision>
  </link>
</robot>"""
        
        elif obj_type == 'cylinder':
            radius = np.random.uniform(0.02, 0.08)
            height = np.random.uniform(0.05, 0.2)
            mass = np.pi * radius**2 * height * 100
            
            urdf_content = f"""<?xml version="1.0" ?>
<robot name="cylinder">
  <link name="base_link">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{radius}" length="{height}"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{radius}" length="{height}"/>
      </geometry>
    </collision>
  </link>
</robot>"""
        
        else:  # sphere
            radius = np.random.uniform(0.02, 0.08)
            mass = (4/3) * np.pi * radius**3 * 100
            
            urdf_content = f"""<?xml version="1.0" ?>
<robot name="sphere">
  <link name="base_link">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="{radius}"/>
      </geometry>
      <material name="green">
        <color rgba="0.0 1.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="{radius}"/>
      </geometry>
    </collision>
  </link>
</robot>"""
        
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        obj_id = p.loadURDF(urdf_path, position)
        
        obj_name = f"{obj_type}_{len(self.objects)}"
        self.objects[obj_name] = {
            'id': obj_id,
            'type': obj_type,
            'position': position,
        }
        self.affordance_ground_truth[obj_name] = affordances
        
        return obj_id
    
    def capture_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        RGB-D 이미지 캡처 (카메라 시뮬레이션)
        
        Returns:
            rgb: (480, 640, 3) uint8
            depth: (480, 640) float32 (미터 단위)
        """
        width, height = 640, 480
        
        # 카메라 매트릭스
        view_matrix = p.computeViewMatrixFromYawPitchDistance(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.01,
            farVal=100
        )
        
        w, h, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        
        rgb_array = np.array(rgb, dtype=np.uint8)[:, :, :3]
        depth_array = np.array(depth, dtype=np.float32)
        
        return rgb_array, depth_array
    
    def detect_affordances_from_physics(self, obj_name: str) -> Dict[str, float]:
        """
        물리 시뮬레이션 기반 affordance 계산 (ground truth)
        
        Returns:
            affordances: {graspable, stackable, insertable, placeable, moveable, fragile}
        """
        obj_data = self.objects[obj_name]
        obj_id = obj_data['id']
        
        # 물체 정보
        mass = p.getDynamicsInfo(obj_id, -1)[0]
        aabb = p.getAABB(obj_id, -1)
        size = np.array(aabb[1]) - np.array(aabb[0])
        
        # Affordance 계산 (휴리스틱)
        affordances = {}
        
        # Graspable: 적당한 크기, 너무 무겁지 않음
        size_ok = (0.02 < np.min(size) < 0.2) and (np.max(size) < 0.5)
        weight_ok = mass < 5.0
        affordances['graspable'] = float(size_ok and weight_ok) * 0.9 + 0.05
        
        # Stackable: 박스 형태, 안정적
        is_box = obj_data['type'] == 'box'
        affordances['stackable'] = float(is_box) * 0.8 + 0.1
        
        # Insertable: 작은 물체, 원통형 또는 구형
        is_small = np.max(size) < 0.1
        is_round = obj_data['type'] in ['cylinder', 'sphere']
        affordances['insertable'] = float(is_small and is_round) * 0.7 + 0.1
        
        # Placeable: 모든 물체 배치 가능 (기본값 높음)
        affordances['placeable'] = 0.85 + np.random.uniform(-0.05, 0.1)
        
        # Moveable: 너무 무겁지 않음
        affordances['moveable'] = float(mass < 3.0) * 0.8 + 0.2
        
        # Fragile: 가벼운 물체 또는 큰 물체 (깨지기 쉬움)
        is_light = mass < 0.5
        is_large = np.max(size) > 0.15
        affordances['fragile'] = float(is_light or is_large) * 0.7 + 0.1
        
        return affordances
    
    def reset(self):
        """환경 초기화"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        self.objects = {}
        self.affordance_ground_truth = {}
    
    def step(self, dt: float = 1/240):
        """물리 시뮬레이션 한 스텝"""
        p.stepSimulation()
    
    def disconnect(self):
        """PyBullet 종료"""
        p.disconnect(self.client)


if __name__ == "__main__":
    # 테스트
    env = IndustrialRobotEnv(use_gui=False)
    
    # 샘플 물건 생성
    env.spawn_industrial_object(
        'box',
        [0.3, 0, 0.2],
        {'graspable': 0.9, 'stackable': 0.8}
    )
    
    env.spawn_industrial_object(
        'cylinder',
        [0.6, 0, 0.2],
        {'graspable': 0.7, 'insertable': 0.8}
    )
    
    # 이미지 캡처
    rgb, depth = env.capture_image()
    print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
    
    # Affordance 감지
    for obj_name in env.objects:
        aff = env.detect_affordances_from_physics(obj_name)
        print(f"{obj_name}: {aff}")
    
    env.disconnect()
