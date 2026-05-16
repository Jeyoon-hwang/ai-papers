"""
Affordance to Robot Action Mapping
affordances → 로봇 제어 파라미터
"""

from typing import Dict, Tuple
import numpy as np


class AffordancePolicy:
    """
    affordance 벡터를 로봇 제어 명령으로 변환
    """
    
    def __init__(self):
        """Initialize policy parameters"""
        self.gripper_force_range = (10, 150)  # Newton
        self.gripper_width_range = (0.0, 0.14)  # Robotiq 2F-140: 0-140mm
        self.approach_speed_range = (0.1, 1.0)  # m/s
        self.grasp_offset_range = (-0.02, 0.02)  # Grasp point offset
        
    def affordance_to_robot_action(self,
                                  affordances: Dict[str, float],
                                  object_info: Dict) -> Dict:
        """
        affordance 벡터를 로봇 액션으로 변환
        
        Args:
            affordances: {
                'graspable': 0-1,
                'stackable': 0-1,
                'insertable': 0-1,
                'placeable': 0-1,
                'moveable': 0-1,
                'fragile': 0-1
            }
            object_info: {
                'position': [x, y, z],
                'size': [dx, dy, dz],
                'weight_estimate': float (kg),
                'shape': 'box'|'cylinder'|'sphere'
            }
        
        Returns:
            action: {
                'gripper_width': float (m),
                'gripper_force': float (N),
                'approach_speed': float (m/s),
                'lift_height': float (m),
                'placement_height': 'table'|'high'|'low',
                'grasp_quality_estimate': float (0-1),
                'recommended_action': str
            }
        """
        
        action = {}
        
        # 1. Gripper 설정
        action['gripper_width'], action['gripper_force'] = \
            self._compute_gripper_params(affordances, object_info)
        
        # 2. 접근 속도
        action['approach_speed'] = self._compute_approach_speed(affordances)
        
        # 3. Lift 높이
        action['lift_height'] = self._compute_lift_height(affordances, object_info)
        
        # 4. 놓기 높이
        action['placement_height'] = self._compute_placement_height(affordances)
        
        # 5. Grasp 품질 추정
        action['grasp_quality_estimate'] = self._estimate_grasp_quality(
            affordances, 
            object_info,
            action
        )
        
        # 6. 추천 행동
        action['recommended_action'] = self._recommend_action(affordances)
        
        return action
    
    def _compute_gripper_params(self,
                               affordances: Dict[str, float],
                               object_info: Dict) -> Tuple[float, float]:
        """
        Gripper opening width와 grip force 계산
        
        Returns:
            (gripper_width, gripper_force)
        """
        
        if affordances['graspable'] < 0.5:
            # Graspable하지 않으면 열어두기
            return self.gripper_width_range[0], 0.0
        
        # 물체 크기 기반 gripper width
        if 'size' in object_info:
            size = np.array(object_info['size'])
            # 가장 작은 두 개 축의 합
            relevant_size = np.sort(size)[:2].sum() / 2
            gripper_width = np.clip(
                relevant_size * 1.2,  # 약간 여유
                self.gripper_width_range[0],
                self.gripper_width_range[1]
            )
        else:
            gripper_width = 0.07  # 기본값
        
        # Gripper force: fragile 여부 + weight
        fragile_penalty = affordances['fragile']
        weight_estimate = object_info.get('weight_estimate', 1.0)
        
        if fragile_penalty > 0.6:
            # Fragile: 약한 그립
            base_force = 30 + weight_estimate * 5
        else:
            # 일반: 강한 그립
            base_force = 80 + weight_estimate * 10
        
        gripper_force = np.clip(base_force, *self.gripper_force_range)
        
        return gripper_width, gripper_force
    
    def _compute_approach_speed(self, affordances: Dict[str, float]) -> float:
        """
        접근 속도: fragile하면 느리게
        """
        fragile = affordances['fragile']
        
        if fragile > 0.7:
            # Very fragile: slow
            return 0.1
        elif fragile > 0.4:
            # Somewhat fragile: medium
            return 0.3
        else:
            # Robust: normal/fast
            return 0.5
    
    def _compute_lift_height(self,
                            affordances: Dict[str, float],
                            object_info: Dict) -> float:
        """
        들어올리는 높이: fragile하면 낮게
        """
        fragile = affordances['fragile']
        
        if fragile > 0.7:
            # Very fragile: just clear table
            return 0.05
        elif fragile > 0.4:
            # Somewhat fragile: moderate
            return 0.15
        else:
            # Robust: high lift
            return 0.3
    
    def _compute_placement_height(self, affordances: Dict[str, float]) -> str:
        """
        놓을 위치 높이: stackable 여부에 따라
        """
        stackable = affordances['stackable']
        
        if stackable > 0.7:
            return 'high'  # Stack on top
        elif stackable > 0.4:
            return 'medium'
        else:
            return 'table'  # Place on table
    
    def _estimate_grasp_quality(self,
                               affordances: Dict[str, float],
                               object_info: Dict,
                               action: Dict) -> float:
        """
        Grasp 성공률 추정 (0-1)
        """
        # Graspable이 높을수록 성공률 높음
        quality = affordances['graspable']
        
        # Moveable도 고려 (움직일 수 있어야 잡을 수 있음)
        quality *= (0.5 + 0.5 * affordances['moveable'])
        
        # Fragile은 성공률을 낮춤 (조심스럽게 다뤄야 함)
        fragile_penalty = affordances['fragile']
        quality *= (1.0 - 0.3 * fragile_penalty)
        
        return np.clip(quality, 0, 1)
    
    def _recommend_action(self, affordances: Dict[str, float]) -> str:
        """
        affordance 기반 추천 행동
        """
        graspable = affordances['graspable']
        stackable = affordances['stackable']
        placeable = affordances['placeable']
        fragile = affordances['fragile']
        
        # 우선순위 결정
        if graspable < 0.5:
            return "CANNOT_GRASP"
        
        if fragile > 0.7:
            return "HANDLE_WITH_EXTREME_CARE"
        
        if stackable > 0.7:
            return "STACK"
        
        if placeable > 0.8:
            return "PLACE_CAREFULLY"
        
        return "STANDARD_PICK_AND_PLACE"


class RobotActionExecutor:
    """
    로봇 액션을 실제 제어 명령으로 변환 (ROS 인터페이스)
    """
    
    def __init__(self):
        """Initialize executor"""
        self.policy = AffordancePolicy()
        
    def execute_action(self, action: Dict) -> bool:
        """
        로봇 액션 실행
        
        Args:
            action: affordance_policy에서 생성한 액션
            
        Returns:
            success: 성공 여부
        """
        print(f"Executing action:")
        print(f"  Gripper width: {action['gripper_width']:.4f} m")
        print(f"  Gripper force: {action['gripper_force']:.1f} N")
        print(f"  Approach speed: {action['approach_speed']:.2f} m/s")
        print(f"  Lift height: {action['lift_height']:.3f} m")
        print(f"  Placement: {action['placement_height']}")
        print(f"  Quality estimate: {action['grasp_quality_estimate']:.2%}")
        print(f"  Recommended: {action['recommended_action']}")
        
        # 이곳에서 실제 ROS 명령 전송
        return True
    
    def pick_and_place(self,
                      obj_pos: np.ndarray,
                      place_pos: np.ndarray,
                      affordances: Dict[str, float],
                      object_info: Dict = None) -> bool:
        """
        Pick and place 작업 실행
        """
        if object_info is None:
            object_info = {'weight_estimate': 1.0}
        
        # Affordance → Action
        action = self.policy.affordance_to_robot_action(affordances, object_info)
        
        # 안전성 검증
        if action['grasp_quality_estimate'] < 0.5:
            print(f"Warning: Low grasp quality ({action['grasp_quality_estimate']:.2%})")
        
        # Action 실행
        success = self.execute_action(action)
        
        return success


# 테스트
if __name__ == "__main__":
    policy = AffordancePolicy()
    executor = RobotActionExecutor()
    
    # 테스트 케이스 1: 일반 박스
    affordances_box = {
        'graspable': 0.9,
        'stackable': 0.8,
        'insertable': 0.2,
        'placeable': 0.85,
        'moveable': 0.85,
        'fragile': 0.1
    }
    object_info_box = {
        'size': [0.1, 0.1, 0.1],
        'weight_estimate': 2.0,
        'shape': 'box'
    }
    
    print("=== Case 1: Standard Box ===")
    action = policy.affordance_to_robot_action(affordances_box, object_info_box)
    executor.execute_action(action)
    
    # 테스트 케이스 2: 깨지기 쉬운 물체
    affordances_fragile = {
        'graspable': 0.85,
        'stackable': 0.3,
        'insertable': 0.2,
        'placeable': 0.8,
        'moveable': 0.7,
        'fragile': 0.85
    }
    object_info_fragile = {
        'size': [0.08, 0.08, 0.15],
        'weight_estimate': 0.5,
        'shape': 'cylinder'
    }
    
    print("\n=== Case 2: Fragile Object ===")
    action = policy.affordance_to_robot_action(affordances_fragile, object_info_fragile)
    executor.execute_action(action)
    
    # 테스트 케이스 3: 집을 수 없는 물체
    affordances_ungraspable = {
        'graspable': 0.2,
        'stackable': 0.1,
        'insertable': 0.1,
        'placeable': 0.5,
        'moveable': 0.1,
        'fragile': 0.5
    }
    object_info_ungraspable = {
        'size': [0.5, 0.5, 0.5],
        'weight_estimate': 50.0,
        'shape': 'box'
    }
    
    print("\n=== Case 3: Ungraspable Object ===")
    action = policy.affordance_to_robot_action(affordances_ungraspable, object_info_ungraspable)
    executor.execute_action(action)
