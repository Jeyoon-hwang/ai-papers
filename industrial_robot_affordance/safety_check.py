"""
Safety Validation for Robot Actions
로봇 안전성 검증
"""

from typing import Dict, Tuple, List
import numpy as np


class SafetyValidator:
    """
    로봇 행동의 안전성 검증
    """
    
    # 안전 임계값
    MAX_JOINT_TORQUE = {
        'shoulder_pan': 330,      # Nm
        'shoulder_lift': 330,
        'elbow': 150,
        'wrist_1': 54,
        'wrist_2': 54,
        'wrist_3': 54
    }
    
    MAX_GRIPPER_FORCE = 150  # Newton
    MIN_GRIPPER_FORCE = 5    # Newton (미끄러지지 않도록)
    
    MAX_APPROACH_SPEED = 1.0  # m/s
    MIN_APPROACH_SPEED = 0.05  # m/s (너무 느림 방지)
    
    COLLISION_CLEARANCE = 0.02  # m
    
    def __init__(self):
        """Initialize validator"""
        self.violation_log = []
    
    def validate_grasp(self,
                      obj_id: int,
                      affordances: Dict[str, float],
                      gripper_config: Dict,
                      joint_torques: Dict[str, float] = None,
                      object_position: np.ndarray = None) -> Tuple[bool, List[str]]:
        """
        Grasp 작업의 안전성 전체 검증
        
        Args:
            obj_id: 물체 ID
            affordances: affordance 벡터
            gripper_config: gripper 설정
            joint_torques: 각 joint의 토크 (optional)
            object_position: 물체 위치 (optional)
            
        Returns:
            (is_safe, warnings/errors)
        """
        checks = {}
        issues = []
        
        # 1. Graspability 확인
        checks['graspable'] = affordances['graspable'] > 0.5
        if not checks['graspable']:
            issues.append(f"Object not graspable (graspable={affordances['graspable']:.2f})")
        
        # 2. Gripper force 확인
        checks['gripper_force'] = self.check_gripper_force(gripper_config)
        if not checks['gripper_force'][0]:
            issues.extend(checks['gripper_force'][1])
        
        # 3. Stability 확인
        checks['stability'] = self.check_grasp_stability(affordances, gripper_config)
        if not checks['stability'][0]:
            issues.extend(checks['stability'][1])
        
        # 4. Joint torque 확인
        if joint_torques:
            checks['torque'] = self.check_joint_torques(joint_torques)
            if not checks['torque'][0]:
                issues.extend(checks['torque'][1])
        
        # 5. Collision 확인
        if object_position is not None:
            checks['collision'] = self.check_collision(object_position)
            if not checks['collision'][0]:
                issues.extend(checks['collision'][1])
        
        # 전체 안전성 판단
        is_safe = all(c[0] if isinstance(c, tuple) else c for c in checks.values())
        
        return is_safe, issues
    
    def check_gripper_force(self, gripper_config: Dict) -> Tuple[bool, List[str]]:
        """
        Gripper force 안전성 검증
        """
        force = gripper_config.get('gripper_force', 0)
        issues = []
        
        if force < self.MIN_GRIPPER_FORCE:
            issues.append(f"Gripper force too low ({force:.1f}N < {self.MIN_GRIPPER_FORCE}N)")
        
        if force > self.MAX_GRIPPER_FORCE:
            issues.append(f"Gripper force too high ({force:.1f}N > {self.MAX_GRIPPER_FORCE}N)")
        
        return len(issues) == 0, issues
    
    def check_grasp_stability(self,
                            affordances: Dict[str, float],
                            gripper_config: Dict) -> Tuple[bool, List[str]]:
        """
        Grasp 안정성 검증
        
        fragile 물체는 약한 그립, 일반 물체는 강한 그립
        """
        issues = []
        force = gripper_config.get('gripper_force', 0)
        fragile = affordances['fragile']
        
        # Fragile 물체 검증
        if fragile > 0.7:
            # Very fragile: force < 50N
            if force > 50:
                issues.append(
                    f"Force too high for fragile object "
                    f"({force:.1f}N > 50N, fragile={fragile:.2f})"
                )
        
        # 일반 물체 검증
        else:
            # Normal objects: force >= 30N
            if force < 30:
                issues.append(
                    f"Force too low for standard grasp "
                    f"({force:.1f}N < 30N)"
                )
        
        return len(issues) == 0, issues
    
    def check_joint_torques(self, joint_torques: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Joint torque 안전성 검증
        """
        issues = []
        
        for joint_name, torque in joint_torques.items():
            max_torque = self.MAX_JOINT_TORQUE.get(joint_name)
            
            if max_torque and abs(torque) > max_torque:
                issues.append(
                    f"Torque limit exceeded: {joint_name} "
                    f"({abs(torque):.1f} Nm > {max_torque} Nm)"
                )
        
        return len(issues) == 0, issues
    
    def check_collision(self,
                       object_position: np.ndarray,
                       robot_workspace: Dict = None) -> Tuple[bool, List[str]]:
        """
        충돌 감지
        
        Args:
            object_position: [x, y, z]
            robot_workspace: {x_min, x_max, y_min, y_max, z_min, z_max}
        """
        issues = []
        
        if robot_workspace is None:
            # UR10e 기본 workspace
            robot_workspace = {
                'x_min': -1.3, 'x_max': 1.3,
                'y_min': -1.3, 'y_max': 1.3,
                'z_min': 0, 'z_max': 1.65
            }
        
        # 물체가 workspace 내에 있는지 확인
        if not (robot_workspace['x_min'] <= object_position[0] <= robot_workspace['x_max']):
            issues.append(f"Object X position out of workspace")
        
        if not (robot_workspace['y_min'] <= object_position[1] <= robot_workspace['y_max']):
            issues.append(f"Object Y position out of workspace")
        
        if not (robot_workspace['z_min'] <= object_position[2] <= robot_workspace['z_max']):
            issues.append(f"Object Z position out of workspace")
        
        return len(issues) == 0, issues
    
    def check_approach_speed(self,
                            affordances: Dict[str, float],
                            speed: float) -> Tuple[bool, List[str]]:
        """
        접근 속도 검증
        """
        issues = []
        
        if speed < self.MIN_APPROACH_SPEED:
            issues.append(f"Speed too low ({speed:.3f} m/s < {self.MIN_APPROACH_SPEED} m/s)")
        
        if speed > self.MAX_APPROACH_SPEED:
            issues.append(f"Speed too high ({speed:.3f} m/s > {self.MAX_APPROACH_SPEED} m/s)")
        
        # Fragile 물체는 느린 속도
        if affordances['fragile'] > 0.7 and speed > 0.2:
            issues.append(
                f"Speed too high for fragile object "
                f"({speed:.3f} m/s > 0.2 m/s)"
            )
        
        return len(issues) == 0, issues
    
    def generate_safety_report(self,
                              affordances: Dict[str, float],
                              gripper_config: Dict,
                              object_info: Dict = None) -> Dict:
        """
        안전성 리포트 생성
        """
        report = {
            'timestamp': None,
            'is_safe': True,
            'checks': {},
            'recommendations': []
        }
        
        # Graspability
        is_graspable = affordances['graspable'] > 0.5
        report['checks']['graspable'] = {
            'passed': is_graspable,
            'value': affordances['graspable'],
            'threshold': 0.5
        }
        if not is_graspable:
            report['is_safe'] = False
            report['recommendations'].append("Object may not be graspable - verify before proceeding")
        
        # Gripper force
        force = gripper_config.get('gripper_force', 0)
        force_ok = self.MIN_GRIPPER_FORCE <= force <= self.MAX_GRIPPER_FORCE
        report['checks']['gripper_force'] = {
            'passed': force_ok,
            'value': force,
            'range': [self.MIN_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE]
        }
        if not force_ok:
            report['is_safe'] = False
        
        # Stability
        fragile = affordances['fragile']
        if fragile > 0.7 and force > 50:
            report['is_safe'] = False
            report['recommendations'].append("Reduce grip force for fragile object")
        
        # Fragility warning
        if fragile > 0.7:
            report['recommendations'].append("Use slow approach speed and careful handling")
        
        return report


# 테스트
if __name__ == "__main__":
    validator = SafetyValidator()
    
    # 테스트 케이스 1: 안전한 표준 그립
    print("=== Test 1: Safe Standard Grasp ===")
    affordances = {
        'graspable': 0.9,
        'stackable': 0.8,
        'insertable': 0.2,
        'placeable': 0.85,
        'moveable': 0.85,
        'fragile': 0.1
    }
    gripper_config = {
        'gripper_force': 80,
        'gripper_width': 0.07,
        'approach_speed': 0.3
    }
    
    is_safe, issues = validator.validate_grasp(1, affordances, gripper_config)
    print(f"Safe: {is_safe}")
    if issues:
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ All checks passed")
    
    # 테스트 케이스 2: 깨지기 쉬운 물체 (force 너무 높음)
    print("\n=== Test 2: Fragile Object (Force Too High) ===")
    affordances_fragile = {
        'graspable': 0.85,
        'stackable': 0.3,
        'insertable': 0.2,
        'placeable': 0.8,
        'moveable': 0.7,
        'fragile': 0.85
    }
    gripper_config_wrong = {
        'gripper_force': 120,  # Too high for fragile
        'gripper_width': 0.05,
        'approach_speed': 0.5  # Too fast
    }
    
    is_safe, issues = validator.validate_grasp(2, affordances_fragile, gripper_config_wrong)
    print(f"Safe: {is_safe}")
    for issue in issues:
        print(f"  ✗ {issue}")
    
    # 테스트 케이스 3: 집을 수 없는 물체
    print("\n=== Test 3: Ungraspable Object ===")
    affordances_ungraspable = {
        'graspable': 0.2,
        'stackable': 0.1,
        'insertable': 0.1,
        'placeable': 0.5,
        'moveable': 0.1,
        'fragile': 0.5
    }
    gripper_config_bad = {
        'gripper_force': 50,
        'gripper_width': 0.14,
        'approach_speed': 0.1
    }
    
    is_safe, issues = validator.validate_grasp(3, affordances_ungraspable, gripper_config_bad)
    print(f"Safe: {is_safe}")
    for issue in issues:
        print(f"  ✗ {issue}")
    
    # 안전성 리포트
    print("\n=== Safety Report ===")
    report = validator.generate_safety_report(affordances, gripper_config)
    print(f"Overall Safe: {report['is_safe']}")
    print(f"Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
