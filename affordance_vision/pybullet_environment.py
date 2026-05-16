#!/usr/bin/env python3
"""
PyBullet-based Affordance Learning Environment
Replaces Pygame with physics-based simulation

Author: FFAL Project
Date: May 2026
"""

import pybullet as p
import pybullet_data
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# AFFORDANCE DEFINITIONS (Physics-Based)
# ============================================================================

class Affordance(Enum):
    """Six core robot affordances with physics-based criteria"""
    SITTABLE = "sittable"
    PUSHABLE = "pushable"
    CLIMBABLE = "climbable"
    BREAKABLE = "breakable"
    HOLDABLE = "holdable"
    STACKABLE = "stackable"


@dataclass
class AffordanceResult:
    """Result of affordance test"""
    affordance: str
    success: bool
    confidence: float  # 0.0-1.0
    details: Dict  # e.g., {"tilt_angle": 12.3, "max_displacement": 8.5}


@dataclass
class SimulationFrame:
    """Single frame from simulation"""
    frame_id: int
    object_id: int
    object_name: str
    affordances: Dict[str, bool]  # {affordance_name: success}
    object_state: Dict  # position, rotation, velocity
    timestamp: float


# ============================================================================
# PYBULLET ENVIRONMENT
# ============================================================================

class AffordanceEnvironment:
    """PyBullet environment for affordance learning"""
    
    def __init__(self, gui=False, gravity=-9.81):
        """
        Args:
            gui: Whether to show PyBullet GUI
            gravity: Gravitational acceleration (m/s²)
        """
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity, physicsClientId=self.client_id)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # Simulation parameters
        self.dt = 0.01  # timestep (100 Hz)
        self.sub_steps = 240  # for stability
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSubSteps=self.sub_steps,
            physicsClientId=self.client_id
        )
        
        self.frame_counter = 0
        self.episode_data = []
        
    def load_object_from_mesh(self, mesh_path: str, scale: float = 1.0, 
                             friction: float = 0.5, density: float = 1000.0) -> int:
        """
        Load object from OBJ/URDF mesh
        
        Args:
            mesh_path: Path to .obj or .urdf file
            scale: Scale factor
            friction: Friction coefficient (0.3-0.8)
            density: Material density (kg/m³)
        
        Returns:
            object_id in PyBullet
        """
        try:
            if mesh_path.endswith('.urdf'):
                obj_id = p.loadURDF(mesh_path, 
                                   basePosition=[0, 0, 0.5],
                                   globalScaling=scale,
                                   physicsClientId=self.client_id)
            else:
                # For OBJ files, need to convert to URDF or load as mesh
                obj_id = p.loadURDF(mesh_path,
                                   basePosition=[0, 0, 0.5],
                                   globalScaling=scale,
                                   physicsClientId=self.client_id)
            
            # Set physics properties
            p.changeDynamics(obj_id, -1,
                           lateralFriction=friction,
                           restitution=0.3,  # bounciness
                           linearDamping=0.04,
                           angularDamping=0.04,
                           physicsClientId=self.client_id)
            
            return obj_id
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            # Fallback: create simple shape (for testing)
            return self._create_simple_box(scale, friction)
    
    def _create_simple_box(self, scale: float, friction: float) -> int:
        """Create simple box as fallback"""
        size = [0.1 * scale, 0.1 * scale, 0.1 * scale]
        shape_id = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=size,
                                         physicsClientId=self.client_id)
        body_id = p.createMultiBody(baseMass=1.0,
                                   baseCollisionShapeIndex=shape_id,
                                   basePosition=[0, 0, 0.5],
                                   physicsClientId=self.client_id)
        
        p.changeDynamics(body_id, -1, lateralFriction=friction,
                        physicsClientId=self.client_id)
        return body_id
    
    # ========================================================================
    # AFFORDANCE TESTS (Physics-Based Criteria)
    # ========================================================================
    
    def test_sittable(self, obj_id: int, agent_mass: float = 75.0, 
                      duration: float = 3.0, tilt_threshold: float = 15.0) -> AffordanceResult:
        """
        Test if object is sittable
        
        Criterion: Object supports agent mass without tilting >15°
        
        Args:
            obj_id: Object ID
            agent_mass: Agent mass (kg)
            duration: Duration to test stability (seconds)
            tilt_threshold: Maximum tilt angle (degrees)
        
        Returns:
            AffordanceResult
        """
        # Place box on object to simulate agent sitting
        _, top_pos = p.getBasePositionAndOrientation(obj_id, 
                                                    physicsClientId=self.client_id)
        agent_box = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[0.2, 0.2, 0.15],
                                          physicsClientId=self.client_id)
        agent_id = p.createMultiBody(baseMass=agent_mass,
                                    baseCollisionShapeIndex=agent_box,
                                    basePosition=[top_pos[0], top_pos[1], top_pos[2] + 0.5],
                                    physicsClientId=self.client_id)
        
        # Simulate for duration
        steps = int(duration / self.dt)
        max_tilt = 0.0
        
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            
            # Check object tilt (rotation)
            _, quat = p.getBasePositionAndOrientation(obj_id,
                                                     physicsClientId=self.client_id)
            rpy = p.getEulerFromQuaternion(quat)
            tilt = np.sqrt(rpy[0]**2 + rpy[1]**2)  # roll + pitch
            tilt_deg = np.degrees(tilt)
            max_tilt = max(max_tilt, tilt_deg)
        
        success = max_tilt < tilt_threshold
        confidence = 1.0 - min(max_tilt / 45.0, 1.0)
        
        # Cleanup
        p.removeBody(agent_id, physicsClientId=self.client_id)
        
        return AffordanceResult(
            affordance=Affordance.SITTABLE.value,
            success=success,
            confidence=confidence,
            details={"max_tilt_deg": float(max_tilt)}
        )
    
    def test_pushable(self, obj_id: int, push_force: float = 50.0,
                     min_displacement: float = 0.1, duration: float = 2.0) -> AffordanceResult:
        """
        Test if object is pushable
        
        Criterion: Object moves >10cm in response to 50N push
        
        Args:
            obj_id: Object ID
            push_force: Applied force (Newtons)
            min_displacement: Minimum displacement (meters)
            duration: Duration of test (seconds)
        
        Returns:
            AffordanceResult
        """
        # Record initial position
        init_pos, _ = p.getBasePositionAndOrientation(obj_id,
                                                      physicsClientId=self.client_id)
        
        # Apply force
        p.applyExternalForce(obj_id, -1,
                           [push_force, 0, 0],  # Push in X direction
                           [0, 0, 0],  # At center of mass
                           p.WORLD_FRAME,
                           physicsClientId=self.client_id)
        
        # Simulate
        steps = int(duration / self.dt)
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
        
        # Measure displacement
        final_pos, _ = p.getBasePositionAndOrientation(obj_id,
                                                       physicsClientId=self.client_id)
        displacement = np.linalg.norm(np.array(final_pos) - np.array(init_pos))
        
        success = displacement >= min_displacement
        confidence = min(displacement / (min_displacement * 2.0), 1.0)
        
        return AffordanceResult(
            affordance=Affordance.PUSHABLE.value,
            success=success,
            confidence=confidence,
            details={"displacement_m": float(displacement)}
        )
    
    def test_climbable(self, obj_id: int, reach_height: float = 1.5) -> AffordanceResult:
        """
        Test if object is climbable
        
        Criterion: Top surface reachable within reach_height
        
        Args:
            obj_id: Object ID
            reach_height: Maximum reach height (meters)
        
        Returns:
            AffordanceResult
        """
        pos, _ = p.getBasePositionAndOrientation(obj_id,
                                                 physicsClientId=self.client_id)
        
        # Get object bounds
        aabb = p.getAABB(obj_id, -1, physicsClientId=self.client_id)
        top_height = aabb[1][2]  # Z coordinate of top
        
        # Agent height (standing)
        agent_reach = pos[2] + reach_height
        
        success = top_height <= agent_reach
        confidence = 1.0 - min(max(top_height - agent_reach, 0) / reach_height, 1.0)
        
        return AffordanceResult(
            affordance=Affordance.CLIMBABLE.value,
            success=success,
            confidence=confidence,
            details={"top_height_m": float(top_height), "reach_height_m": reach_height}
        )
    
    def test_breakable(self, obj_id: int, drop_mass: float = 5.0,
                      drop_height: float = 1.0, integrity_threshold: float = 0.3) -> AffordanceResult:
        """
        Test if object is breakable
        
        Criterion: Object structural integrity <30% after impact
        
        NOTE: PyBullet doesn't natively support structural damage.
        We simulate "breakability" by whether object can withstand impact without
        large deformation/displacement.
        
        Args:
            obj_id: Object ID
            drop_mass: Mass of dropped object (kg)
            drop_height: Drop height (meters)
            integrity_threshold: Threshold below which object breaks
        
        Returns:
            AffordanceResult
        """
        # For now, estimate breakability from material properties
        # Lighter, smaller objects → more breakable
        try:
            mass_info = p.getDynamicsInfo(obj_id, -1, physicsClientId=self.client_id)
            mass = mass_info[0]
            aabb = p.getAABB(obj_id, -1, physicsClientId=self.client_id)
            size = np.linalg.norm(np.array(aabb[1]) - np.array(aabb[0]))
            
            # Heuristic: light + small = more breakable
            breakability_score = (1.0 / (mass + 0.1)) * (1.0 / (size + 0.1))
            breakability_norm = min(breakability_score / 0.5, 1.0)
            
            success = breakability_norm > (1.0 - integrity_threshold)
            confidence = breakability_norm
            
        except:
            success = False
            confidence = 0.0
        
        return AffordanceResult(
            affordance=Affordance.BREAKABLE.value,
            success=success,
            confidence=confidence,
            details={"breakability_heuristic": float(confidence)}
        )
    
    def test_holdable(self, obj_id: int, max_size: Tuple = (0.15, 0.15, 0.20),
                     max_mass: float = 2.0) -> AffordanceResult:
        """
        Test if object is holdable
        
        Criterion: Fits within grasp constraints
        
        Args:
            obj_id: Object ID
            max_size: Maximum dimensions (meters)
            max_mass: Maximum mass (kg)
        
        Returns:
            AffordanceResult
        """
        try:
            # Check mass
            mass_info = p.getDynamicsInfo(obj_id, -1, physicsClientId=self.client_id)
            mass = mass_info[0]
            
            # Check size
            aabb = p.getAABB(obj_id, -1, physicsClientId=self.client_id)
            dimensions = tuple(aabb[1][i] - aabb[0][i] for i in range(3))
            
            # All dimensions must fit
            size_ok = all(dimensions[i] <= max_size[i] for i in range(3))
            mass_ok = mass <= max_mass
            
            success = size_ok and mass_ok
            
            # Confidence based on "graspability"
            size_confidence = min(1.0, np.prod(max_size) / np.prod(dimensions))
            mass_confidence = 1.0 - min(mass / max_mass, 1.0)
            confidence = (size_confidence + mass_confidence) / 2.0
            
        except:
            success = False
            confidence = 0.0
        
        return AffordanceResult(
            affordance=Affordance.HOLDABLE.value,
            success=success,
            confidence=confidence,
            details={"size_ok": size_ok, "mass_ok": mass_ok}
        )
    
    def test_stackable(self, obj_id: int, duration: float = 3.0) -> AffordanceResult:
        """
        Test if object is stackable
        
        Criterion: Two stacked objects remain stable for 3 seconds
        
        Args:
            obj_id: Object ID (base object)
            duration: Stability duration (seconds)
        
        Returns:
            AffordanceResult
        """
        # Get object dimensions
        aabb = p.getAABB(obj_id, -1, physicsClientId=self.client_id)
        height = aabb[1][2] - aabb[0][2]
        
        # Create duplicate object on top
        pos, orn = p.getBasePositionAndOrientation(obj_id,
                                                    physicsClientId=self.client_id)
        stack_pos = [pos[0], pos[1], pos[2] + height + 0.01]
        
        # Clone object
        stack_id = p.loadURDF(p.getObjectProperties(obj_id, -1, self.client_id)[1],
                             basePosition=stack_pos,
                             physicsClientId=self.client_id)
        
        # Simulate
        steps = int(duration / self.dt)
        max_tilt = 0.0
        
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            
            # Check if stacked object tips over
            _, quat = p.getBasePositionAndOrientation(stack_id,
                                                      physicsClientId=self.client_id)
            rpy = p.getEulerFromQuaternion(quat)
            tilt = np.sqrt(rpy[0]**2 + rpy[1]**2)
            max_tilt = max(max_tilt, tilt)
        
        # Success if stays upright
        success = max_tilt < np.radians(20.0)  # < 20° tilt
        confidence = max(0.0, 1.0 - max_tilt / np.radians(45.0))
        
        # Cleanup
        p.removeBody(stack_id, physicsClientId=self.client_id)
        
        return AffordanceResult(
            affordance=Affordance.STACKABLE.value,
            success=success,
            confidence=confidence,
            details={"max_tilt_rad": float(max_tilt)}
        )
    
    def test_all_affordances(self, obj_id: int) -> Dict[str, AffordanceResult]:
        """Test all affordances for an object"""
        results = {
            Affordance.SITTABLE.value: self.test_sittable(obj_id),
            Affordance.PUSHABLE.value: self.test_pushable(obj_id),
            Affordance.CLIMBABLE.value: self.test_climbable(obj_id),
            Affordance.BREAKABLE.value: self.test_breakable(obj_id),
            Affordance.HOLDABLE.value: self.test_holdable(obj_id),
            Affordance.STACKABLE.value: self.test_stackable(obj_id),
        }
        return results
    
    def step(self):
        """Single simulation step"""
        p.stepSimulation(physicsClientId=self.client_id)
        self.frame_counter += 1
    
    def reset(self):
        """Reset environment"""
        p.resetSimulation(physicsClientId=self.client_id)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.frame_counter = 0
    
    def disconnect(self):
        """Close PyBullet"""
        p.disconnect(physicsClientId=self.client_id)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_affordance_dataset(output_dir: str = "data/pybullet_train",
                               num_objects: int = 30,
                               num_episodes: int = 500,
                               episode_length: int = 10):
    """
    Collect affordance dataset using PyBullet
    
    Args:
        output_dir: Output directory
        num_objects: Number of objects to test
        num_episodes: Number of episodes per object
        episode_length: Episode length (seconds)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize environment
    env = AffordanceEnvironment(gui=False)
    
    # For now, use simple shapes as placeholder
    # In real implementation, load ShapeNet objects
    object_configs = [
        {"name": f"box_{i}", "type": "box"} for i in range(num_objects)
    ]
    
    all_data = []
    
    for ep in range(num_episodes):
        for obj_config in object_configs:
            # Create object
            obj_id = env._create_simple_box(scale=1.0, friction=0.5)
            
            # Test affordances
            affordances = env.test_all_affordances(obj_id)
            
            # Record
            record = {
                "episode": ep,
                "object": obj_config["name"],
                "affordances": {k: asdict(v) for k, v in affordances.items()}
            }
            all_data.append(record)
            
            # Cleanup
            p.removeBody(obj_id, physicsClientId=env.client_id)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes} completed")
    
    # Save dataset
    output_file = os.path.join(output_dir, "affordance_dataset.json")
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    env.disconnect()


if __name__ == "__main__":
    # Test environment
    print("Testing PyBullet Affordance Environment...")
    env = AffordanceEnvironment(gui=False)
    
    # Create test object
    test_obj = env._create_simple_box(scale=1.0, friction=0.5)
    
    # Test affordances
    results = env.test_all_affordances(test_obj)
    
    for affordance, result in results.items():
        print(f"\n{affordance.upper()}:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Details: {result.details}")
    
    env.disconnect()
    print("\n✅ PyBullet environment test complete!")
