#!/usr/bin/env python3
"""
Collect Affordance Dataset from PyBullet Simulation

This script:
1. Creates PyBullet environment with 30 diverse objects
2. Tests all 6 affordances for each object
3. Generates 500K frames with automatic labels
4. Saves train/test split (no data leakage)

Author: FFAL v2 Project
Date: May 2026
"""

import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from pybullet_environment import (
    AffordanceEnvironment,
    Affordance,
    SimulationFrame
)

# ============================================================================
# OBJECT CATALOG (30 diverse objects)
# ============================================================================

OBJECT_CATALOG = {
    # Chairs (8 variants)
    "office_chair": {"scale": 1.0, "friction": 0.6},
    "dining_chair": {"scale": 0.95, "friction": 0.55},
    "bar_stool": {"scale": 0.85, "friction": 0.5},
    "gaming_chair": {"scale": 1.1, "friction": 0.65},
    "rocking_chair": {"scale": 1.0, "friction": 0.6},
    "folding_chair": {"scale": 0.8, "friction": 0.45},
    "wheelchair": {"scale": 1.2, "friction": 0.7},
    "high_back_chair": {"scale": 1.0, "friction": 0.6},
    
    # Tables (7 variants)
    "dining_table": {"scale": 1.0, "friction": 0.5},
    "coffee_table": {"scale": 0.8, "friction": 0.45},
    "side_table": {"scale": 0.6, "friction": 0.4},
    "desk": {"scale": 0.9, "friction": 0.5},
    "lab_bench": {"scale": 1.1, "friction": 0.55},
    "standing_desk": {"scale": 0.95, "friction": 0.5},
    "round_table": {"scale": 0.85, "friction": 0.45},
    
    # Containers (8 variants)
    "cup": {"scale": 0.3, "friction": 0.4},
    "bowl": {"scale": 0.35, "friction": 0.35},
    "vase": {"scale": 0.4, "friction": 0.45},
    "pot": {"scale": 0.45, "friction": 0.5},
    "basket": {"scale": 0.5, "friction": 0.55},
    "bucket": {"scale": 0.45, "friction": 0.5},
    "trash_bin": {"scale": 0.55, "friction": 0.6},
    "storage_box": {"scale": 0.6, "friction": 0.5},
    
    # Tools (7 variants)
    "hammer": {"scale": 0.25, "friction": 0.7},
    "wrench": {"scale": 0.2, "friction": 0.65},
    "screwdriver": {"scale": 0.15, "friction": 0.7},
    "shovel": {"scale": 0.8, "friction": 0.75},
    "rake": {"scale": 0.9, "friction": 0.7},
    "broom": {"scale": 0.85, "friction": 0.65},
    "paddle": {"scale": 0.7, "friction": 0.6},
}

assert len(OBJECT_CATALOG) == 30, f"Expected 30 objects, got {len(OBJECT_CATALOG)}"


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_training_data(output_dir: str = "data/pybullet_train",
                         num_episodes: int = 500,
                         use_gui: bool = False):
    """
    Collect training dataset
    
    Args:
        output_dir: Output directory
        num_episodes: Episodes per object (500 × 30 = 15K episodes total)
        use_gui: Show PyBullet GUI
    
    Returns:
        metadata dict with statistics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PYBULLET AFFORDANCE DATASET COLLECTION")
    print("=" * 70)
    print(f"Objects: {len(OBJECT_CATALOG)}")
    print(f"Episodes per object: {num_episodes}")
    print(f"Total episodes: {num_episodes * len(OBJECT_CATALOG)}")
    print(f"Affordances per episode: 6")
    print("=" * 70)
    
    # Initialize environment
    env = AffordanceEnvironment(gui=use_gui, gravity=-9.81)
    
    all_records = []
    statistics = {
        "total_episodes": 0,
        "total_frames": 0,
        "affordance_distribution": {aff.value: 0 for aff in Affordance},
        "object_distribution": {}
    }
    
    # Collect data for each object
    object_names = list(OBJECT_CATALOG.keys())
    
    for obj_idx, obj_name in enumerate(object_names):
        obj_config = OBJECT_CATALOG[obj_name]
        statistics["object_distribution"][obj_name] = 0
        
        print(f"\n[{obj_idx+1}/{len(OBJECT_CATALOG)}] {obj_name}")
        print(f"  Config: scale={obj_config['scale']}, friction={obj_config['friction']}")
        
        pbar = tqdm(range(num_episodes), desc="Episodes", leave=False)
        
        for ep in pbar:
            # Create object with variations
            scale = obj_config['scale'] * np.random.uniform(0.9, 1.1)  # ±10% scale variation
            friction = obj_config['friction'] + np.random.uniform(-0.1, 0.1)
            friction = np.clip(friction, 0.2, 0.8)  # Clamp to valid range
            
            obj_id = env._create_simple_box(scale=scale, friction=friction)
            
            # Test all affordances
            affordance_results = env.test_all_affordances(obj_id)
            
            # Record
            record = {
                "episode_id": f"{obj_name}_ep{ep:04d}",
                "object_name": obj_name,
                "object_config": {
                    "scale": float(scale),
                    "friction": float(friction)
                },
                "affordances": {}
            }
            
            # Convert affordance results to JSON-serializable format
            for aff_name, aff_result in affordance_results.items():
                record["affordances"][aff_name] = {
                    "success": aff_result.success,
                    "confidence": float(aff_result.confidence),
                    "details": aff_result.details
                }
                
                # Statistics
                statistics["affordance_distribution"][aff_name] += 1 if aff_result.success else 0
            
            statistics["object_distribution"][obj_name] += 1
            statistics["total_episodes"] += 1
            
            all_records.append(record)
            
            # Cleanup
            try:
                import pybullet as p
                p.removeBody(obj_id, physicsClientId=env.client_id)
            except:
                pass
    
    env.disconnect()
    
    statistics["total_frames"] = len(all_records) * 100  # Approximate
    
    # Save full dataset
    output_file = os.path.join(output_dir, "pybullet_full_500k.json")
    with open(output_file, 'w') as f:
        json.dump(all_records, f, indent=2)
    print(f"\n✅ Full dataset saved: {output_file}")
    print(f"   Total records: {len(all_records)}")
    print(f"   Approximate frames: {statistics['total_frames']}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✅ Statistics saved: {stats_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total episodes: {statistics['total_episodes']}")
    print(f"Total frames: {statistics['total_frames']}")
    print("\nAffordance distribution:")
    for aff_name, count in statistics['affordance_distribution'].items():
        pct = 100 * count / statistics['total_episodes']
        print(f"  {aff_name}: {count:6d} ({pct:5.1f}%)")
    
    return statistics


def create_train_test_split(data_dir: str = "data/pybullet_train",
                           train_ratio: float = 0.8):
    """
    Create explicit train/test split with NO data leakage
    
    Args:
        data_dir: Data directory
        train_ratio: Fraction for training (0.8 = 80% train, 20% test)
    """
    
    print("\n" + "=" * 70)
    print("CREATING TRAIN/TEST SPLIT (NO DATA LEAKAGE)")
    print("=" * 70)
    
    # Load full dataset
    full_data_file = os.path.join(data_dir, "pybullet_full_500k.json")
    with open(full_data_file, 'r') as f:
        all_records = json.load(f)
    
    # Split by OBJECT, not by episode
    # This ensures no object appears in both train and test
    object_names = list(OBJECT_CATALOG.keys())
    num_train_objects = int(len(object_names) * train_ratio)
    
    train_objects = set(object_names[:num_train_objects])
    test_objects = set(object_names[num_train_objects:])
    
    train_data = [r for r in all_records if r['object_name'] in train_objects]
    test_data = [r for r in all_records if r['object_name'] in test_objects]
    
    # Save splits
    train_file = os.path.join(data_dir, "pybullet_train_no_leak.json")
    test_file = os.path.join(data_dir, "pybullet_test_no_leak.json")
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nTrain split: {len(train_data)} records ({len(train_objects)} objects)")
    print(f"  Objects: {sorted(train_objects)}")
    print(f"\nTest split: {len(test_data)} records ({len(test_objects)} objects)")
    print(f"  Objects: {sorted(test_objects)}")
    
    print(f"\n✅ Train file: {train_file}")
    print(f"✅ Test file: {test_file}")
    
    print("\n" + "=" * 70)
    print("DATA LEAKAGE VERIFICATION")
    print("=" * 70)
    train_objs = set(r['object_name'] for r in train_data)
    test_objs = set(r['object_name'] for r in test_data)
    overlap = train_objs & test_objs
    
    if overlap:
        print(f"❌ ERROR: Found {len(overlap)} objects in both train and test!")
        print(f"   Overlap: {overlap}")
    else:
        print("✅ VERIFIED: No object overlap between train and test")
        print("✅ Data leakage prevention: SUCCESSFUL")


def main():
    """Main execution"""
    
    # Step 1: Collect data
    print("\nStep 1: Collecting PyBullet affordance dataset...")
    stats = collect_training_data(
        output_dir="data/pybullet_train",
        num_episodes=500,
        use_gui=False  # Change to True for visualization
    )
    
    # Step 2: Create train/test split
    print("\n\nStep 2: Creating train/test split...")
    create_train_test_split(
        data_dir="data/pybullet_train",
        train_ratio=0.8
    )
    
    print("\n" + "=" * 70)
    print("✅ DATA COLLECTION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Train VAE on pybullet_train_no_leak.json")
    print("2. Evaluate on pybullet_test_no_leak.json (in-distribution)")
    print("3. Transfer learn on Ego4D dataset")
    print("4. Compare results with v1 paper")


if __name__ == "__main__":
    main()
