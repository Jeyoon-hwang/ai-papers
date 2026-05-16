# Isaac Gym Setup Guide

## Installation

### Prerequisites
- **NVIDIA GPU** (RTX 3090, A100, or better)
- **CUDA 11.1+**
- **cuDNN 8.0+**
- **Ubuntu 20.04 LTS** (recommended)

### Step 1: Download Isaac Gym

```bash
# Download from NVIDIA
# https://developer.nvidia.com/isaac-gym
# Sign in with NVIDIA developer account

# Extract
tar -xf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym
```

### Step 2: Install

```bash
cd isaacgym
pip install -e .

# Verify installation
python3 -c "from isaacgym import gymapi; print('Isaac Gym ready!')"
```

### Step 3: Verify GPU

```bash
python3 examples/demo.py
# Should show GPU simulation running
```

## Asset Setup

### Download NVIDIA Assets

```bash
cd isaacgym/assets
# Assets are included in Isaac Gym package
```

### Add Custom Objects (ShapeNet, etc.)

```bash
# Download ShapeNet objects
python3 ../utils/download_shapenet.py --categories chair,table,container,tool

# Convert to MJCF/URDF format
# Isaac Gym can load .obj, .urdf, .mjcf, .gltf
```

## Running Affordance Collection

```bash
cd affordance_vision
python3 isaac_affordance_environment.py

# Expected output:
# GPU: Using CUDA device 0
# Parallel simulations: 256
# Simulation speed: 10000+ Hz
# Data collection: ~500K frames in ~5 minutes
```

## Performance Expected

| Metric | Value |
|--------|-------|
| Simulation speed | 10,000+ Hz |
| Parallel envs | 256-512 |
| Time for 500K frames | ~5-10 minutes |
| GPU memory | ~8-16GB |

## Troubleshooting

### "CUDA out of memory"
- Reduce `num_envs` in config
- Use smaller batch size

### "Physics instability"
- Adjust `dt` (timestep)
- Increase `num_substeps`
- Reduce gravity

### "Asset loading fails"
- Check asset paths
- Verify asset format (.obj, .urdf, etc.)
- Use bundled assets first

## References

- [Isaac Gym Docs](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Isaac Gym Examples](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
