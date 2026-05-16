# Sequence Engine: Function-First Learning for Affordance Recognition

**Author:** Jeyoon-hwan  
**Date:** May 2026  
**Status:** ✅ Submitted to JMLR (Manuscript ID: 26-1603)

---

## 📚 Papers

### Main Publication
- **paper_ffal.md** - Complete research paper (6,800+ words)
- **paper_ffal.pdf** - PDF version for publication
- **HTML versions** - Browser-readable formats

### Supporting Materials
- **PAPER_SEQUENCE_ENGINE.md** - Core theoretical framework
- **paper_summary.md** - Research summary and key findings
- **paper_next_steps.md** - Future work and open questions
- **PAPER_LANDSCAPE.md** - Related work and literature review
- **KEY_PAPERS_SUMMARY.md** - Summary of key referenced papers

---

## 🔬 Research Implementation

### Project Structure

```
sequence-engine/
├── papers/              # Academic publications
├── src/                 # Core implementation
│   ├── auto_improvement_loop.py      # Automated improvement pipeline
│   ├── comprehensive_validation.py   # Validation & metrics
│   └── phase10_advanced_memory_system.py # Memory management
├── affordance_vision/   # Vision-based affordance detection
├── auto_improvement/    # Automatic model improvement
├── industrial_robot_affordance/  # Robot deployment code
└── README.md
```

### Key Modules

#### 1. **affordance_vision/** 
- Visual affordance recognition using VAE
- Form-independence validation
- Transfer learning pipelines

#### 2. **auto_improvement/**
- Automated criticism & fix generation
- Iterative refinement loops
- Performance benchmarking

#### 3. **industrial_robot_affordance/**
- UR10e robot integration
- Real-world deployment
- End-to-end reasoning (555ms latency, 30 fps)

#### 4. **src/**
- Core algorithms and utilities
- Validation frameworks
- Memory systems

---

## 🎯 Key Contributions

### C1: Form-Independent Affordances
- Affordances learned independent of visual form
- Rotation invariant (±360°)
- Scale invariant (0.5x–2.0x)
- **Result: 99.51% Form-Independence Score (FIS)**

### C2: Zero-Cost Annotation
- Automatic labeling via agent success/failure signals
- Game-based bootstrapping (500K+ frames)
- **Result: 35,043 affordance annotations with zero manual effort**

### C3: Multi-Phase Transfer
- Synthetic-to-real transfer learning
- Game → Ego4D → Industrial robot
- **Result: 99.99% transfer performance, 88.5% real-world accuracy**

### C4: Sequence Engine Principle
- Unified framework for iterative AI improvement
- Coarse→Fine, t→t+1, Simulation→Reality, Try→Function

---

## 📊 Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Form-Independence Score (FIS) | 99.51% | >90% | ✅ |
| Affordance Coverage | 100% | >90% | ✅ |
| Transfer Performance | 99.99% | >80% | ✅ |
| Real-World Accuracy | 88.5% | >85% | ✅ |
| Latency | 555ms | <600ms | ✅ |
| FPS | 30 | ≥30 | ✅ |

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Jeyoon-hwang/ai-papers.git
cd ai-papers
pip install -r requirements.txt
```

### Training
```bash
python src/auto_improvement_loop.py
```

### Evaluation
```bash
python src/comprehensive_validation.py
```

### Deployment (Robot)
```bash
cd industrial_robot_affordance
python deploy_ur10e.py
```

---

## 📰 Citation

If you use this research, please cite:

```bibtex
@article{hwang2026ffal,
  title={Function-First Learning: Unveiling Universal Affordances Through Form-Independent Representations},
  author={Hwang, Jeyoon},
  journal={Journal of Machine Learning Research},
  year={2026},
  note={Manuscript ID: 26-1603}
}
```

---

## 📖 Reading the Paper

- **Full paper**: [paper_ffal.md](paper_ffal.md) or [paper_ffal.pdf](paper_ffal.pdf)
- **Quick summary**: [paper_summary.md](paper_summary.md)
- **Theoretical details**: [PAPER_SEQUENCE_ENGINE.md](PAPER_SEQUENCE_ENGINE.md)
- **Related work**: [PAPER_LANDSCAPE.md](PAPER_LANDSCAPE.md)

---

## 📋 Submission Status

- ✅ **GitHub**: Public repository
- ✅ **JMLR**: Submitted (Manuscript #26-1603)
- ⏳ **Peer Review**: In progress (3-6 months expected)

---

## 🔗 Links

- **GitHub**: https://github.com/Jeyoon-hwang/ai-papers
- **JMLR Submission**: http://jmlr.csail.mit.edu/manudb/center/
- **Author**: Jeyoon-hwan (Independent Researcher)

---

## 📄 License

Creative Commons Attribution 4.0 International (CC-BY-4.0)

---

*This research represents a shift from form-based to function-based AI perception. By learning affordances independent of visual appearance, we enable embodied agents to understand what objects can do, not just what they look like.* ⚡
