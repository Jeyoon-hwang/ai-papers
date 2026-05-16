#!/usr/bin/env python3
"""
Performance Benchmark for Real-Time Affordance Vision System
응답시간, 신뢰도, FPS 측정
"""

import cv2
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from demo_realtime_webcam import RealtimeAffordanceSystem
import statistics


class AffordanceBenchmark:
    """Benchmark tool for affordance system"""
    
    def __init__(self, num_frames: int = 300):
        self.num_frames = num_frames
        self.system = RealtimeAffordanceSystem(use_gemma=True, save_results=False, use_mock=True)
        self.results = []
        self.benchmark_dir = Path('affordance_vision/results/benchmark')
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    def run_benchmark(self, camera_id: int = 0) -> Dict[str, Any]:
        """Run performance benchmark"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return {}
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 70)
        print("🏁 Affordance Vision System - Performance Benchmark")
        print("=" * 70)
        print(f"Target frames: {self.num_frames}")
        print(f"Camera: {camera_id}")
        print("=" * 70 + "\n")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while frame_count < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Cannot read from camera")
                    break
                
                # Process frame
                result = self.system.process_frame(frame)
                
                # Extract confidence scores
                affordances = result['affordances']
                avg_confidence = np.mean(list(affordances.values()))
                
                # Store result
                self.results.append({
                    'frame': frame_count,
                    'latencies': result['latencies'],
                    'confidence': avg_confidence,
                    'affordances': affordances,
                })
                
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    avg_latency = np.mean([r['latencies']['total'] for r in self.results])
                    print(f"[{frame_count:3d}/{self.num_frames}] FPS: {fps:5.1f} | "
                          f"Avg Latency: {avg_latency:6.1f}ms | "
                          f"Elapsed: {elapsed:6.1f}s")
                
                # Show frame with counter
                vis = frame.copy()
                cv2.putText(vis, f"Frame: {frame_count}/{self.num_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Benchmark Running...", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        return self._calculate_stats(total_time)
    
    def _calculate_stats(self, total_time: float) -> Dict[str, Any]:
        """Calculate benchmark statistics"""
        if not self.results:
            return {}
        
        # Latency stats
        total_latencies = [r['latencies']['total'] for r in self.results]
        vision_latencies = [r['latencies']['vision'] for r in self.results]
        reasoning_latencies = [r['latencies']['reasoning'] for r in self.results]
        gemma_latencies = [r['latencies']['gemma'] for r in self.results]
        
        # Confidence stats
        confidences = [r['confidence'] for r in self.results]
        
        # FPS calculation
        fps = len(self.results) / total_time
        
        stats = {
            'metadata': {
                'total_frames': len(self.results),
                'total_time_sec': total_time,
                'average_fps': fps,
                'benchmark_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'latency_ms': {
                'vision': {
                    'mean': statistics.mean(vision_latencies),
                    'median': statistics.median(vision_latencies),
                    'min': min(vision_latencies),
                    'max': max(vision_latencies),
                    'stdev': statistics.stdev(vision_latencies) if len(vision_latencies) > 1 else 0,
                },
                'reasoning': {
                    'mean': statistics.mean(reasoning_latencies),
                    'median': statistics.median(reasoning_latencies),
                    'min': min(reasoning_latencies),
                    'max': max(reasoning_latencies),
                    'stdev': statistics.stdev(reasoning_latencies) if len(reasoning_latencies) > 1 else 0,
                },
                'gemma': {
                    'mean': statistics.mean(gemma_latencies),
                    'median': statistics.median(gemma_latencies),
                    'min': min(gemma_latencies),
                    'max': max(gemma_latencies),
                    'stdev': statistics.stdev(gemma_latencies) if len(gemma_latencies) > 1 else 0,
                },
                'total': {
                    'mean': statistics.mean(total_latencies),
                    'median': statistics.median(total_latencies),
                    'min': min(total_latencies),
                    'max': max(total_latencies),
                    'stdev': statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0,
                },
            },
            'confidence': {
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'min': min(confidences),
                'max': max(confidences),
            },
            'affordance_scores': self._calculate_affordance_stats(),
        }
        
        return stats
    
    def _calculate_affordance_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each affordance type"""
        affordance_keys = ['graspable', 'stackable', 'insertable', 'placeable', 'moveable', 'fragile']
        stats = {}
        
        for key in affordance_keys:
            scores = [r['affordances'][key] for r in self.results]
            stats[key] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores),
            }
        
        return stats
    
    def print_report(self, stats: Dict[str, Any]):
        """Print formatted benchmark report"""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK REPORT")
        print("=" * 70)
        
        meta = stats['metadata']
        print(f"\n📈 Overall Performance:")
        print(f"  Total frames: {meta['total_frames']}")
        print(f"  Total time: {meta['total_time_sec']:.2f}s")
        print(f"  Average FPS: {meta['average_fps']:.1f}")
        
        latency = stats['latency_ms']
        print(f"\n⏱️  Latency (milliseconds):")
        print(f"  Vision:")
        print(f"    Mean: {latency['vision']['mean']:.2f}ms | Median: {latency['vision']['median']:.2f}ms | "
              f"Min: {latency['vision']['min']:.2f}ms | Max: {latency['vision']['max']:.2f}ms")
        print(f"  Reasoning:")
        print(f"    Mean: {latency['reasoning']['mean']:.2f}ms | Median: {latency['reasoning']['median']:.2f}ms | "
              f"Min: {latency['reasoning']['min']:.2f}ms | Max: {latency['reasoning']['max']:.2f}ms")
        print(f"  Gemma:")
        print(f"    Mean: {latency['gemma']['mean']:.2f}ms | Median: {latency['gemma']['median']:.2f}ms | "
              f"Min: {latency['gemma']['min']:.2f}ms | Max: {latency['gemma']['max']:.2f}ms")
        print(f"  Total:")
        print(f"    Mean: {latency['total']['mean']:.2f}ms | Median: {latency['total']['median']:.2f}ms | "
              f"Min: {latency['total']['min']:.2f}ms | Max: {latency['total']['max']:.2f}ms")
        
        conf = stats['confidence']
        print(f"\n🎯 Affordance Confidence:")
        print(f"  Mean: {conf['mean']:.1%} | Median: {conf['median']:.1%} | "
              f"Min: {conf['min']:.1%} | Max: {conf['max']:.1%}")
        
        aff = stats['affordance_scores']
        print(f"\n🤖 Affordance Type Analysis:")
        for key, values in aff.items():
            print(f"  {key.capitalize():15} → Mean: {values['mean']:.1%} | "
                  f"Range: {values['min']:.1%} ~ {values['max']:.1%}")
        
        print("\n" + "=" * 70 + "\n")
    
    def save_report(self, stats: Dict[str, Any], filename: str = 'benchmark_report.json'):
        """Save benchmark report to JSON"""
        report_path = self.benchmark_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Report saved to: {report_path}")
        
        # Also save as markdown
        md_path = self.benchmark_dir / 'benchmark_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(stats))
        print(f"✅ Markdown report saved to: {md_path}")
    
    def _generate_markdown_report(self, stats: Dict[str, Any]) -> str:
        """Generate markdown formatted report"""
        meta = stats['metadata']
        latency = stats['latency_ms']
        aff = stats['affordance_scores']
        
        md = f"""# Affordance Vision System - Benchmark Report

**Generated:** {meta['benchmark_time']}

## Performance Metrics

### Overall
- **Total Frames:** {meta['total_frames']}
- **Total Time:** {meta['total_time_sec']:.2f}s
- **Average FPS:** {meta['average_fps']:.1f}

### Latency (ms)

| Component | Mean | Median | Min | Max | StDev |
|-----------|------|--------|-----|-----|-------|
| Vision | {latency['vision']['mean']:.2f} | {latency['vision']['median']:.2f} | {latency['vision']['min']:.2f} | {latency['vision']['max']:.2f} | {latency['vision']['stdev']:.2f} |
| Reasoning | {latency['reasoning']['mean']:.2f} | {latency['reasoning']['median']:.2f} | {latency['reasoning']['min']:.2f} | {latency['reasoning']['max']:.2f} | {latency['reasoning']['stdev']:.2f} |
| Gemma | {latency['gemma']['mean']:.2f} | {latency['gemma']['median']:.2f} | {latency['gemma']['min']:.2f} | {latency['gemma']['max']:.2f} | {latency['gemma']['stdev']:.2f} |
| **Total** | **{latency['total']['mean']:.2f}** | **{latency['total']['median']:.2f}** | **{latency['total']['min']:.2f}** | **{latency['total']['max']:.2f}** | **{latency['total']['stdev']:.2f}** |

### Affordance Scores

| Affordance | Mean | Median | Min | Max |
|------------|------|--------|-----|-----|
"""
        
        for key, values in aff.items():
            md += f"| {key.capitalize()} | {values['mean']:.1%} | {values['median']:.1%} | {values['min']:.1%} | {values['max']:.1%} |\n"
        
        md += """
## Analysis

- **FPS Target:** 30 FPS → ~33.3ms per frame
- **Current Performance:** {:.1f}ms per frame
- **Headroom:** {:.1f}ms

""".format(latency['total']['mean'], max(0, 33.3 - latency['total']['mean']))
        
        return md


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Affordance Vision System')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to process (default: 300)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    benchmark = AffordanceBenchmark(num_frames=args.frames)
    stats = benchmark.run_benchmark(camera_id=args.camera)
    
    if stats:
        benchmark.print_report(stats)
        benchmark.save_report(stats)
