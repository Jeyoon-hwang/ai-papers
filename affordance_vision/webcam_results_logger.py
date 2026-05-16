#!/usr/bin/env python3
"""
Results Logger for Affordance Vision System
각 프레임의 affordance 기록, JSON 저장, 통계 생성
"""

import json
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime


class AffordanceResultsLogger:
    """Log and analyze affordance vision results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.json_files = list(self.results_dir.glob('result_*.json'))
        self.results = []
        self.stats = {}
    
    def load_results(self) -> bool:
        """Load all results from JSON files"""
        if not self.json_files:
            print(f"❌ No result files found in {self.results_dir}")
            return False
        
        print(f"📂 Loading {len(self.json_files)} result files...")
        
        for filepath in sorted(self.json_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results.append(data)
            except Exception as e:
                print(f"⚠️  Failed to load {filepath}: {e}")
        
        print(f"✅ Loaded {len(self.results)} results")
        return len(self.results) > 0
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not self.results:
            print("❌ No results loaded")
            return {}
        
        stats = {
            'metadata': {
                'total_frames': len(self.results),
                'timestamp': datetime.now().isoformat(),
            },
            'latency_stats': self._calc_latency_stats(),
            'affordance_stats': self._calc_affordance_stats(),
            'confidence_stats': self._calc_confidence_stats(),
            'temporal_trends': self._calc_temporal_trends(),
        }
        
        self.stats = stats
        return stats
    
    def _calc_latency_stats(self) -> Dict[str, Any]:
        """Calculate latency statistics"""
        total_latencies = [r['latencies']['total'] for r in self.results]
        vision_latencies = [r['latencies']['vision'] for r in self.results]
        reasoning_latencies = [r['latencies']['reasoning'] for r in self.results]
        gemma_latencies = [r['latencies']['gemma'] for r in self.results]
        
        def stats_dict(values):
            if not values:
                return {}
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': sorted(values)[int(len(values) * 0.95)],
                'p99': sorted(values)[int(len(values) * 0.99)],
            }
        
        return {
            'vision_ms': stats_dict(vision_latencies),
            'reasoning_ms': stats_dict(reasoning_latencies),
            'gemma_ms': stats_dict(gemma_latencies),
            'total_ms': stats_dict(total_latencies),
        }
    
    def _calc_affordance_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate affordance statistics"""
        affordance_keys = ['graspable', 'stackable', 'insertable', 'placeable', 'moveable', 'fragile']
        stats = {}
        
        for key in affordance_keys:
            scores = [r['affordances'][key] for r in self.results]
            stats[key] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores),
                'stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
            }
        
        return stats
    
    def _calc_confidence_stats(self) -> Dict[str, float]:
        """Calculate average confidence per affordance type"""
        stats = {}
        
        for key in ['graspable', 'stackable', 'insertable', 'placeable', 'moveable', 'fragile']:
            scores = [r['affordances'][key] for r in self.results]
            avg = statistics.mean(scores)
            
            # Categorize confidence
            if avg >= 0.8:
                confidence = 'High'
            elif avg >= 0.5:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            stats[key] = {
                'average': avg,
                'confidence_level': confidence,
            }
        
        return stats
    
    def _calc_temporal_trends(self) -> Dict[str, List[float]]:
        """Calculate temporal trends (moving averages)"""
        window_size = 30  # 30-frame moving average
        
        trends = {
            'graspable': [],
            'stackable': [],
            'insertable': [],
            'placeable': [],
            'moveable': [],
            'fragile': [],
        }
        
        for key in trends.keys():
            values = [r['affordances'][key] for r in self.results]
            
            for i in range(len(values)):
                window_start = max(0, i - window_size // 2)
                window_end = min(len(values), i + window_size // 2)
                window = values[window_start:window_end]
                trends[key].append(statistics.mean(window))
        
        return trends
    
    def export_csv(self, filename: str = 'affordance_results.csv') -> Path:
        """Export results to CSV"""
        csv_path = self.results_dir / filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if not self.results:
                return csv_path
            
            # Get all keys from first result
            fieldnames = ['frame', 'vision_ms', 'reasoning_ms', 'gemma_ms', 'total_ms',
                         'graspable', 'stackable', 'insertable', 'placeable', 'moveable', 'fragile']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'frame': result['frame'],
                    'vision_ms': f"{result['latencies']['vision']:.2f}",
                    'reasoning_ms': f"{result['latencies']['reasoning']:.2f}",
                    'gemma_ms': f"{result['latencies']['gemma']:.2f}",
                    'total_ms': f"{result['latencies']['total']:.2f}",
                }
                
                for key in ['graspable', 'stackable', 'insertable', 'placeable', 'moveable', 'fragile']:
                    row[key] = f"{result['affordances'][key]:.3f}"
                
                writer.writerow(row)
        
        print(f"✅ CSV exported to: {csv_path}")
        return csv_path
    
    def export_json(self, filename: str = 'affordance_statistics.json') -> Path:
        """Export statistics to JSON"""
        json_path = self.results_dir / filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Statistics exported to: {json_path}")
        return json_path
    
    def print_summary(self):
        """Print summary report"""
        if not self.stats:
            print("❌ No statistics available. Run calculate_statistics() first.")
            return
        
        meta = self.stats['metadata']
        latency = self.stats['latency_stats']
        aff = self.stats['affordance_stats']
        conf = self.stats['confidence_stats']
        
        print("\n" + "=" * 70)
        print("📊 AFFORDANCE VISION RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\n📈 Overview:")
        print(f"  Total frames analyzed: {meta['total_frames']}")
        print(f"  Generated at: {meta['timestamp']}")
        
        print(f"\n⏱️  Latency Statistics (milliseconds):")
        for component in ['vision_ms', 'reasoning_ms', 'gemma_ms', 'total_ms']:
            stats = latency[component]
            comp_name = component.replace('_ms', '').capitalize()
            print(f"  {comp_name}:")
            print(f"    Mean: {stats['mean']:.2f}ms | Median: {stats['median']:.2f}ms")
            print(f"    Min: {stats['min']:.2f}ms | Max: {stats['max']:.2f}ms | StDev: {stats['stdev']:.2f}ms")
            print(f"    P95: {stats['p95']:.2f}ms | P99: {stats['p99']:.2f}ms")
        
        print(f"\n🎯 Affordance Scores:")
        for key, values in aff.items():
            print(f"  {key.capitalize():15} → Mean: {values['mean']:.1%} | "
                  f"Range: {values['min']:.1%} ~ {values['max']:.1%} | "
                  f"StDev: {values['stdev']:.3f}")
        
        print(f"\n🔍 Confidence Levels:")
        for key, conf_data in conf.items():
            print(f"  {key.capitalize():15} → {conf_data['confidence_level']:6} "
                  f"(avg: {conf_data['average']:.1%})")
        
        print("\n" + "=" * 70 + "\n")
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report"""
        if not self.stats:
            return "No statistics available"
        
        meta = self.stats['metadata']
        latency = self.stats['latency_stats']
        aff = self.stats['affordance_stats']
        conf = self.stats['confidence_stats']
        
        md = f"""# Affordance Vision System - Results Analysis Report

**Generated:** {meta['timestamp']}

## Summary

- **Total Frames Analyzed:** {meta['total_frames']}
- **Analysis Scope:** Complete session results

## Latency Analysis

### Response Times (milliseconds)

| Component | Mean | Median | Min | Max | P95 | P99 |
|-----------|------|--------|-----|-----|-----|-----|
"""
        
        for component in ['vision_ms', 'reasoning_ms', 'gemma_ms', 'total_ms']:
            stats = latency[component]
            comp_name = component.replace('_ms', '').capitalize()
            md += f"| {comp_name} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['p95']:.2f} | {stats['p99']:.2f} |\n"
        
        md += """
## Affordance Type Analysis

| Affordance | Mean Score | Min | Max | StDev | Confidence |
|------------|------------|-----|-----|-------|------------|
"""
        
        for key, values in aff.items():
            conf_level = conf[key]['confidence_level']
            md += f"| {key.capitalize()} | {values['mean']:.1%} | {values['min']:.1%} | {values['max']:.1%} | {values['stdev']:.3f} | {conf_level} |\n"
        
        md += """
## Interpretation Guide

### Latency Targets
- Vision: 45ms (image processing)
- Reasoning: 300ms (logical analysis)
- Gemma: 500ms (text generation)
- **Total Target:** ≤850ms for real-time capability

### Affordance Confidence Levels
- **High (≥80%):** Reliable for autonomous decision-making
- **Medium (50-80%):** Requires human verification
- **Low (<50%):** Not suitable for critical tasks

## Recommendations

"""
        
        total_stats = latency['total_ms']
        if total_stats['mean'] > 850:
            md += "⚠️ **Latency exceeds target.** Consider:\n"
            md += "  - Reducing frame resolution\n"
            md += "  - Optimizing model inference\n"
            md += "  - Using lower-precision models\n\n"
        
        high_conf = sum(1 for c in conf.values() if c['confidence_level'] == 'High')
        if high_conf >= 4:
            md += "✅ **High confidence achieved** on most affordance types.\n\n"
        else:
            md += "⚠️ **Low confidence** on some affordance types. Consider:\n"
            md += "  - Collecting more training data\n"
            md += "  - Improving lighting conditions\n"
            md += "  - Fine-tuning model parameters\n\n"
        
        return md
    
    def save_markdown_report(self, filename: str = 'RESULTS.md') -> Path:
        """Save markdown report"""
        md_path = self.results_dir / filename
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        print(f"✅ Markdown report saved to: {md_path}")
        return md_path


def analyze_results_directory(results_dir: str):
    """Convenience function to analyze a results directory"""
    logger = AffordanceResultsLogger(results_dir)
    
    if logger.load_results():
        stats = logger.calculate_statistics()
        logger.print_summary()
        logger.export_csv()
        logger.export_json()
        logger.save_markdown_report()
        
        return logger, stats
    
    return None, None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Affordance Vision Results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--csv', action='store_true', help='Export to CSV')
    parser.add_argument('--json', action='store_true', help='Export to JSON')
    parser.add_argument('--markdown', action='store_true', help='Export to Markdown')
    parser.add_argument('--all', action='store_true', help='Export all formats')
    
    args = parser.parse_args()
    
    logger = AffordanceResultsLogger(args.results_dir)
    
    if logger.load_results():
        stats = logger.calculate_statistics()
        logger.print_summary()
        
        export_csv = args.csv or args.all
        export_json = args.json or args.all
        export_md = args.markdown or args.all
        
        if not any([export_csv, export_json, export_md]):
            export_csv = export_json = export_md = True
        
        if export_csv:
            logger.export_csv()
        if export_json:
            logger.export_json()
        if export_md:
            logger.save_markdown_report()
