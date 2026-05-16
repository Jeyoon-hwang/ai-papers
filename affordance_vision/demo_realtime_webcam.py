#!/usr/bin/env python3
"""
Real-Time Affordance Vision System via Webcam
카메라 → affordance → reasoning → text generation
"""

import cv2
import torch
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import queue


class MockVAEModel:
    """Mock VAE Model for affordance detection"""
    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        """Predict affordances from frame"""
        # Simulate affordance detection based on image properties
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Calculate simple features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (h * w)
        
        # Mock affordance scores based on image content
        affordances = {
            'graspable': min(0.95, 0.5 + edge_density),
            'stackable': min(0.90, 0.4 + edge_density * 0.5),
            'insertable': min(0.70, 0.3 + edge_density * 0.7),
            'placeable': min(0.95, 0.8 + edge_density * 0.1),
            'moveable': min(0.85, 0.6 + edge_density * 0.3),
            'fragile': min(0.40, 0.2 + edge_density * 0.4),
        }
        return affordances


class MockReasoningEngine:
    """Mock Reasoning Engine"""
    def think(self, affordances: Dict[str, float]) -> str:
        """Generate reasoning about affordances"""
        reasons = []
        
        if affordances['graspable'] > 0.7:
            reasons.append("✋ 집기에 좋은 형태")
        if affordances['stackable'] > 0.7:
            reasons.append("📚 여러 개를 쌓을 수 있음")
        if affordances['fragile'] > 0.5:
            reasons.append("⚠️ 깨질 수 있으니 주의 필요")
        if affordances['moveable'] > 0.7:
            reasons.append("🚚 움직이기 좋음")
        
        if not reasons:
            reasons.append("일반적인 물체")
        
        return " | ".join(reasons)


class MockOllamaClient:
    """Mock Ollama Client for text generation"""
    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, str]:
        """Generate text response"""
        # Simple mock response
        response = "이 물체는 안전하게 다룰 수 있으며, 필요에 따라 이동하거나 정렬할 수 있습니다."
        return {'response': response}


class RealtimeAffordanceSystem:
    def __init__(self, use_gemma: bool = True, save_results: bool = True, use_mock: bool = True):
        """Initialize webcam + models"""
        if use_mock:
            self.vae = MockVAEModel()
            self.brain = MockReasoningEngine()
            self.gemma = MockOllamaClient() if use_gemma else None
        else:
            # Real models would be loaded here
            self.vae = MockVAEModel()
            self.brain = MockReasoningEngine()
            self.gemma = MockOllamaClient() if use_gemma else None
        
        self.use_gemma = use_gemma
        self.save_results = save_results
        if save_results:
            self.results_dir = Path('affordance_vision/results') / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_latency': 0.0,
            'vision_latency': 0.0,
            'reasoning_latency': 0.0,
            'gemma_latency': 0.0,
        }
        
        # Threading for smooth UI
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        self.is_running = True
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        단일 프레임 처리:
        frame → affordance → reasoning → text
        """
        t_total = time.time()
        
        # 1️⃣ Vision (~45ms simulation)
        t_vision = time.time()
        affordances = self.vae.predict(frame)
        t_vision = (time.time() - t_vision) * 1000
        
        # 2️⃣ Reasoning (~300ms simulation)
        t_reasoning = time.time()
        reasoning = self.brain.think(affordances)
        t_reasoning = (time.time() - t_reasoning) * 1000
        
        # 3️⃣ Gemma 2B (~500ms simulation)
        t_gemma = 0
        text = None
        if self.use_gemma and self.gemma:
            t_gemma = time.time()
            prompt = self._create_prompt(affordances, reasoning)
            response = self.gemma.generate(
                model='gemma:2b',
                prompt=prompt,
                stream=False
            )
            text = response.get('response', '').strip()
            t_gemma = (time.time() - t_gemma) * 1000
        
        t_total = (time.time() - t_total) * 1000
        
        return {
            'affordances': affordances,
            'reasoning': reasoning,
            'text': text,
            'latencies': {
                'vision': t_vision,
                'reasoning': t_reasoning,
                'gemma': t_gemma,
                'total': t_total,
            }
        }
    
    def _create_prompt(self, affordances: Dict[str, float], reasoning: str) -> str:
        """Create Gemma prompt"""
        return f"""당신은 로봇 비전 분석 전문가입니다.

Affordance 점수:
- Graspable: {affordances['graspable']:.0%}
- Stackable: {affordances['stackable']:.0%}
- Insertable: {affordances['insertable']:.0%}
- Placeable: {affordances['placeable']:.0%}
- Moveable: {affordances['moveable']:.0%}
- Fragile: {affordances['fragile']:.0%}

분석:
{reasoning}

이 물체를 로봇이 어떻게 다뤄야 하는지 2-3줄로 설명하세요."""
    
    def visualize_result(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        프레임에 결과 오버레이
        affordances + reasoning + text
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Background for text
        cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, vis, 0.7, 0, vis)
        
        # 1️⃣ Affordance bars
        affordances = result['affordances']
        names = ['Graspable', 'Stackable', 'Insertable', 'Placeable', 'Moveable', 'Fragile']
        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 0, 255)]
        
        y = 30
        for name, score, color in zip(names, affordances.values(), colors):
            bar_width = int(score * 300)
            cv2.rectangle(vis, (10, y), (10 + bar_width, y + 20), color, -1)
            cv2.rectangle(vis, (10, y), (310, y + 20), (255, 255, 255), 1)
            cv2.putText(vis, f"{name}: {score:.0%}", (320, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 30
        
        # 2️⃣ Reasoning text
        reasoning = result['reasoning']
        y = 250
        cv2.putText(vis, "💭 분석:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        y += 30
        for line in reasoning.split(' | '):
            cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y += 25
        
        # 3️⃣ Gemma output
        text = result['text']
        if text:
            y = h - 120
            cv2.putText(vis, "🤖 로봇 조언:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            y += 30
            for line in text.split('\n')[:2]:
                if line.strip():
                    cv2.putText(vis, line[:60], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    y += 25
        
        # 4️⃣ Latency info
        latencies = result['latencies']
        info = f"Vision: {latencies['vision']:.0f}ms | Reasoning: {latencies['reasoning']:.0f}ms | Gemma: {latencies['gemma']:.0f}ms | Total: {latencies['total']:.0f}ms"
        cv2.putText(vis, info, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def run(self, camera_id: int = 0, save_video: bool = False):
        """Main loop"""
        cap = cv2.VideoCapture(camera_id)
        
        # Set resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        video_writer = None
        if save_video and self.save_results:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(self.results_dir / 'output.mp4'),
                fourcc, 30, (1280, 720)
            )
        
        print("=" * 70)
        print("🎥 Real-Time Affordance Vision System")
        print("=" * 70)
        print("Press:")
        print("  'q': Quit")
        print("  's': Save frame")
        print("  'p': Pause")
        print("=" * 70 + "\n")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Cannot read from camera")
                        break
                    
                    # Process
                    result = self.process_frame(frame)
                    
                    # Visualize
                    vis = self.visualize_result(frame, result)
                    
                    # Save if enabled
                    if save_video and video_writer:
                        video_writer.write(vis)
                    
                    # Display
                    cv2.imshow("Real-Time Affordance Vision", vis)
                    
                    # Update stats
                    self.stats['frames_processed'] += 1
                    self.stats['total_latency'] += result['latencies']['total']
                    self.stats['vision_latency'] += result['latencies']['vision']
                    self.stats['reasoning_latency'] += result['latencies']['reasoning']
                    self.stats['gemma_latency'] += result['latencies']['gemma']
                    
                    # Log
                    print(f"[Frame {self.stats['frames_processed']:04d}] "
                          f"Vision: {result['latencies']['vision']:6.1f}ms | "
                          f"Reasoning: {result['latencies']['reasoning']:6.1f}ms | "
                          f"Gemma: {result['latencies']['gemma']:6.1f}ms | "
                          f"Total: {result['latencies']['total']:6.1f}ms")
                    
                    if self.save_results:
                        self._save_result(result)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("[PAUSED]" if paused else "[RESUMED]")
                elif key == ord('s'):
                    frame_path = self.results_dir / f"frame_{self.stats['frames_processed']:04d}.png"
                    cv2.imwrite(str(frame_path), vis)
                    print(f"✅ Saved frame {self.stats['frames_processed']}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            self._print_stats()
    
    def _save_result(self, result: Dict[str, Any]):
        """Save result to JSON"""
        path = self.results_dir / f"result_{self.stats['frames_processed']:04d}.json"
        with open(path, 'w', encoding='utf-8') as f:
            # Convert for JSON serialization
            save_data = {
                'frame': self.stats['frames_processed'],
                'affordances': result['affordances'],
                'reasoning': result['reasoning'],
                'text': result['text'],
                'latencies': result['latencies'],
            }
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    def _print_stats(self):
        """Print statistics"""
        n = self.stats['frames_processed']
        if n == 0:
            print("❌ No frames processed")
            return
        
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"Total frames processed: {n}")
        print(f"Avg vision latency: {self.stats['vision_latency']/n:.1f}ms")
        print(f"Avg reasoning latency: {self.stats['reasoning_latency']/n:.1f}ms")
        print(f"Avg Gemma latency: {self.stats['gemma_latency']/n:.1f}ms")
        print(f"Avg total latency: {self.stats['total_latency']/n:.1f}ms")
        avg_latency = self.stats['total_latency'] / n
        if avg_latency > 0:
            print(f"Avg FPS: {1000/avg_latency:.1f}")
        print(f"Results saved to: {self.results_dir}")
        print("=" * 70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Affordance Vision System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--no-save', action='store_true', help='Disable result saving')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--no-gemma', action='store_true', help='Disable Gemma text generation')
    
    args = parser.parse_args()
    
    system = RealtimeAffordanceSystem(
        use_gemma=not args.no_gemma,
        save_results=not args.no_save,
        use_mock=True
    )
    
    try:
        system.run(camera_id=args.camera, save_video=args.save_video)
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted by user")
