"""
🚀 ULTRA-OPTIMIZATION REAL-TIME MONITORING SYSTEM
=================================================

This script provides comprehensive real-time monitoring of the Revolutionary Framework
fine-tuning process with detailed epoch tracking, performance visualization, and 
optimization progress analytics.
"""

import threading
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import psutil
import os
import queue
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class UltraOptimizationMonitor:
    """
    🌟 ULTRA-OPTIMIZATION REAL-TIME MONITOR
    
    Provides comprehensive monitoring of the Revolutionary Framework fine-tuning
    with real-time visualizations, performance tracking, and system analytics.
    """
    
    def __init__(self):
        """Initialize the ultra-optimization monitor"""
        print("🚀 Initializing Ultra-Optimization Real-Time Monitor...")
        
        self.monitoring_active = False
        self.start_time = None
        self.metrics_queue = queue.Queue()
        self.performance_history = defaultdict(list)
        self.system_metrics = defaultdict(deque)
        self.current_phase = "Initialization"
        self.current_model = "None"
        self.current_trial = 0
        self.total_trials = 0
        self.best_score = 0.0
        self.epoch_progress = {}
        
        # Monitoring configuration
        self.update_interval = 2.0  # seconds
        self.max_history_length = 1000
        self.visualization_enabled = True
        
        # Create monitoring directory
        os.makedirs("monitoring", exist_ok=True)
        
    def start_monitoring(self):
        """Start the real-time monitoring system"""
        print("🔥 Starting Ultra-Optimization Real-Time Monitoring...")
        print("=" * 80)
        
        self.monitoring_active = True
        self.start_time = datetime.now()
        
        # Start monitoring threads
        self.system_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self.display_thread = threading.Thread(target=self._display_real_time_stats, daemon=True)
        
        self.system_thread.start()
        self.display_thread.start()
        
        print("✅ Real-time monitoring system activated!")
        print("📊 System metrics collection started")
        print("🖥️ Real-time display thread started")
        print("=" * 80)
    
    def stop_monitoring(self):
        """Stop the monitoring system and generate summary"""
        print("\n🛑 Stopping Ultra-Optimization Monitor...")
        self.monitoring_active = False
        
        # Wait for threads to finish
        time.sleep(3)
        
        # Generate final summary
        self._generate_monitoring_summary()
        print("✅ Monitoring stopped and summary generated!")
    
    def update_phase(self, phase_name):
        """Update the current optimization phase"""
        self.current_phase = phase_name
        print(f"🚀 Phase Update: {phase_name}")
    
    def start_trial(self, trial_num, total_trials):
        """Start a new optimization trial"""
        self.current_trial = trial_num
        self.total_trials = total_trials
        print(f"🔬 Starting Trial {trial_num}/{total_trials}")
    
    def update_trial_result(self, score):
        """Update the result of the current trial"""
        if score > self.best_score:
            self.best_score = score
            print(f"🏆 New Best Score: {score:.6f}")
    
    def log_breakthrough(self, model_name, score):
        """Log a performance breakthrough"""
        print(f"\n🎉🎉🎉 BREAKTHROUGH DETECTED! 🎉🎉🎉")
        print(f"🚀 Model: {model_name}")
        print(f"📈 Score: {score:.6f}")
        print(f"⚡ Achievement: Ultra-optimization success!")
    
    
    def update_progress(self, phase, model=None, trial=None, total_trials=None, score=None, **kwargs):
        """Update the current optimization progress"""
        self.current_phase = phase
        if model:
            self.current_model = model
        if trial is not None:
            self.current_trial = trial
        if total_trials:
            self.total_trials = total_trials
        if score is not None and score > self.best_score:
            self.best_score = score
        
        # Add to performance history
        timestamp = datetime.now()
        self.performance_history['scores'].append({
            'timestamp': timestamp,
            'phase': phase,
            'model': model or self.current_model,
            'trial': trial or self.current_trial,
            'score': score or 0,
            **kwargs
        })
        
        # Limit history size
        if len(self.performance_history['scores']) > self.max_history_length:
            self.performance_history['scores'] = self.performance_history['scores'][-self.max_history_length:]
    
    def log_epoch_progress(self, model_name, epoch, total_epochs, loss=None, accuracy=None, **metrics):
        """Log detailed epoch progress for models"""
        if model_name not in self.epoch_progress:
            self.epoch_progress[model_name] = []
        
        epoch_data = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'timestamp': datetime.now(),
            'loss': loss,
            'accuracy': accuracy,
            **metrics
        }
        
        self.epoch_progress[model_name].append(epoch_data)
        
        # Real-time epoch display
        progress_pct = (epoch / total_epochs) * 100 if total_epochs > 0 else 0
        print(f"    🔥 {model_name} - Epoch {epoch:4d}/{total_epochs} ({progress_pct:5.1f}%) ", end="")
        if loss is not None:
            print(f"Loss: {loss:.6f} ", end="")
        if accuracy is not None:
            print(f"Acc: {accuracy:.6f} ", end="")
        print()
    
    def _monitor_system_metrics(self):
        """Monitor system resource usage"""
        while self.monitoring_active:
            try:
                # CPU and Memory metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                timestamp = datetime.now()
                
                # Store metrics with size limits
                self.system_metrics['cpu'].append((timestamp, cpu_percent))
                self.system_metrics['memory'].append((timestamp, memory.percent))
                self.system_metrics['memory_used_gb'].append((timestamp, memory.used / (1024**3)))
                
                if disk_io:
                    self.system_metrics['disk_read_mb'].append((timestamp, disk_io.read_bytes / (1024**2)))
                    self.system_metrics['disk_write_mb'].append((timestamp, disk_io.write_bytes / (1024**2)))
                
                # Limit deque sizes
                for metric_deque in self.system_metrics.values():
                    if len(metric_deque) > 500:  # Keep last 500 measurements
                        metric_deque.popleft()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"⚠️ System monitoring error: {e}")
                time.sleep(5)
    
    def _display_real_time_stats(self):
        """Display real-time statistics"""
        while self.monitoring_active:
            try:
                if self.start_time:
                    elapsed = datetime.now() - self.start_time
                    
                    # Clear screen for real-time update (works in most terminals)
                    print("\n" + "="*80)
                    print("📊 ULTRA-OPTIMIZATION REAL-TIME MONITOR")
                    print("="*80)
                    
                    print(f"🕐 Elapsed Time: {str(elapsed).split('.')[0]}")
                    print(f"🚀 Current Phase: {self.current_phase}")
                    print(f"🤖 Current Model: {self.current_model}")
                    
                    if self.total_trials > 0:
                        progress_pct = (self.current_trial / self.total_trials) * 100
                        print(f"📈 Trial Progress: {self.current_trial}/{self.total_trials} ({progress_pct:.1f}%)")
                    
                    print(f"🏆 Best Score: {self.best_score:.6f}")
                    
                    # System metrics
                    if self.system_metrics['cpu']:
                        latest_cpu = self.system_metrics['cpu'][-1][1]
                        latest_memory = self.system_metrics['memory'][-1][1]
                        latest_memory_gb = self.system_metrics['memory_used_gb'][-1][1]
                        
                        print(f"💻 CPU Usage: {latest_cpu:.1f}%")
                        print(f"🧠 Memory Usage: {latest_memory:.1f}% ({latest_memory_gb:.2f} GB)")
                    
                    # Recent performance
                    if self.performance_history['scores']:
                        recent_scores = [s['score'] for s in self.performance_history['scores'][-10:] if s['score'] > 0]
                        if recent_scores:
                            avg_recent = np.mean(recent_scores)
                            print(f"📊 Recent Avg Score: {avg_recent:.6f}")
                    
                    print("="*80)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"⚠️ Display error: {e}")
                time.sleep(10)
    
    def _generate_monitoring_summary(self):
        """Generate comprehensive monitoring summary"""
        print("\n📊 Generating Ultra-Optimization Monitoring Summary...")
        
        end_time = datetime.now()
        total_duration = end_time - self.start_time if self.start_time else timedelta(0)
        
        summary = {
            'monitoring_session': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration.total_seconds(),
                'total_duration_formatted': str(total_duration).split('.')[0]
            },
            'optimization_progress': {
                'final_phase': self.current_phase,
                'final_model': self.current_model,
                'trials_completed': self.current_trial,
                'total_trials': self.total_trials,
                'best_score_achieved': self.best_score,
                'completion_percentage': (self.current_trial / self.total_trials * 100) if self.total_trials > 0 else 0
            },
            'performance_statistics': {},
            'system_resource_usage': {},
            'epoch_details': {}
        }
        
        # Performance statistics
        if self.performance_history['scores']:
            scores = [s['score'] for s in self.performance_history['scores'] if s['score'] > 0]
            if scores:
                summary['performance_statistics'] = {
                    'total_evaluations': len(scores),
                    'best_score': max(scores),
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'score_improvement': max(scores) - scores[0] if len(scores) > 1 else 0,
                    'final_score': scores[-1] if scores else 0
                }
        
        # System resource statistics
        if self.system_metrics['cpu']:
            cpu_values = [val for _, val in self.system_metrics['cpu']]
            memory_values = [val for _, val in self.system_metrics['memory']]
            memory_gb_values = [val for _, val in self.system_metrics['memory_used_gb']]
            
            summary['system_resource_usage'] = {
                'cpu_usage': {
                    'average': np.mean(cpu_values),
                    'maximum': max(cpu_values),
                    'minimum': min(cpu_values)
                },
                'memory_usage': {
                    'average_percent': np.mean(memory_values),
                    'maximum_percent': max(memory_values),
                    'average_gb': np.mean(memory_gb_values),
                    'peak_gb': max(memory_gb_values)
                }
            }
        
        # Epoch details summary
        for model_name, epochs in self.epoch_progress.items():
            if epochs:
                summary['epoch_details'][model_name] = {
                    'total_epochs': len(epochs),
                    'final_epoch': epochs[-1]['epoch'] if epochs else 0,
                    'training_duration_seconds': (epochs[-1]['timestamp'] - epochs[0]['timestamp']).total_seconds() if len(epochs) > 1 else 0
                }
                
                # Add accuracy/loss progression if available
                accuracies = [e['accuracy'] for e in epochs if e['accuracy'] is not None]
                losses = [e['loss'] for e in epochs if e['loss'] is not None]
                
                if accuracies:
                    summary['epoch_details'][model_name]['accuracy_progression'] = {
                        'initial': accuracies[0],
                        'final': accuracies[-1],
                        'best': max(accuracies),
                        'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
                    }
                
                if losses:
                    summary['epoch_details'][model_name]['loss_progression'] = {
                        'initial': losses[0],
                        'final': losses[-1],
                        'best': min(losses),
                        'reduction': losses[0] - losses[-1] if len(losses) > 1 else 0
                    }
        
        # Save summary to file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"monitoring/ultra_optimization_summary_{timestamp_str}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Monitoring summary saved: {summary_file}")
        
        # Display key results
        print("\n🏆 ULTRA-OPTIMIZATION SESSION SUMMARY:")
        print("=" * 60)
        print(f"⏱️ Total Duration: {summary['monitoring_session']['total_duration_formatted']}")
        print(f"🎯 Best Score Achieved: {summary['optimization_progress']['best_score_achieved']:.6f}")
        print(f"📊 Trials Completed: {summary['optimization_progress']['trials_completed']}")
        
        if 'performance_statistics' in summary and summary['performance_statistics']:
            perf_stats = summary['performance_statistics']
            print(f"📈 Score Improvement: +{perf_stats['score_improvement']:.6f}")
            print(f"📊 Average Score: {perf_stats['average_score']:.6f}")
        
        if 'system_resource_usage' in summary and summary['system_resource_usage']:
            sys_stats = summary['system_resource_usage']
            print(f"💻 Peak CPU Usage: {sys_stats['cpu_usage']['maximum']:.1f}%")
            print(f"🧠 Peak Memory Usage: {sys_stats['memory_usage']['peak_gb']:.2f} GB")
        
        print("=" * 60)
        
        return summary

def create_optimization_monitor():
    """Factory function to create and return an optimization monitor"""
    return UltraOptimizationMonitor()

# Example usage for integration with the fine-tuner
if __name__ == "__main__":
    # Demo of the monitoring system
    monitor = UltraOptimizationMonitor()
    monitor.start_monitoring()
    
    # Simulate optimization process
    print("🔄 Simulating optimization process...")
    
    for trial in range(1, 11):
        monitor.update_progress(
            phase="Hyperparameter Optimization",
            model="CatBoost",
            trial=trial,
            total_trials=10,
            score=0.75 + trial * 0.02 + np.random.normal(0, 0.01)
        )
        
        # Simulate epoch training
        for epoch in range(1, 6):
            monitor.log_epoch_progress(
                model_name="CatBoost",
                epoch=epoch,
                total_epochs=5,
                loss=1.0 - epoch * 0.1 + np.random.normal(0, 0.05),
                accuracy=0.7 + epoch * 0.05 + np.random.normal(0, 0.02)
            )
            time.sleep(0.5)
        
        time.sleep(2)
    
    time.sleep(10)  # Let monitoring run for a bit
    monitor.stop_monitoring()
