"""
Advanced Model Monitoring System
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class UltraAdvancedModelMonitor:
    """
    Ultra Advanced Model Monitoring with comprehensive drift detection
    """
    
    def __init__(self, reference_data, reference_labels):
        self.reference_data = reference_data
        self.reference_labels = reference_labels
        self.drift_threshold = 0.05
        self.performance_threshold = 0.1
        self.monitoring_history = []
        
    def detect_data_drift(self, new_data, feature_names):
        """Comprehensive data drift detection"""
        drift_results = {}
        
        for i, feature in enumerate(feature_names):
            if i < new_data.shape[1] and i < self.reference_data.shape[1]:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p_value = stats.ks_2samp(
                    self.reference_data[:, i], new_data[:, i]
                )
                
                # Population Stability Index (PSI)
                psi_score = self._calculate_psi(
                    self.reference_data[:, i], new_data[:, i]
                )
                
                drift_results[feature] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'psi_score': psi_score,
                    'drift_detected': (ks_p_value < self.drift_threshold or psi_score > 0.25),
                    'drift_severity': self._assess_drift_severity(ks_p_value, psi_score)
                }
        
        return drift_results
    
    def _calculate_psi(self, reference, new_data, bins=10):
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference, bins=bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            new_counts, _ = np.histogram(new_data, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            new_props = new_counts / len(new_data)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.001, ref_props)
            new_props = np.where(new_props == 0, 0.001, new_props)
            
            # Calculate PSI
            psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
            return psi
        except:
            return 0
    
    def _assess_drift_severity(self, ks_p_value, psi_score):
        """Assess drift severity level"""
        if ks_p_value < 0.001 or psi_score > 0.5:
            return 'SEVERE'
        elif ks_p_value < 0.01 or psi_score > 0.25:
            return 'MODERATE'
        elif ks_p_value < 0.05 or psi_score > 0.1:
            return 'MILD'
        else:
            return 'NONE'
    
    def monitor_performance(self, model, new_data, new_labels):
        """Advanced performance monitoring"""
        try:
            # Get predictions
            if hasattr(model, 'predict_ultra_advanced'):
                predictions = model.predict_ultra_advanced(new_data)
            elif hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(new_data)[:, 1]
            else:
                predictions = model.predict(new_data)
            
            # Calculate current performance
            current_metrics = self._calculate_comprehensive_metrics(new_labels, predictions)
            
            # Calculate reference performance
            if hasattr(model, 'predict_ultra_advanced'):
                ref_predictions = model.predict_ultra_advanced(self.reference_data)
            elif hasattr(model, 'predict_proba'):
                ref_predictions = model.predict_proba(self.reference_data)[:, 1]
            else:
                ref_predictions = model.predict(self.reference_data)
            
            reference_metrics = self._calculate_comprehensive_metrics(self.reference_labels, ref_predictions)
            
            # Calculate degradation
            degradation = {}
            alerts = []
            
            for metric in current_metrics:
                degradation[f"{metric}_degradation"] = reference_metrics[metric] - current_metrics[metric]
                
                if degradation[f"{metric}_degradation"] > self.performance_threshold:
                    alerts.append(f"ALERT: {metric} degraded by {degradation[f'{metric}_degradation']:.3f}")
            
            return {
                'current_metrics': current_metrics,
                'reference_metrics': reference_metrics,
                'degradation': degradation,
                'performance_alerts': alerts,
                'overall_health': 'GOOD' if len(alerts) == 0 else 'DEGRADED'
            }
            
        except Exception as e:
            return {'error': f"Performance monitoring failed: {e}"}
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred_proba):
        """Calculate comprehensive performance metrics"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
            }
        except Exception as e:
            # Fallback metrics
            metrics = {
                'accuracy': np.mean(y_pred == y_true),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.5
            }
        
        return metrics
    
    def generate_monitoring_report(self, model, new_data, new_labels, feature_names):
        """Generate comprehensive monitoring report"""
        report_timestamp = datetime.now()
        
        # Data drift analysis
        drift_results = self.detect_data_drift(new_data, feature_names)
        
        # Performance monitoring
        performance_results = self.monitor_performance(model, new_data, new_labels)
        
        # Summary statistics
        total_features = len(drift_results)
        drifted_features = sum(1 for result in drift_results.values() if result['drift_detected'])
        severe_drift_features = sum(1 for result in drift_results.values() 
                                  if result['drift_severity'] == 'SEVERE')
        
        # Generate comprehensive report
        report = {
            'timestamp': report_timestamp.isoformat(),
            'monitoring_summary': {
                'total_features': total_features,
                'drifted_features': drifted_features,
                'severe_drift_features': severe_drift_features,
                'drift_percentage': (drifted_features / total_features) * 100 if total_features > 0 else 0,
                'overall_status': self._determine_overall_status(drifted_features, total_features, performance_results)
            },
            'data_drift_analysis': {
                'summary': {
                    'features_with_drift': drifted_features,
                    'drift_percentage': (drifted_features / total_features) * 100 if total_features > 0 else 0,
                    'most_severe_drifts': self._get_most_severe_drifts(drift_results, top_n=5)
                },
                'detailed_results': drift_results
            },
            'performance_monitoring': performance_results,
            'recommendations': self._generate_recommendations(drift_results, performance_results),
            'next_monitoring_date': (report_timestamp + timedelta(days=7)).isoformat()
        }
        
        # Store in monitoring history
        self.monitoring_history.append(report)
        
        return report
    
    def _determine_overall_status(self, drifted_features, total_features, performance_results):
        """Determine overall monitoring status"""
        drift_percentage = (drifted_features / total_features) * 100 if total_features > 0 else 0
        performance_alerts = len(performance_results.get('performance_alerts', []))
        
        if drift_percentage > 30 or performance_alerts > 2:
            return 'CRITICAL'
        elif drift_percentage > 15 or performance_alerts > 0:
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def _get_most_severe_drifts(self, drift_results, top_n=5):
        """Get most severe drift features"""
        severity_map = {'SEVERE': 4, 'MODERATE': 3, 'MILD': 2, 'NONE': 1}
        
        sorted_drifts = sorted(
            drift_results.items(),
            key=lambda x: (severity_map.get(x[1]['drift_severity'], 0), x[1]['psi_score']),
            reverse=True
        )
        
        return [(feature, data['drift_severity'], data['psi_score']) 
                for feature, data in sorted_drifts[:top_n]]
    
    def _generate_recommendations(self, drift_results, performance_results):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Drift-based recommendations
        severe_drifts = [name for name, data in drift_results.items() 
                        if data['drift_severity'] == 'SEVERE']
        
        if len(severe_drifts) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'DATA_DRIFT',
                'message': f"Severe drift detected in {len(severe_drifts)} features: {severe_drifts[:3]}",
                'action': 'Consider retraining the model with recent data'
            })
        
        # Performance-based recommendations
        performance_alerts = performance_results.get('performance_alerts', [])
        if len(performance_alerts) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'PERFORMANCE_DEGRADATION',
                'message': f"Performance degradation detected in {len(performance_alerts)} metrics",
                'action': 'Immediate model retraining recommended'
            })
        
        return recommendations
