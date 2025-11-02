"""
Ultra Advanced Feature Engineering specifically for your UPI dataset
Enhanced with State-of-the-Art Modern Features
"""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.spatial.distance import cdist
from collections import Counter
import hashlib
import warnings
warnings.filterwarnings('ignore')

class UltraAdvancedFeatureEngineer:
    """
    Ultra Advanced Feature Engineering with State-of-the-Art Modern Features
    Enhanced with cutting-edge fraud detection techniques
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.graph_cache = {}
        self.behavioral_profiles = {}
        self.temporal_windows = [1, 3, 7, 14, 30, 90]
        self.anomaly_detectors = {}
        self.sequence_encoders = {}
        self.embedding_models = {}
        
    def create_adversarial_features(self, df):
        """Create adversarial learning features to detect sophisticated fraud patterns"""
        print("🤖 Creating State-of-the-Art Adversarial Features...")
        df_enhanced = df.copy()
        
        # Adversarial noise resistance features
        np.random.seed(42)
        noise_levels = [0.01, 0.05, 0.1]
        
        for noise_level in noise_levels:
            # Add controlled noise to amount and test stability
            noise = np.random.normal(0, noise_level * df_enhanced['trans_amount'].std(), len(df_enhanced))
            df_enhanced[f'amount_noise_resistance_{int(noise_level*100)}'] = np.abs(
                df_enhanced['trans_amount'] - (df_enhanced['trans_amount'] + noise)
            ) / (df_enhanced['trans_amount'] + 1e-6)
        
        # Gradient-based feature importance simulation
        feature_cols = ['trans_amount', 'age', 'trans_hour']
        for col in feature_cols:
            if col in df_enhanced.columns:
                # Simulate gradient magnitude (proxy for feature importance in adversarial context)
                df_enhanced[f'{col}_gradient_magnitude'] = np.abs(np.gradient(df_enhanced[col]))
                
                # Local Lipschitz constant estimation (measures local sensitivity)
                df_enhanced[f'{col}_lipschitz_estimate'] = np.abs(np.gradient(np.gradient(df_enhanced[col])))
        
        # Adversarial boundary distance (how close to decision boundary)
        if len(df_enhanced) > 1:
            # Simulate distance to adversarial boundary using statistical measures
            for col in ['trans_amount', 'age']:
                if col in df_enhanced.columns:
                    mean_val = df_enhanced[col].mean()
                    std_val = df_enhanced[col].std()
                    df_enhanced[f'{col}_boundary_distance'] = np.abs(df_enhanced[col] - mean_val) / (std_val + 1e-6)
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} adversarial features")
        return df_enhanced
    
    def create_transformer_attention_features(self, df):
        """Create transformer-inspired attention mechanism features"""
        print("🎯 Creating Transformer-Inspired Attention Features...")
        df_enhanced = df.copy()
        
        # Multi-head attention simulation for transaction patterns
        key_features = ['trans_amount', 'trans_hour', 'age']
        
        # Self-attention weights simulation
        for i, feat1 in enumerate(key_features):
            if feat1 in df_enhanced.columns:
                for j, feat2 in enumerate(key_features):
                    if feat2 in df_enhanced.columns and i != j:
                        # Simulate attention weights between features
                        attention_weight = np.exp(df_enhanced[feat1] * df_enhanced[feat2]) / (
                            np.exp(df_enhanced[feat1] * df_enhanced[feat2]).sum() + 1e-6
                        )
                        df_enhanced[f'attention_{feat1}_{feat2}'] = attention_weight
        
        # Positional encoding for temporal sequences
        if 'upi_number' in df_enhanced.columns:
            df_enhanced = df_enhanced.sort_values(['upi_number', 'trans_year', 'trans_month', 'trans_day'])
            df_enhanced['position'] = df_enhanced.groupby('upi_number').cumcount()
            
            # Sinusoidal positional encoding
            max_pos = df_enhanced['position'].max() if df_enhanced['position'].max() > 0 else 1
            df_enhanced['pos_encoding_sin'] = np.sin(df_enhanced['position'] / (10000 ** (0/64)))
            df_enhanced['pos_encoding_cos'] = np.cos(df_enhanced['position'] / (10000 ** (0/64)))
        else:
            df_enhanced['position'] = 0
            df_enhanced['pos_encoding_sin'] = 0
            df_enhanced['pos_encoding_cos'] = 1
        
        # Cross-attention between user behavior and transaction context
        if 'category' in df_enhanced.columns and 'state' in df_enhanced.columns:
            # Simulate cross-attention between category and state
            cat_encoded = LabelEncoder().fit_transform(df_enhanced['category'].astype(str))
            state_encoded = LabelEncoder().fit_transform(df_enhanced['state'].astype(str))
            
            df_enhanced['category_state_attention'] = np.tanh(cat_encoded * state_encoded / 10)
            df_enhanced['category_amount_attention'] = np.tanh(cat_encoded * df_enhanced['trans_amount'] / 1000)
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} transformer attention features")
        return df_enhanced
    
    def create_graph_neural_network_features(self, df):
        """Create Graph Neural Network inspired features"""
        print("🕸️ Creating Advanced Graph Neural Network Features...")
        df_enhanced = df.copy()
        
        # Enhanced graph construction with multiple edge types
        G = nx.MultiDiGraph()  # Directed multigraph for complex relationships
        
        # Add nodes with rich attributes
        if 'upi_number' in df_enhanced.columns:
            upi_nodes = df_enhanced['upi_number'].unique()
            for upi in upi_nodes:
                user_data = df_enhanced[df_enhanced['upi_number'] == upi]
                G.add_node(f"upi_{upi}", 
                          node_type='user',
                          avg_amount=user_data['trans_amount'].mean(),
                          transaction_count=len(user_data),
                          age=user_data['age'].iloc[0])
        
        # Create heterogeneous graph with multiple node types
        merchants = df_enhanced['category'].unique()
        locations = df_enhanced['state'].unique()
        
        for merchant in merchants:
            G.add_node(f"merchant_{merchant}", node_type='merchant')
        for location in locations:
            G.add_node(f"location_{location}", node_type='location')
        
        # Multi-relational edges
        edge_types = ['transacts_with', 'located_in', 'same_time', 'amount_similar']
        
        for _, row in df_enhanced.iterrows():
            if 'upi_number' in df_enhanced.columns:
                user_node = f"upi_{row['upi_number']}"
            else:
                user_node = "upi_PRED"
                
            merchant_node = f"merchant_{row['category']}"
            location_node = f"location_{row['state']}"
            
            # Add different types of edges
            G.add_edge(user_node, merchant_node, edge_type='transacts_with', weight=row['trans_amount'])
            G.add_edge(user_node, location_node, edge_type='located_in', weight=1)
        
        # Advanced graph metrics
        try:
            # PageRank with personalization
            pagerank = nx.pagerank(G.to_undirected(), weight='weight')
            
            # Katz centrality
            try:
                katz_centrality = nx.katz_centrality(G.to_undirected(), weight='weight', alpha=0.1)
            except:
                katz_centrality = pagerank  # Fallback
            
            # Eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G.to_undirected(), weight='weight')
            except:
                eigenvector_centrality = pagerank  # Fallback
            
            # Local clustering coefficient
            clustering = nx.clustering(G.to_undirected(), weight='weight')
            
            # Map back to dataframe
            if 'upi_number' in df_enhanced.columns:
                df_enhanced['pagerank_centrality'] = df_enhanced['upi_number'].apply(
                    lambda x: pagerank.get(f"upi_{x}", 0)
                )
                df_enhanced['katz_centrality'] = df_enhanced['upi_number'].apply(
                    lambda x: katz_centrality.get(f"upi_{x}", 0)
                )
                df_enhanced['eigenvector_centrality'] = df_enhanced['upi_number'].apply(
                    lambda x: eigenvector_centrality.get(f"upi_{x}", 0)
                )
                df_enhanced['clustering_coefficient'] = df_enhanced['upi_number'].apply(
                    lambda x: clustering.get(f"upi_{x}", 0)
                )
            else:
                df_enhanced['pagerank_centrality'] = pagerank.get("upi_PRED", 0)
                df_enhanced['katz_centrality'] = katz_centrality.get("upi_PRED", 0)
                df_enhanced['eigenvector_centrality'] = eigenvector_centrality.get("upi_PRED", 0)
                df_enhanced['clustering_coefficient'] = clustering.get("upi_PRED", 0)
            
            # Graph neural network message passing simulation
            # Aggregate neighbor features
            neighbor_features = []
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_weights = [G[node][neighbor].get('weight', 1) for neighbor in neighbors]
                    avg_weight = np.mean(neighbor_weights)
                else:
                    avg_weight = 0
                neighbor_features.append(avg_weight)
            
            if 'upi_number' in df_enhanced.columns:
                df_enhanced['neighbor_avg_weight'] = df_enhanced['upi_number'].apply(
                    lambda x: neighbor_features[0] if f"upi_{x}" in G.nodes() else 0
                )
            else:
                df_enhanced['neighbor_avg_weight'] = neighbor_features[0] if neighbor_features else 0
                
        except Exception as e:
            print(f"⚠️ Advanced graph metrics failed: {e}")
            # Fallback values
            for col in ['pagerank_centrality', 'katz_centrality', 'eigenvector_centrality', 
                       'clustering_coefficient', 'neighbor_avg_weight']:
                df_enhanced[col] = 0.1
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} graph neural network features")
        return df_enhanced
        
    def create_ultra_temporal_features(self, df):
        """Create ultra advanced temporal features from your dataset"""
        print("🕐 Creating Ultra Advanced Temporal Features...")
        df_enhanced = df.copy()
        
        # Cyclical encoding for better temporal pattern recognition
        df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['trans_hour'] / 24)
        df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['trans_hour'] / 24)
        df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['trans_day'] / 31)
        df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['trans_day'] / 31)
        df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['trans_month'] / 12)
        df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['trans_month'] / 12)
        
        # Advanced time-based risk patterns
        df_enhanced['is_night'] = ((df_enhanced['trans_hour'] >= 23) | 
                                  (df_enhanced['trans_hour'] <= 5)).astype(int)
        df_enhanced['is_weekend'] = (df_enhanced['trans_day'] % 7 < 2).astype(int)
        df_enhanced['is_business_hour'] = ((df_enhanced['trans_hour'] >= 9) & 
                                          (df_enhanced['trans_hour'] <= 17)).astype(int)
        df_enhanced['is_late_night'] = ((df_enhanced['trans_hour'] >= 2) & 
                                       (df_enhanced['trans_hour'] <= 4)).astype(int)
        
        # Temporal interaction features
        df_enhanced['hour_day_interaction'] = df_enhanced['trans_hour'] * df_enhanced['trans_day']
        df_enhanced['month_day_interaction'] = df_enhanced['trans_month'] * df_enhanced['trans_day']
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} ultra temporal features")
        return df_enhanced
        
    def create_deep_behavioral_embeddings(self, df):
        """Create deep behavioral embeddings using advanced techniques"""
        print("🧠 Creating Deep Behavioral Embeddings...")
        df_enhanced = df.copy()
        
        # Sequential pattern mining for fraud detection
        if 'upi_number' in df_enhanced.columns:
            df_enhanced = df_enhanced.sort_values(['upi_number', 'trans_year', 'trans_month', 'trans_day', 'trans_hour'])
            
            # N-gram analysis of transaction patterns
            for n in [2, 3]:  # Bigrams and trigrams
                df_enhanced[f'category_ngram_{n}'] = df_enhanced.groupby('upi_number')['category'].transform(
                    lambda x: x.astype(str).rolling(window=n, min_periods=1).apply(
                        lambda y: hash(''.join([str(val) for val in y])) % 1000, raw=False
                    )
                )
                
                df_enhanced[f'amount_ngram_{n}'] = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                    lambda x: x.rolling(window=n, min_periods=1).apply(
                        lambda y: np.std(y) if len(y) > 1 else 0, raw=False
                    )
                )
        
        # Behavioral fingerprinting
        behavior_cols = ['trans_hour', 'trans_amount', 'category']
        if all(col in df_enhanced.columns for col in behavior_cols):
            # Create behavioral hash
            df_enhanced['behavioral_hash'] = df_enhanced.apply(
                lambda row: hash(f"{row['trans_hour']}_{row['category']}_{int(row['trans_amount']/100)}") % 10000,
                axis=1
            )
            
            # Behavioral entropy (measure of unpredictability)
            if 'upi_number' in df_enhanced.columns:
                df_enhanced['behavioral_entropy'] = df_enhanced.groupby('upi_number')['behavioral_hash'].transform(
                    lambda x: -np.sum([p * np.log2(p) for p in np.bincount(x) / len(x) if p > 0])
                )
            else:
                df_enhanced['behavioral_entropy'] = 0
        
        # Advanced temporal embeddings
        # Fourier transform features for cyclical patterns
        if len(df_enhanced) > 1:
            # Weekly pattern detection
            df_enhanced['hour_fft_real'] = np.real(np.fft.fft(df_enhanced['trans_hour']))[0] / len(df_enhanced)
            df_enhanced['hour_fft_imag'] = np.imag(np.fft.fft(df_enhanced['trans_hour']))[0] / len(df_enhanced)
            
            # Amount pattern detection
            df_enhanced['amount_fft_real'] = np.real(np.fft.fft(df_enhanced['trans_amount']))[0] / len(df_enhanced)
            df_enhanced['amount_fft_imag'] = np.imag(np.fft.fft(df_enhanced['trans_amount']))[0] / len(df_enhanced)
        else:
            for col in ['hour_fft_real', 'hour_fft_imag', 'amount_fft_real', 'amount_fft_imag']:
                df_enhanced[col] = 0
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} deep behavioral embeddings")
        return df_enhanced
    
    def create_anomaly_detection_features(self, df):
        """Create advanced anomaly detection features"""
        print("🔍 Creating Advanced Anomaly Detection Features...")
        df_enhanced = df.copy()
        
        # Isolation Forest anomaly scores
        numeric_cols = ['trans_amount', 'age', 'trans_hour']
        available_cols = [col for col in numeric_cols if col in df_enhanced.columns]
        
        if len(available_cols) >= 2 and len(df_enhanced) > 1:
            try:
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                X_anomaly = df_enhanced[available_cols].fillna(0)
                anomaly_scores = isolation_forest.fit_predict(X_anomaly)
                df_enhanced['isolation_anomaly_score'] = isolation_forest.score_samples(X_anomaly)
                df_enhanced['is_isolation_anomaly'] = (anomaly_scores == -1).astype(int)
            except:
                df_enhanced['isolation_anomaly_score'] = 0
                df_enhanced['is_isolation_anomaly'] = 0
        else:
            df_enhanced['isolation_anomaly_score'] = 0
            df_enhanced['is_isolation_anomaly'] = 0
        
        # Local Outlier Factor (LOF) simulation
        if len(df_enhanced) > 5:
            for col in available_cols:
                # Calculate local density
                values = df_enhanced[col].values.reshape(-1, 1)
                distances = cdist(values, values)
                k = min(5, len(df_enhanced) - 1)
                
                # K-nearest neighbors distances
                knn_distances = np.partition(distances, k, axis=1)[:, 1:k+1]
                local_density = 1 / (np.mean(knn_distances, axis=1) + 1e-6)
                
                df_enhanced[f'{col}_local_density'] = local_density
                df_enhanced[f'{col}_lof_score'] = local_density / (local_density.mean() + 1e-6)
        
        # Statistical anomaly detection
        for col in available_cols:
            # Z-score based anomaly
            z_scores = np.abs(stats.zscore(df_enhanced[col].fillna(df_enhanced[col].median())))
            df_enhanced[f'{col}_zscore_anomaly'] = (z_scores > 3).astype(int)
            
            # Modified Z-score (using median)
            median = df_enhanced[col].median()
            mad = np.median(np.abs(df_enhanced[col] - median))
            modified_z_scores = 0.6745 * (df_enhanced[col] - median) / (mad + 1e-6)
            df_enhanced[f'{col}_modified_zscore'] = np.abs(modified_z_scores)
        
        # Ensemble anomaly score
        anomaly_cols = [col for col in df_enhanced.columns if 'anomaly' in col or 'lof_score' in col]
        if anomaly_cols:
            df_enhanced['ensemble_anomaly_score'] = df_enhanced[anomaly_cols].mean(axis=1)
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} anomaly detection features")
        return df_enhanced
    
    def create_advanced_clustering_features(self, df):
        """Create advanced clustering-based features"""
        print("🎯 Creating Advanced Clustering Features...")
        df_enhanced = df.copy()
        
        # Multi-level clustering
        numeric_cols = ['trans_amount', 'age', 'trans_hour']
        available_cols = [col for col in numeric_cols if col in df_enhanced.columns]
        
        if len(available_cols) >= 2 and len(df_enhanced) > 3:
            X_cluster = df_enhanced[available_cols].fillna(0)
            
            # K-means clustering with multiple k values
            for k in [3, 5, 8]:
                if len(df_enhanced) > k:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_cluster)
                    df_enhanced[f'kmeans_cluster_{k}'] = cluster_labels
                    
                    # Distance to cluster center
                    distances = kmeans.transform(X_cluster)
                    df_enhanced[f'kmeans_distance_{k}'] = np.min(distances, axis=1)
                    
                    # Silhouette score proxy (intra-cluster distance)
                    centers = kmeans.cluster_centers_
                    df_enhanced[f'intra_cluster_distance_{k}'] = [
                        np.linalg.norm(X_cluster.iloc[i] - centers[cluster_labels[i]])
                        for i in range(len(X_cluster))
                    ]
            
            # DBSCAN clustering for density-based grouping
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=3)
                dbscan_labels = dbscan.fit_predict(StandardScaler().fit_transform(X_cluster))
                df_enhanced['dbscan_cluster'] = dbscan_labels
                df_enhanced['is_dbscan_outlier'] = (dbscan_labels == -1).astype(int)
                
                # Core point indicator
                df_enhanced['is_core_point'] = 0
                for i, label in enumerate(dbscan_labels):
                    if i in dbscan.core_sample_indices_:
                        df_enhanced.iloc[i, df_enhanced.columns.get_loc('is_core_point')] = 1
            except:
                df_enhanced['dbscan_cluster'] = 0
                df_enhanced['is_dbscan_outlier'] = 0
                df_enhanced['is_core_point'] = 0
            
            # Hierarchical clustering features
            try:
                hierarchical = AgglomerativeClustering(n_clusters=5)
                hier_labels = hierarchical.fit_predict(X_cluster)
                df_enhanced['hierarchical_cluster'] = hier_labels
            except:
                df_enhanced['hierarchical_cluster'] = 0
        else:
            # Default values for insufficient data
            for k in [3, 5, 8]:
                df_enhanced[f'kmeans_cluster_{k}'] = 0
                df_enhanced[f'kmeans_distance_{k}'] = 0
                df_enhanced[f'intra_cluster_distance_{k}'] = 0
            df_enhanced['dbscan_cluster'] = 0
            df_enhanced['is_dbscan_outlier'] = 0
            df_enhanced['is_core_point'] = 0
            df_enhanced['hierarchical_cluster'] = 0
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} advanced clustering features")
        return df_enhanced
    
    def create_time_series_features(self, df):
        """Create advanced time series features"""
        print("📈 Creating Advanced Time Series Features...")
        df_enhanced = df.copy()
        
        # Sort by time for time series analysis
        if 'upi_number' in df_enhanced.columns:
            df_enhanced = df_enhanced.sort_values(['upi_number', 'trans_year', 'trans_month', 'trans_day', 'trans_hour'])
        
        # Advanced moving averages
        windows = [3, 7, 14, 30]
        for window in windows:
            if 'upi_number' in df_enhanced.columns:
                # Exponential moving average
                df_enhanced[f'ema_amount_{window}'] = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                    lambda x: x.ewm(span=window).mean()
                )
                
                # Bollinger Bands
                rolling_mean = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                rolling_std = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                
                df_enhanced[f'bollinger_upper_{window}'] = rolling_mean + (2 * rolling_std)
                df_enhanced[f'bollinger_lower_{window}'] = rolling_mean - (2 * rolling_std)
                df_enhanced[f'bollinger_position_{window}'] = (
                    (df_enhanced['trans_amount'] - rolling_mean) / (2 * rolling_std + 1e-6)
                )
                
                # Rate of change
                df_enhanced[f'roc_amount_{window}'] = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                    lambda x: x.pct_change(periods=window).fillna(0)
                )
            else:
                # Single transaction defaults
                df_enhanced[f'ema_amount_{window}'] = df_enhanced['trans_amount']
                df_enhanced[f'bollinger_upper_{window}'] = df_enhanced['trans_amount'] * 1.1
                df_enhanced[f'bollinger_lower_{window}'] = df_enhanced['trans_amount'] * 0.9
                df_enhanced[f'bollinger_position_{window}'] = 0
                df_enhanced[f'roc_amount_{window}'] = 0
        
        # Seasonal decomposition simulation
        if 'upi_number' in df_enhanced.columns and len(df_enhanced) > 12:
            df_enhanced['hour_trend'] = df_enhanced.groupby('upi_number')['trans_hour'].transform(
                lambda x: x.rolling(window=min(12, len(x)), min_periods=1).mean()
            )
            df_enhanced['hour_seasonal'] = df_enhanced['trans_hour'] - df_enhanced['hour_trend']
        else:
            df_enhanced['hour_trend'] = df_enhanced['trans_hour']
            df_enhanced['hour_seasonal'] = 0
        
        # Autocorrelation features
        if 'upi_number' in df_enhanced.columns:
            # Lag-1 autocorrelation simulation
            df_enhanced['amount_lag1_corr'] = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                lambda x: x.corr(x.shift(1)) if len(x) > 2 else 0
            ).fillna(0)
        else:
            df_enhanced['amount_lag1_corr'] = 0
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} time series features")
        return df_enhanced
    
    def create_ultra_amount_features(self, df):
        """Create ultra advanced amount-based features"""
        print("💰 Creating Ultra Advanced Amount Features...")
        df_enhanced = df.copy()
        
        # Advanced amount transformations
        df_enhanced['log_amount'] = np.log1p(df_enhanced['trans_amount'])
        df_enhanced['sqrt_amount'] = np.sqrt(df_enhanced['trans_amount'])
        df_enhanced['amount_percentile'] = df_enhanced['trans_amount'].rank(pct=True)
        
        # Amount pattern recognition
        df_enhanced['is_round_amount'] = (df_enhanced['trans_amount'] % 100 == 0).astype(int)
        df_enhanced['is_small_amount'] = (df_enhanced['trans_amount'] <= 50).astype(int)
        df_enhanced['is_large_amount'] = (df_enhanced['trans_amount'] >= 10000).astype(int)
        df_enhanced['is_suspicious_amount'] = ((df_enhanced['trans_amount'] % 1000 == 0) & 
                                              (df_enhanced['trans_amount'] >= 5000)).astype(int)
        
        # Statistical amount features by category and state
        for group_col in ['category', 'state']:
            if group_col in df_enhanced.columns:
                group_stats = df_enhanced.groupby(group_col)['trans_amount'].agg([
                    'mean', 'std', 'median', 'min', 'max', 'count'
                ]).add_prefix(f'{group_col}_amount_')
                
                df_enhanced = df_enhanced.merge(group_stats, left_on=group_col, right_index=True, how='left')
                
                # Deviation from group statistics
                df_enhanced[f'amount_dev_from_{group_col}_mean'] = (
                    (df_enhanced['trans_amount'] - df_enhanced[f'{group_col}_amount_mean']) / 
                    (df_enhanced[f'{group_col}_amount_std'] + 1e-6)
                )
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} amount features")
        return df_enhanced
    
    def create_ultra_behavioral_features(self, df):
        """Create ultra advanced behavioral features"""
        print("🧠 Creating Ultra Advanced Behavioral Features...")
        df_enhanced = df.copy()
        
        # Check if we have upi_number for grouping (training mode)
        if 'upi_number' in df_enhanced.columns:
            # Sort by UPI number and time for sequence analysis
            df_enhanced = df_enhanced.sort_values(['upi_number', 'trans_year', 'trans_month', 'trans_day', 'trans_hour'])
            
            # Advanced user-level aggregations
            user_agg = df_enhanced.groupby('upi_number').agg({
                'trans_amount': ['count', 'mean', 'std', 'min', 'max', 'sum', 'skew'],
                'category': ['nunique', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]],
                'state': ['nunique'],
                'trans_hour': ['mean', 'std', 'min', 'max'],
                'age': 'first'
            }).round(4)
        else:
            # For prediction mode, create pseudo user behavior based on current transaction
            print("ℹ️  No UPI number provided - using pseudo user for behavioral features")
            user_agg = pd.DataFrame({
                ('trans_amount', 'count'): [1],
                ('trans_amount', 'mean'): [df_enhanced['trans_amount'].iloc[0]],
                ('trans_amount', 'std'): [0.0],
                ('trans_amount', 'min'): [df_enhanced['trans_amount'].iloc[0]],
                ('trans_amount', 'max'): [df_enhanced['trans_amount'].iloc[0]],
                ('trans_amount', 'sum'): [df_enhanced['trans_amount'].iloc[0]],
                ('trans_amount', 'skew'): [0.0],
                ('category', 'nunique'): [1],
                ('category', '<lambda>'): [df_enhanced['category'].iloc[0]],
                ('state', 'nunique'): [1],
                ('trans_hour', 'mean'): [df_enhanced['trans_hour'].iloc[0]],
                ('trans_hour', 'std'): [0.0],
                ('trans_hour', 'min'): [df_enhanced['trans_hour'].iloc[0]],
                ('trans_hour', 'max'): [df_enhanced['trans_hour'].iloc[0]],
                ('age', 'first'): [df_enhanced['age'].iloc[0]]
            }, index=[0])
        
        # Flatten column names
        user_agg.columns = ['_'.join(map(str, col)).strip() for col in user_agg.columns]
        user_agg = user_agg.add_prefix('user_')
        
        if 'upi_number' in df_enhanced.columns:
            df_enhanced = df_enhanced.merge(user_agg, left_on='upi_number', right_index=True, how='left')
        else:
            # For prediction mode, join based on row index
            for col in user_agg.columns:
                df_enhanced[col] = user_agg[col].iloc[0]
        
        # Advanced velocity features
        for window in self.temporal_windows:
            if 'upi_number' in df_enhanced.columns:
                # Transaction frequency
                df_enhanced[f'txn_freq_{window}d'] = df_enhanced.groupby('upi_number').cumcount() + 1
                
                # Amount velocity (simplified for demonstration)
                df_enhanced[f'amount_velocity_{window}d'] = df_enhanced.groupby('upi_number')['trans_amount'].transform(
                    lambda x: x.rolling(window=min(len(x), window), min_periods=1).sum()
                )
            else:
                # For prediction mode, use single transaction values
                df_enhanced[f'txn_freq_{window}d'] = 1
                df_enhanced[f'amount_velocity_{window}d'] = df_enhanced['trans_amount']
            
            # Category diversity
            if 'upi_number' in df_enhanced.columns:
                df_enhanced[f'category_diversity_{window}d'] = df_enhanced.groupby('upi_number')['category'].transform(
                    lambda x: x.rolling(window=min(len(x), window), min_periods=1).apply(lambda y: len(set(y)))
                )
            else:
                df_enhanced[f'category_diversity_{window}d'] = 1
        
        # Inter-transaction patterns
        if 'upi_number' in df_enhanced.columns:
            df_enhanced['time_since_last'] = df_enhanced.groupby('upi_number').apply(
                lambda group: pd.Series(range(len(group)), index=group.index)
            ).reset_index(0, drop=True)
        else:
            df_enhanced['time_since_last'] = 0
        
        # Behavioral anomaly indicators
        df_enhanced['is_amount_outlier'] = (
            np.abs(df_enhanced['trans_amount'] - df_enhanced['user_trans_amount_mean']) > 
            3 * df_enhanced['user_trans_amount_std']
        ).astype(int)
        
        df_enhanced['is_time_outlier'] = (
            np.abs(df_enhanced['trans_hour'] - df_enhanced['user_trans_hour_mean']) > 
            2 * df_enhanced['user_trans_hour_std']
        ).astype(int)
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} behavioral features")
        return df_enhanced
    
    def create_ultra_graph_features(self, df):
        """Create ultra advanced graph-based features"""
        print("🕸️ Creating Ultra Advanced Graph Features...")
        df_enhanced = df.copy()
        
        # Create transaction network graph
        G = nx.Graph()
        
        # Add nodes with attributes
        if 'upi_number' in df_enhanced.columns:
            upi_nodes = df_enhanced['upi_number'].unique()
            # Add UPI nodes
            for upi in upi_nodes:
                G.add_node(f"upi_{upi}", node_type='upi')
        else:
            # For prediction mode, create a pseudo UPI node
            upi_nodes = ['PRED']
            G.add_node(f"upi_PRED", node_type='upi')
        
        state_nodes = df_enhanced['state'].unique()
        category_nodes = df_enhanced['category'].unique()
        
        # Add state and category nodes
        for state in state_nodes:
            G.add_node(f"state_{state}", node_type='state')
        for category in category_nodes:
            G.add_node(f"cat_{category}", node_type='category')
        
        # Create weighted edges based on transaction patterns
        edge_weights = {}
        for _, row in df_enhanced.iterrows():
            if 'upi_number' in df_enhanced.columns:
                upi_node = f"upi_{row['upi_number']}"
            else:
                upi_node = f"upi_PRED"
            state_node = f"state_{row['state']}"
            cat_node = f"cat_{row['category']}"
            
            # UPI-State connections
            edge_key = (upi_node, state_node)
            if edge_key in edge_weights:
                edge_weights[edge_key] += row['trans_amount']
            else:
                edge_weights[edge_key] = row['trans_amount']
            
            # UPI-Category connections
            edge_key = (upi_node, cat_node)
            if edge_key in edge_weights:
                edge_weights[edge_key] += row['trans_amount']
            else:
                edge_weights[edge_key] = row['trans_amount']
        
        # Add weighted edges
        for (node1, node2), weight in edge_weights.items():
            G.add_edge(node1, node2, weight=weight)
        
        # Calculate graph metrics
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
            closeness_centrality = nx.closeness_centrality(G)
            
            # Map centralities back to transactions
            if 'upi_number' in df_enhanced.columns:
                df_enhanced['degree_centrality'] = df_enhanced['upi_number'].apply(
                    lambda x: degree_centrality.get(f"upi_{x}", 0)
                )
                df_enhanced['betweenness_centrality'] = df_enhanced['upi_number'].apply(
                    lambda x: betweenness_centrality.get(f"upi_{x}", 0)
                )
                df_enhanced['closeness_centrality'] = df_enhanced['upi_number'].apply(
                    lambda x: closeness_centrality.get(f"upi_{x}", 0)
                )
            else:
                # For prediction mode, calculate based on current transaction
                pseudo_upi = f"upi_PRED"
                df_enhanced['degree_centrality'] = degree_centrality.get(pseudo_upi, 0)
                df_enhanced['betweenness_centrality'] = betweenness_centrality.get(pseudo_upi, 0)
                df_enhanced['closeness_centrality'] = closeness_centrality.get(pseudo_upi, 0)
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(G)
                community_map = {}
                for i, community in enumerate(communities):
                    for node in community:
                        community_map[node] = i
                
                if 'upi_number' in df_enhanced.columns:
                    df_enhanced['community_id'] = df_enhanced['upi_number'].apply(
                        lambda x: community_map.get(f"upi_{x}", -1)
                    )
                    df_enhanced['community_size'] = df_enhanced['community_id'].map(
                        df_enhanced['community_id'].value_counts()
                    ).fillna(1)
                else:
                    pseudo_upi = f"upi_PRED"
                    df_enhanced['community_id'] = community_map.get(pseudo_upi, 0)
                    df_enhanced['community_size'] = 1
            except:
                df_enhanced['community_id'] = 0
                df_enhanced['community_size'] = 1
                
        except Exception as e:
            print(f"⚠️ Graph metrics calculation failed: {e}")
            # Fallback: create dummy graph features
            df_enhanced['degree_centrality'] = 0.1
            df_enhanced['betweenness_centrality'] = 0.1
            df_enhanced['closeness_centrality'] = 0.1
            df_enhanced['community_id'] = 0
            df_enhanced['community_size'] = 1
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} graph features")
        return df_enhanced
    
    def create_ultra_risk_features(self, df):
        """Create ultra advanced risk assessment features"""
        print("⚠️ Creating Ultra Advanced Risk Features...")
        df_enhanced = df.copy()
        
        # Age-based risk profiling
        df_enhanced['age_risk_score'] = np.select([
            df_enhanced['age'] < 25,
            df_enhanced['age'] > 65,
            (df_enhanced['age'] >= 25) & (df_enhanced['age'] <= 35)
        ], [2.5, 2.0, 1.5], default=1.0)
        
        # Geographic risk assessment
        if 'fraud_risk' in df_enhanced.columns:
            state_fraud_rates = df_enhanced.groupby('state')['fraud_risk'].mean()
            df_enhanced['state_fraud_rate'] = df_enhanced['state'].map(state_fraud_rates).fillna(0.1)
            
            category_fraud_rates = df_enhanced.groupby('category')['fraud_risk'].mean()
            df_enhanced['category_fraud_rate'] = df_enhanced['category'].map(category_fraud_rates).fillna(0.1)
        else:
            df_enhanced['state_fraud_rate'] = 0.1
            df_enhanced['category_fraud_rate'] = 0.1
        
        # Composite risk scoring
        risk_components = [
            df_enhanced['age_risk_score'] * 0.2,
            df_enhanced['state_fraud_rate'] * 0.3,
            df_enhanced['category_fraud_rate'] * 0.3,
            df_enhanced.get('is_night', 0) * 0.1,
            df_enhanced.get('is_large_amount', 0) * 0.1
        ]
        
        df_enhanced['composite_risk_score'] = sum(risk_components)
        
        # Risk level categorization
        df_enhanced['risk_level'] = pd.cut(
            df_enhanced['composite_risk_score'],
            bins=[0, 0.3, 0.7, 1.0, np.inf],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} risk features")
        return df_enhanced
    
    def create_ensemble_features(self, df):
        """Create advanced ensemble features with state-of-the-art techniques"""
        print("🎯 Creating Advanced Ensemble Features...")
        df_enhanced = df.copy()
        
        # Select numeric columns for dimensionality reduction
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).drop(['fraud_risk'], axis=1, errors='ignore')
        
        if len(numeric_cols.columns) > 5 and len(df_enhanced) > 1:
            # Advanced dimensionality reduction techniques
            X_scaled = StandardScaler().fit_transform(numeric_cols.fillna(0))
            
            # PCA with explained variance
            n_components = min(15, len(numeric_cols.columns), len(df_enhanced))
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(X_scaled)
            
            for i in range(pca_features.shape[1]):
                df_enhanced[f'pca_component_{i}'] = pca_features[:, i]
            
            # Store explained variance ratios
            df_enhanced['pca_explained_variance'] = np.sum(pca.explained_variance_ratio_[:5])  # Top 5 components
            
            # Independent Component Analysis (ICA)
            if len(df_enhanced) > n_components:
                try:
                    ica = FastICA(n_components=min(10, n_components), random_state=42)
                    ica_features = ica.fit_transform(X_scaled)
                    
                    for i in range(min(5, ica_features.shape[1])):  # Limit to 5 ICA components
                        df_enhanced[f'ica_component_{i}'] = ica_features[:, i]
                except:
                    for i in range(5):
                        df_enhanced[f'ica_component_{i}'] = 0
            
            # Truncated SVD for sparse-like representation
            try:
                svd = TruncatedSVD(n_components=min(8, len(numeric_cols.columns)-1), random_state=42)
                svd_features = svd.fit_transform(X_scaled)
                
                for i in range(min(5, svd_features.shape[1])):
                    df_enhanced[f'svd_component_{i}'] = svd_features[:, i]
            except:
                for i in range(5):
                    df_enhanced[f'svd_component_{i}'] = 0
            
            # Non-linear dimensionality reduction (t-SNE simulation)
            if len(df_enhanced) > 10 and len(df_enhanced) <= 1000:  # t-SNE is expensive
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_enhanced)-1))
                    tsne_features = tsne.fit_transform(X_scaled)
                    df_enhanced['tsne_component_0'] = tsne_features[:, 0]
                    df_enhanced['tsne_component_1'] = tsne_features[:, 1]
                except:
                    df_enhanced['tsne_component_0'] = 0
                    df_enhanced['tsne_component_1'] = 0
            else:
                df_enhanced['tsne_component_0'] = 0
                df_enhanced['tsne_component_1'] = 0
                
        else:
            # For single sample or few features, create simplified ensemble features
            print("ℹ️  Limited data - using simplified ensemble features")
            for i in range(5):
                df_enhanced[f'pca_component_{i}'] = 0.0
                df_enhanced[f'ica_component_{i}'] = 0.0
                df_enhanced[f'svd_component_{i}'] = 0.0
            df_enhanced['pca_explained_variance'] = 0.0
            df_enhanced['tsne_component_0'] = 0
            df_enhanced['tsne_component_1'] = 0
        
        # Advanced statistical ensemble features
        if len(numeric_cols.columns) > 1:
            df_enhanced['feature_mean'] = numeric_cols.mean(axis=1)
            df_enhanced['feature_std'] = numeric_cols.std(axis=1)
            df_enhanced['feature_skew'] = numeric_cols.skew(axis=1)
            df_enhanced['feature_kurt'] = numeric_cols.kurtosis(axis=1)
            df_enhanced['feature_entropy'] = -np.sum(
                numeric_cols.apply(lambda x: x/x.sum() * np.log(x/x.sum() + 1e-6), axis=1), axis=1
            )
            df_enhanced['feature_gini'] = numeric_cols.apply(
                lambda x: 1 - np.sum((x.sort_values(ascending=False) / x.sum()).cumsum() * 
                                   (x.sort_values(ascending=False) / x.sum())), axis=1
            )
        else:
            for col in ['feature_mean', 'feature_std', 'feature_skew', 'feature_kurt', 
                       'feature_entropy', 'feature_gini']:
                df_enhanced[col] = 0
        
        # Multi-order interaction features
        key_features = ['trans_amount', 'age', 'trans_hour']
        available_features = [f for f in key_features if f in df_enhanced.columns]
        
        # Second-order interactions
        for i, feat1 in enumerate(available_features):
            for j, feat2 in enumerate(available_features[i+1:], i+1):
                df_enhanced[f'{feat1}_{feat2}_interaction'] = df_enhanced[feat1] * df_enhanced[feat2]
                df_enhanced[f'{feat1}_{feat2}_ratio'] = df_enhanced[feat1] / (df_enhanced[feat2] + 1e-6)
        
        # Third-order interactions (limited to avoid explosion)
        if len(available_features) >= 3:
            df_enhanced['triple_interaction'] = (
                df_enhanced[available_features[0]] * 
                df_enhanced[available_features[1]] * 
                df_enhanced[available_features[2]]
            )
        
        # Polynomial features (up to degree 3)
        for feat in available_features:
            df_enhanced[f'{feat}_squared'] = df_enhanced[feat] ** 2
            df_enhanced[f'{feat}_cubed'] = df_enhanced[feat] ** 3
            df_enhanced[f'{feat}_sqrt'] = np.sqrt(np.abs(df_enhanced[feat]))
            df_enhanced[f'{feat}_log'] = np.log1p(np.abs(df_enhanced[feat]))
        
        # Advanced categorical encoding
        if 'category' in df_enhanced.columns and 'trans_amount' in df_enhanced.columns:
            # Target encoding simulation (mean amount by category)
            category_means = df_enhanced.groupby('category')['trans_amount'].mean()
            df_enhanced['category_mean_encoding'] = df_enhanced['category'].map(category_means)
            
            # Frequency encoding
            category_counts = df_enhanced['category'].value_counts()
            df_enhanced['category_frequency'] = df_enhanced['category'].map(category_counts)
        
        print(f"✅ Created {len([col for col in df_enhanced.columns if col not in df.columns])} ensemble features")
        return df_enhanced
    
    def apply_ultra_feature_engineering(self, df):
        """Apply all ultra advanced feature engineering techniques including state-of-the-art methods"""
        print("🚀 Applying Ultra Advanced Feature Engineering Pipeline...")
        print("🎯 Including State-of-the-Art Modern Features")
        print("=" * 80)
        
        original_features = len(df.columns)
        
        # Apply all feature engineering steps (existing + new state-of-the-art)
        df_enhanced = self.create_ultra_temporal_features(df)
        df_enhanced = self.create_time_series_features(df_enhanced)  # NEW: Advanced time series
        df_enhanced = self.create_ultra_amount_features(df_enhanced)
        df_enhanced = self.create_ultra_behavioral_features(df_enhanced)
        df_enhanced = self.create_deep_behavioral_embeddings(df_enhanced)  # NEW: Deep embeddings
        df_enhanced = self.create_ultra_graph_features(df_enhanced)
        df_enhanced = self.create_graph_neural_network_features(df_enhanced)  # NEW: GNN features
        df_enhanced = self.create_ultra_risk_features(df_enhanced)
        df_enhanced = self.create_adversarial_features(df_enhanced)  # NEW: Adversarial features
        df_enhanced = self.create_transformer_attention_features(df_enhanced)  # NEW: Transformer features
        df_enhanced = self.create_anomaly_detection_features(df_enhanced)  # NEW: Advanced anomaly detection
        df_enhanced = self.create_advanced_clustering_features(df_enhanced)  # NEW: Advanced clustering
        df_enhanced = self.create_ensemble_features(df_enhanced)  # Enhanced ensemble features
        
        new_features = len(df_enhanced.columns)
        
        print(f"\n🎉 ULTRA ADVANCED FEATURE ENGINEERING COMPLETE!")
        print(f"🎯 Including Cutting-Edge State-of-the-Art Features!")
        print(f"📊 Original features: {original_features}")
        print(f"🚀 Enhanced features: {new_features}")
        print(f"✨ New features added: {new_features - original_features}")
        print(f"🔬 Advanced techniques applied:")
        print(f"   • Adversarial Learning Features")
        print(f"   • Transformer Attention Mechanisms") 
        print(f"   • Graph Neural Network Features")
        print(f"   • Deep Behavioral Embeddings")
        print(f"   • Advanced Anomaly Detection")
        print(f"   • Multi-Level Clustering")
        print(f"   • Advanced Time Series Analysis")
        print(f"   • Non-linear Dimensionality Reduction")
        
        return df_enhanced
