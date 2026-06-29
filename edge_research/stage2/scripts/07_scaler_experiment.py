import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier

STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
ARTIFACTS_DIR = os.path.join(STAGE2_DIR, "artifacts")
FIGURES_DIR = os.path.join(STAGE2_DIR, "figures")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")
DATA_DIR_2017 = os.path.abspath(os.path.join(STAGE2_DIR, "../../data/CICIDS2017"))
DATA_DIR_2018 = os.path.abspath(os.path.join(STAGE2_DIR, "../../data/datasets for cross validation/CSE-CIC-IDS2018"))

for d in [TABLES_DIR, ARTIFACTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "07_scaler_experiment.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_sample_data(data_dir, raw_cols, feat_map, max_rows=50000):
    sample_dfs = []
    total_loaded = 0
    raw_cols_clean = {c.strip() for c in raw_cols}
    feat_map_clean = {k.strip(): v for k, v in feat_map.items()}
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            path = os.path.join(data_dir, f)
            try:
                chunk = pd.read_csv(path, nrows=max_rows, encoding="latin1", usecols=lambda c: c.strip() in raw_cols_clean or c.strip() in ["Label", "label"])
                chunk.columns = [c.strip() for c in chunk.columns]
                
                lbl_col = None
                for c in ["Label", "label"]:
                    if c in chunk.columns:
                        lbl_col = c
                        break
                if lbl_col is None:
                    lbl_col = chunk.columns[-1]
                    
                chunk = chunk.rename(columns=lambda c: feat_map_clean.get(c, c))
                
                # Standardize label to Binary (0: BENIGN, 1: ATTACK)
                lbl_series = chunk[lbl_col].astype(str).str.strip()
                chunk["Binary_Target"] = lbl_series.apply(lambda x: 0 if x in ["BENIGN", "Benign"] else 1)
                
                sample_dfs.append(chunk)
                total_loaded += len(chunk)
                if total_loaded >= max_rows * 2:
                    break
            except Exception as e:
                logging.error(f"Error reading {f}: {e}")
                continue
                
    if not sample_dfs:
        return pd.DataFrame()
        
    df_full = pd.concat(sample_dfs, ignore_index=True)
    return df_full

def main():
    print("="*60)
    print(" STAGE 2: SCALER ROBUSTNESS EXPERIMENT (Train 2017 -> Eval 2018) ")
    print("="*60)
    
    align_path = os.path.join(TABLES_DIR, "feature_alignment.csv")
    if not os.path.exists(align_path):
        logging.error(f"Feature alignment file not found at {align_path}. Run 06_feature_alignment.py first.")
        sys.exit(1)
        
    df_align = pd.read_csv(align_path)
    features = df_align["Canonical_Feature"].tolist()
    map_2017 = dict(zip(df_align["CIC2017_Raw"], df_align["Canonical_Feature"]))
    map_2018 = dict(zip(df_align["CIC2018_Raw"], df_align["Canonical_Feature"]))
    
    print("Loading sample datasets for scaler validation...")
    df_2017 = load_sample_data(DATA_DIR_2017, df_align["CIC2017_Raw"].tolist(), map_2017, max_rows=40000)
    df_2018 = load_sample_data(DATA_DIR_2018, df_align["CIC2018_Raw"].tolist(), map_2018, max_rows=40000)
    
    if df_2017.empty or df_2018.empty:
        logging.error("Failed to load sample data for 2017 or 2018.")
        sys.exit(1)
        
    # Ensure all features exist
    for ft in features:
        if ft not in df_2017.columns: df_2017[ft] = 0.0
        if ft not in df_2018.columns: df_2018[ft] = 0.0
        
    X_train = df_2017[features].copy()
    y_train = df_2017["Binary_Target"].values
    
    X_test = df_2018[features].copy()
    y_test = df_2018["Binary_Target"].values
    
    # Preprocess numeric / missing
    for col in features:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    print(f"Data ready. Train shape (CIC2017): {X_train_imp.shape}, Test shape (CIC2018): {X_test_imp.shape}")
    
    # Define Scalers
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42)
    }
    
    results = []
    scaled_test_data = {}
    
    for name, scaler in scalers.items():
        print(f"\nEvaluating Scaler: {name} ...")
        X_tr_scaled = scaler.fit_transform(X_train_imp)
        X_te_scaled = scaler.transform(X_test_imp)
        
        scaled_test_data[name] = X_te_scaled
        
        # Save scaler artifact
        pkl_name = name.replace("Scaler", "_scaler").replace("Transformer", "_scaler").lower() + ".pkl"
        if pkl_name == "standard_scaler.pkl": pkl_path = os.path.join(ARTIFACTS_DIR, "standard_scaler.pkl")
        elif pkl_name == "robust_scaler.pkl": pkl_path = os.path.join(ARTIFACTS_DIR, "robust_scaler.pkl")
        else: pkl_path = os.path.join(ARTIFACTS_DIR, "quantile_scaler.pkl")
        
        with open(pkl_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Saved scaler artifact to {pkl_path}")
        
        # Train small validation LightGBM
        model = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_tr_scaled, y_train)
        
        y_pred = model.predict(X_te_scaled)
        y_proba = model.predict_proba(X_te_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            "Scaler": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc_auc
        })
        
    df_results = pd.DataFrame(results)
    res_path = os.path.join(TABLES_DIR, "scaler_comparison.csv")
    df_results.to_csv(res_path, index=False)
    
    print(f"\nScaler comparison report saved to {res_path}")
    print("\nScaler Cross-Dataset Robustness Evaluation Table (Train: CIC2017 -> Test: CIC2018):")
    print(df_results.to_string(index=False))
    
    # Generate Figure 1: Scaler comparison bar plot
    plt.figure(figsize=(10, 6))
    df_melted = df_results.melt(id_vars="Scaler", value_vars=["F1_Score", "ROC_AUC"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Scaler", y="Score", hue="Metric", palette="viridis")
    plt.title("Scaler Comparison: Cross-Dataset Robustness (CIC2017 -> CIC2018)")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig_cmp_path = os.path.join(FIGURES_DIR, "scaler_comparison.png")
    plt.savefig(fig_cmp_path, dpi=300)
    plt.close()
    print(f"Saved scaler comparison figure to {fig_cmp_path}")
    
    # Generate Figure 2: Feature distributions after scaling (Selecting 4 sample features)
    sample_feats = [f for f in ["Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Destination Port"] if f in features]
    if not sample_feats: sample_feats = features[:4]
    
    fig, axes = plt.subplots(len(sample_feats), 3, figsize=(15, 3 * len(sample_feats)))
    for i, feat in enumerate(sample_feats):
        feat_idx = features.index(feat)
        for j, (name, data) in enumerate(scaled_test_data.items()):
            ax = axes[i, j] if len(sample_feats) > 1 else axes[j]
            sns.histplot(data[:, feat_idx], ax=ax, kde=True, bins=30, color=sns.color_palette("Set2")[j])
            ax.set_title(f"{feat} ({name})")
            ax.set_xlabel("Scaled Value")
            ax.set_ylabel("Density")
            
    plt.tight_layout()
    fig_dist_path = os.path.join(FIGURES_DIR, "feature_distributions_after_scaling.png")
    plt.savefig(fig_dist_path, dpi=300)
    plt.close()
    print(f"Saved feature distributions figure to {fig_dist_path}")
    
    print("\nNOTE: Final scaler selection deferred as per Stage 2 requirements.")
    logging.info("Scaler experiment complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
