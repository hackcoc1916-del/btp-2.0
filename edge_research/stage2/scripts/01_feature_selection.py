import os
import sys
import logging
import pandas as pd

STAGE1_TABLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../stage1/tables"))
STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")
REPORTS_DIR = os.path.join(STAGE2_DIR, "reports")
ARTIFACTS_DIR = os.path.join(STAGE2_DIR, "artifacts")
FIGURES_DIR = os.path.join(STAGE2_DIR, "figures")

for d in [TABLES_DIR, LOGS_DIR, REPORTS_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "01_feature_selection.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    print("="*60)
    print(" STAGE 2: FEATURE SELECTION & TOPOLOGY LEAKAGE REMOVAL ")
    print("="*60)
    
    feat_inv_path = os.path.join(STAGE1_TABLES_DIR, "feature_inventory.csv")
    if not os.path.exists(feat_inv_path):
        logging.error(f"Stage 1 feature inventory not found at {feat_inv_path}. Cannot proceed.")
        sys.exit(1)
        
    df_inv = pd.read_csv(feat_inv_path)
    print(f"Loaded Stage 1 feature inventory with {len(df_inv)} features.")
    
    # Define remove list and topology evaluation list
    remove_candidates = ["Source Port", "Timestamp"]
    topology_candidates = ["Destination Port", "Flow Bytes/s", "Flow Packets/s"]
    
    removed_records = []
    selected_records = []
    topology_records = []
    
    for _, row in df_inv.iterrows():
        feat = row["Feature"].strip()
        if feat in remove_candidates:
            removed_records.append({
                "Feature": feat,
                "CIC2017": row.get("CIC2017", feat),
                "CIC2018": row.get("CIC2018", feat),
                "Lycos": row.get("Lycos", feat),
                "Status": "REMOVED",
                "Reason": "Direct topology leakage / missing across external validation sets."
            })
        elif feat in topology_candidates:
            reason = "Temporarily retained for separate evaluation: Essential service indicator / flow rate metric with potential numerical drift."
            selected_records.append({
                "Feature": feat,
                "CIC2017": row.get("CIC2017", feat),
                "CIC2018": row.get("CIC2018", feat),
                "Lycos": row.get("Lycos", feat),
                "Status": "TOPOLOGY_CANDIDATE",
                "Notes": reason
            })
            topology_records.append({
                "Feature": feat,
                "CIC2017": row.get("CIC2017", feat),
                "CIC2018": row.get("CIC2018", feat),
                "Lycos": row.get("Lycos", feat),
                "Evaluation_Plan": "Assess downstream classification dependence and cross-dataset drift."
            })
        else:
            selected_records.append({
                "Feature": feat,
                "CIC2017": row.get("CIC2017", feat),
                "CIC2018": row.get("CIC2018", feat),
                "Lycos": row.get("Lycos", feat),
                "Status": "SELECTED",
                "Notes": "Standard robust flow characteristic."
            })
            
    df_selected = pd.DataFrame(selected_records)
    df_removed = pd.DataFrame(removed_records)
    df_topology = pd.DataFrame(topology_records)
    
    selected_out = os.path.join(TABLES_DIR, "selected_features.csv")
    removed_out = os.path.join(TABLES_DIR, "removed_features.csv")
    topology_out = os.path.join(TABLES_DIR, "topology_candidates.csv")
    
    df_selected.to_csv(selected_out, index=False)
    df_removed.to_csv(removed_out, index=False)
    df_topology.to_csv(topology_out, index=False)
    
    print(f"\nFeature selection complete:")
    print(f"  Selected Features:    {len(df_selected)} saved to {selected_out}")
    print(f"  Removed Features:     {len(df_removed)} saved to {removed_out}")
    print(f"  Topology Candidates:  {len(df_topology)} saved to {topology_out}")
    
    print("\nRemoved Features Summary:")
    print(df_removed[["Feature", "Status", "Reason"]].to_string(index=False))
    
    print("\nTopology Candidates Summary:")
    print(df_topology.to_string(index=False))
    
    logging.info(f"Feature selection complete. Selected: {len(df_selected)}, Removed: {len(df_removed)}, Topology Candidates: {len(df_topology)}")
    print("\nDone!")

if __name__ == "__main__":
    main()
