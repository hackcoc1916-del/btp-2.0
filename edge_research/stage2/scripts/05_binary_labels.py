import os
import sys
import logging
import pickle
import pandas as pd

STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
ARTIFACTS_DIR = os.path.join(STAGE2_DIR, "artifacts")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "05_binary_labels.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    print("="*60)
    print(" STAGE 2: BINARY LABEL CREATION & SERIALIZATION ")
    print("="*60)
    
    mapping_path = os.path.join(TABLES_DIR, "mapping_report.csv")
    if not os.path.exists(mapping_path):
        logging.error(f"Mapping report not found at {mapping_path}. Run 04_attack_family_mapping.py first.")
        sys.exit(1)
        
    df_map = pd.read_csv(mapping_path)
    print(f"Loaded multiclass attack family distributions across {len(df_map)} dataset-family combinations.")
    
    # Establish Binary Mapping Rule
    # BENIGN -> BENIGN, anything else -> ATTACK
    df_map["Binary_Label"] = df_map["Attack_Family"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
    
    df_binary = df_map.groupby(["Dataset", "Binary_Label"])["Frequency"].sum().reset_index()
    # Calculate percentages
    df_totals = df_binary.groupby("Dataset")["Frequency"].sum().rename("Total").reset_index()
    df_binary = df_binary.merge(df_totals, on="Dataset")
    df_binary["Percentage"] = (df_binary["Frequency"] / df_binary["Total"]) * 100
    
    print("\nConsolidated Binary Label Distributions by Dataset:")
    print(df_binary[["Dataset", "Binary_Label", "Frequency", "Percentage"]].to_string(index=False))
    
    # Define and save binary label artifact (Dictionary mapping multiclass families and raw classes to Binary)
    binary_mapping = {
        # Attack Families
        "BENIGN": "BENIGN",
        "DOS_DDOS": "ATTACK",
        "PROBING": "ATTACK",
        "BRUTE_FORCE": "ATTACK",
        "WEB_ATTACK": "ATTACK",
        "BOT": "ATTACK",
        
        # Raw Classes
        "Benign": "BENIGN",
        "DoS Hulk": "ATTACK",
        "DoS GoldenEye": "ATTACK",
        "DoS slowloris": "ATTACK",
        "DoS Slowhttptest": "ATTACK",
        "DDoS": "ATTACK",
        "DDOS attack-HOIC": "ATTACK",
        "DDoS attacks-LOIC-HTTP": "ATTACK",
        "DoS attacks-Hulk": "ATTACK",
        "DoS attacks-SlowHTTPTest": "ATTACK",
        "DoS attacks-GoldenEye": "ATTACK",
        "DoS attacks-Slowloris": "ATTACK",
        "DDOS attack-LOIC-UDP": "ATTACK",
        "DDoS HOIC": "ATTACK",
        "DDoS LOIC-HTTP": "ATTACK",
        "DDoS LOIC-UDP": "ATTACK",
        "PortScan": "ATTACK",
        "FTP-Patator": "ATTACK",
        "SSH-Patator": "ATTACK",
        "FTP-BruteForce": "ATTACK",
        "SSH-Bruteforce": "ATTACK",
        "Web Attack – Brute Force": "ATTACK",
        "Web Attack – XSS": "ATTACK",
        "Web Attack – Sql Injection": "ATTACK",
        "Brute Force -Web": "ATTACK",
        "Brute Force -XSS": "ATTACK",
        "SQL Injection": "ATTACK",
        "Web Attack - Brute Force": "ATTACK",
        "Web Attack - XSS": "ATTACK",
        "Web Attack - Sql Injection": "ATTACK",
        "Bot": "ATTACK"
    }
    
    art_path = os.path.join(ARTIFACTS_DIR, "binary_labels.pkl")
    with open(art_path, "wb") as f:
        pickle.dump(binary_mapping, f)
        
    print(f"\nSaved Binary Label mapping artifact to: {art_path}")
    logging.info(f"Binary label creation complete. Artifact saved to {art_path}.")
    print("\nDone!")

if __name__ == "__main__":
    main()
