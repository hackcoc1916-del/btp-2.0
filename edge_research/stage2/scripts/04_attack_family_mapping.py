import os
import sys
import logging
import pandas as pd

STAGE1_TABLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../stage1/tables"))
STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "04_attack_family_mapping.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Formal Mapping Dictionary
ATTACK_MAPPING = {
    # Benign
    "BENIGN": "BENIGN",
    "Benign": "BENIGN",
    
    # DoS / DDoS
    "DoS Hulk": "DOS_DDOS",
    "DoS GoldenEye": "DOS_DDOS",
    "DoS slowloris": "DOS_DDOS",
    "DoS Slowloris": "DOS_DDOS",
    "DoS Slowhttptest": "DOS_DDOS",
    "DDoS": "DOS_DDOS",
    "DDOS attack-HOIC": "DOS_DDOS",
    "DDoS attacks-LOIC-HTTP": "DOS_DDOS",
    "DoS attacks-Hulk": "DOS_DDOS",
    "DoS attacks-SlowHTTPTest": "DOS_DDOS",
    "DoS attacks-GoldenEye": "DOS_DDOS",
    "DoS attacks-Slowloris": "DOS_DDOS",
    "DDOS attack-LOIC-UDP": "DOS_DDOS",
    "DDoS HOIC": "DOS_DDOS",
    "DDoS LOIC-HTTP": "DOS_DDOS",
    "DDoS LOIC-UDP": "DOS_DDOS",
    
    # Probing
    "PortScan": "PROBING",
    
    # Brute Force
    "FTP-Patator": "BRUTE_FORCE",
    "SSH-Patator": "BRUTE_FORCE",
    "FTP-BruteForce": "BRUTE_FORCE",
    "SSH-Bruteforce": "BRUTE_FORCE",
    
    # Web Attack
    "Web Attack – Brute Force": "WEB_ATTACK",
    "Web Attack – XSS": "WEB_ATTACK",
    "Web Attack – Sql Injection": "WEB_ATTACK",
    "Brute Force -Web": "WEB_ATTACK",
    "Brute Force -XSS": "WEB_ATTACK",
    "SQL Injection": "WEB_ATTACK",
    "Web Attack - Brute Force": "WEB_ATTACK",
    "Web Attack - XSS": "WEB_ATTACK",
    "Web Attack - Sql Injection": "WEB_ATTACK",
    
    # Bot
    "Bot": "BOT",
    
    # To Remove
    "Infiltration": "REMOVE",
    "Infilteration": "REMOVE",
    "Heartbleed": "REMOVE"
}

def main():
    print("="*60)
    print(" STAGE 2: ATTACK FAMILY MAPPING & RARE CLASS REMOVAL ")
    print("="*60)
    
    class_dist_path = os.path.join(STAGE1_TABLES_DIR, "class_distribution.csv")
    if not os.path.exists(class_dist_path):
        logging.error(f"Stage 1 class distribution table not found at {class_dist_path}. Cannot proceed.")
        sys.exit(1)
        
    df_dist = pd.read_csv(class_dist_path)
    print(f"Loaded Stage 1 raw class distribution with {len(df_dist)} dataset-class combinations.")
    
    # Apply Mapping
    df_dist["Raw_Class"] = df_dist["Class"].str.strip().str.replace("\u2013", "-").str.replace("\x96", "-")
    df_dist["Attack_Family"] = df_dist["Raw_Class"].map(ATTACK_MAPPING)
    
    # Check for unmapped classes
    unmapped = df_dist[df_dist["Attack_Family"].isna()]
    if not unmapped.empty:
        print("\nWARNING: Found unmapped raw classes:")
        unmapped_disp = unmapped[["Dataset", "Raw_Class", "Frequency"]].copy()
        unmapped_disp["Raw_Class"] = unmapped_disp["Raw_Class"].apply(lambda x: x.encode('ascii', 'replace').decode('ascii'))
        print(unmapped_disp)
        df_dist["Attack_Family"] = df_dist["Attack_Family"].fillna("OTHER")
        
    # Remove specified classes
    removed = df_dist[df_dist["Attack_Family"] == "REMOVE"]
    df_kept = df_dist[df_dist["Attack_Family"] != "REMOVE"].copy()
    
    print("\nRemoved Threat Categories (Infiltration / Infilteration / Heartbleed):")
    removed_disp = removed[["Dataset", "Raw_Class", "Frequency"]].copy()
    removed_disp["Raw_Class"] = removed_disp["Raw_Class"].apply(lambda x: x.encode('ascii', 'replace').decode('ascii'))
    print(removed_disp.to_string(index=False))
    
    # Generate Mapping Report
    df_mapping_report = df_kept.groupby(["Dataset", "Attack_Family"])["Frequency"].sum().reset_index()
    # Calculate dataset-level total for percentages
    df_totals = df_mapping_report.groupby("Dataset")["Frequency"].sum().rename("Total_Dataset_Freq").reset_index()
    df_mapping_report = df_mapping_report.merge(df_totals, on="Dataset")
    df_mapping_report["Percentage"] = (df_mapping_report["Frequency"] / df_mapping_report["Total_Dataset_Freq"]) * 100
    
    out_path = os.path.join(TABLES_DIR, "mapping_report.csv")
    df_mapping_report.to_csv(out_path, index=False)
    
    print(f"\nAttack family mapping report saved to {out_path}")
    print("\nConsolidated Attack Family Distributions by Dataset:")
    print(df_mapping_report[["Dataset", "Attack_Family", "Frequency", "Percentage"]].to_string(index=False))
    
    logging.info(f"Attack family mapping complete. Total retained rows: {df_kept['Frequency'].sum()}")
    print("\nDone!")

if __name__ == "__main__":
    main()
