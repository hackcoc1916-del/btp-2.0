import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

STAGE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE3_DIR, "tables")
FIGURES_DIR = os.path.join(STAGE3_DIR, "figures")
LOGS_DIR = os.path.join(STAGE3_DIR, "logs")

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "09_gpu_benchmark.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_benchmark():
    print("Generating benchmark matrix (50,000 samples, 81 features) for CPU vs GPU fallback...")
    np.random.seed(42)
    X = np.random.rand(50000, 81)
    y = np.random.randint(0, 2, 50000)
    
    benchmark_results = []
    
    # XGBoost Benchmark
    try:
        from xgboost import XGBClassifier
        # CPU Training & Inference
        start = time.time()
        model_cpu = XGBClassifier(n_estimators=100, tree_method="hist", device="cpu", random_state=42)
        model_cpu.fit(X, y)
        cpu_train_time = time.time() - start
        
        start_inf = time.time()
        model_cpu.predict(X)
        cpu_inf_time = time.time() - start_inf
        cpu_throughput = len(X) / cpu_inf_time if cpu_inf_time > 0 else 0
        
        # GPU Training & Inference
        start = time.time()
        try:
            model_gpu = XGBClassifier(n_estimators=100, tree_method="hist", device="cuda", random_state=42)
            model_gpu.fit(X, y)
            gpu_train_time = time.time() - start
            
            start_inf = time.time()
            model_gpu.predict(X)
            gpu_inf_time = time.time() - start_inf
            gpu_throughput = len(X) / gpu_inf_time if gpu_inf_time > 0 else 0
            status = "SUCCESS_CUDA"
        except Exception as e:
            logging.warning(f"XGBoost CUDA fit failed: {e}. Fallback to CPU.")
            gpu_train_time = cpu_train_time
            gpu_inf_time = cpu_inf_time
            gpu_throughput = cpu_throughput
            status = "FALLBACK_CPU"
            
        benchmark_results.append({
            "Library": "XGBoost",
            "CPU_Train_Time_sec": cpu_train_time,
            "GPU_Train_Time_sec": gpu_train_time,
            "Train_Speedup": cpu_train_time / gpu_train_time if gpu_train_time > 0 else 1.0,
            "CPU_Inference_Throughput_sps": cpu_throughput,
            "GPU_Inference_Throughput_sps": gpu_throughput,
            "GPU_Status": status
        })
    except ImportError:
        logging.error("XGBoost not installed.")
    
    # LightGBM Benchmark
    try:
        from lightgbm import LGBMClassifier
        # CPU Training & Inference
        start = time.time()
        model_cpu = LGBMClassifier(n_estimators=100, device="cpu", random_state=42, n_jobs=-1, verbose=-1)
        model_cpu.fit(X, y)
        cpu_train_time = time.time() - start
        
        start_inf = time.time()
        model_cpu.predict(X)
        cpu_inf_time = time.time() - start_inf
        cpu_throughput = len(X) / cpu_inf_time if cpu_inf_time > 0 else 0
        
        # GPU Training & Inference
        start = time.time()
        try:
            model_gpu = LGBMClassifier(n_estimators=100, device="gpu", random_state=42, n_jobs=-1, verbose=-1)
            model_gpu.fit(X, y)
            gpu_train_time = time.time() - start
            
            start_inf = time.time()
            model_gpu.predict(X)
            gpu_inf_time = time.time() - start_inf
            gpu_throughput = len(X) / gpu_inf_time if gpu_inf_time > 0 else 0
            status = "SUCCESS_GPU"
        except Exception as e:
            logging.warning(f"LightGBM GPU fit failed: {e}. Fallback to CPU.")
            gpu_train_time = cpu_train_time
            gpu_inf_time = cpu_inf_time
            gpu_throughput = cpu_throughput
            status = "FALLBACK_CPU"
            
        benchmark_results.append({
            "Library": "LightGBM",
            "CPU_Train_Time_sec": cpu_train_time,
            "GPU_Train_Time_sec": gpu_train_time,
            "Train_Speedup": cpu_train_time / gpu_train_time if gpu_train_time > 0 else 1.0,
            "CPU_Inference_Throughput_sps": cpu_throughput,
            "GPU_Inference_Throughput_sps": gpu_throughput,
            "GPU_Status": status
        })
    except ImportError:
        logging.error("LightGBM not installed.")
        
    return pd.DataFrame(benchmark_results)

def main():
    print("="*60)
    print(" STAGE 3: GPU HARDWARE ACCELERATION BENCHMARK ")
    print("="*60)
    
    df_bench = run_benchmark()
    
    bench_path = os.path.join(TABLES_DIR, "gpu_benchmark.csv")
    df_bench.to_csv(bench_path, index=False)
    print(f"\nGPU benchmark report saved to {bench_path}")
    print("\nGPU Benchmark Table:")
    print(df_bench.to_string(index=False))
    
    # Generate plot for Inference Throughput
    plt.figure(figsize=(9, 5))
    df_plot = df_bench.melt(id_vars="Library", value_vars=["CPU_Inference_Throughput_sps", "GPU_Inference_Throughput_sps"], var_name="Mode", value_name="Throughput (Samples/sec)")
    df_plot["Mode"] = df_plot["Mode"].map({"CPU_Inference_Throughput_sps": "CPU Throughput", "GPU_Inference_Throughput_sps": "GPU Throughput (or Fallback)"})
    
    sns.barplot(data=df_plot, x="Library", y="Throughput (Samples/sec)", hue="Mode", palette="crest")
    plt.title("Hardware Acceleration Benchmark: Inference Throughput (Samples/sec)")
    plt.ylabel("Samples per Second (sps)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, "gpu_inference_throughput.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved GPU throughput figure to {fig_path}")
    
    logging.info("GPU hardware acceleration benchmark complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
