"""
run_pipeline.py
───────────────
Master orchestrator — runs the full end-to-end pipeline in one command.

Usage
─────
    python run_pipeline.py [--skip-download] [--model all|<name>]

Steps
─────
  1. Download UCI HAR dataset
  2. Validate data pipeline
  3. Train all models (or selected model)
  4. Evaluate + generate all plots
  5. Run real-time simulation
  6. Run explainability module
  7. Print final summary
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: str, step: str):
    print(f"\n{'='*65}")
    print(f"  STEP: {step}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step}' failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  [OK] Finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="HAR full pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (already present)")
    parser.add_argument("--model", default="all",
                        help="Model: all | ANN | LSTM | CNN_LSTM | CNN_LSTM_Attention")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║  Real-Time Human Activity Recognition — Full Pipeline        ║
║  CNN-LSTM + Attention · Robustness · Explainability          ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Step 1 — Download
    if not args.skip_download:
        run("python data/download_data.py", "Download UCI HAR Dataset")
    else:
        print("[SKIP] Dataset download")

    # Step 2 — Validate pipeline
    run("python src/data_pipeline.py",
        "Validate Data Pipeline (normalisation + noise injection)")

    # Step 3 — Train
    run(f"python src/train.py --model {args.model}",
        f"Train Models ({args.model})")

    # Step 4 — Evaluate
    run("python src/evaluate.py",
        "Evaluate All Models (confusion matrix, robustness, comparison)")

    # Step 5 — Real-time simulation
    run("python src/realtime_sim.py",
        "Real-Time Simulation + Latency Benchmark")

    # Step 6 — Explainability
    run("python src/explainability.py",
        "Attention Heatmaps + SHAP Feature Importance")

    # Step 7 — Summary
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Pipeline Complete ✅                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Saved outputs:                                              ║
║    models_saved/     — trained model weights (.keras)        ║
║    outputs/          — metrics, CSVs, training logs          ║
║    plots/            — all visualisation PNGs                ║
╠══════════════════════════════════════════════════════════════╣
║  Next step — launch the Streamlit app:                       ║
║    streamlit run app/streamlit_app.py                        ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
