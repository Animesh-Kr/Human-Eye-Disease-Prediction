"""
quantise_benchmark.py
=====================
Export the OCT Retinal AI model to ONNX, apply INT8 post-training
quantisation via ONNX Runtime, and benchmark latency across three
execution paths:

  1. TensorFlow / Keras  — GPU (original)
  2. ONNX Runtime        — GPU (CUDAExecutionProvider)
  3. ONNX Runtime INT8   — CPU (CPUExecutionProvider, quantised)

Usage
-----
  python quantise_benchmark.py \
      --weights_dir models/ \
      --n_warmup 10 \
      --n_runs 100 \
      --batch_size 1

Requirements
------------
  pip install tf2onnx onnxruntime-gpu onnxruntime onnx

Output
------
  models/oct_retinal.onnx          — FP32 ONNX model
  models/oct_retinal_int8.onnx     — INT8 quantised ONNX model
  benchmark_results.csv            — latency table
  benchmark_results.md             — formatted markdown table
"""

import argparse
import time
import os
import csv
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

# ── 0. Parse arguments ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--weights_dir",  default="models/",  type=str)
parser.add_argument("--n_warmup",     default=10,          type=int)
parser.add_argument("--n_runs",       default=100,         type=int)
parser.add_argument("--batch_size",   default=1,           type=int)
parser.add_argument("--image_size",   default=224,         type=int)
parser.add_argument("--skip_gpu",     action="store_true",
                    help="Skip GPU benchmarks (CPU-only machine)")
args = parser.parse_args()

WEIGHTS_DIR = args.weights_dir
N_WARMUP    = args.n_warmup
N_RUNS      = args.n_runs
BS          = args.batch_size
IMG_SIZE    = args.image_size
ONNX_FP32   = os.path.join(WEIGHTS_DIR, "oct_retinal.onnx")
ONNX_INT8   = os.path.join(WEIGHTS_DIR, "oct_retinal_int8.onnx")
KERAS_PATH  = os.path.join(WEIGHTS_DIR, "Final_CNN_Transformer.keras")
XGB_PATH    = os.path.join(WEIGHTS_DIR, "Final_XGBoost_Hybrid.json")

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

os.makedirs(WEIGHTS_DIR, exist_ok=True)

print("=" * 60)
print("  OCT Retinal AI — ONNX Export + INT8 Quantisation")
print("=" * 60)

# ── 1. Create a dummy calibration batch ───────────────────────────────────────
dummy_input = np.random.rand(BS, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
print(f"\n[Input]  shape={dummy_input.shape}  dtype={dummy_input.dtype}")

# ── 2. Load Keras model ───────────────────────────────────────────────────────
print(f"\n[Step 1] Loading Keras model from {KERAS_PATH} ...")
keras_model = tf.keras.models.load_model(KERAS_PATH, compile=False)
print(f"         Loaded. Parameters: "
      f"{keras_model.count_params():,}")

# Get input name for correct dict-style inference
input_name = keras_model.inputs[0].name.split(":")[0]
print(f"         Input tensor name: '{input_name}'")

# ── 3. Verify Keras prediction ────────────────────────────────────────────────
print("\n[Step 2] Verifying Keras output ...")
keras_out = keras_model.predict(
    {input_name: dummy_input}, verbose=0)
print(f"         Output shape: {keras_out.shape}")
print(f"         Predicted class: {CLASSES[np.argmax(keras_out[0])]}")
print(f"         Max confidence:  {np.max(keras_out[0]):.4f}")

# ── 4. Export to ONNX FP32 ────────────────────────────────────────────────────
print(f"\n[Step 3] Exporting to ONNX FP32 → {ONNX_FP32} ...")
try:
    import tf2onnx
    import onnx

    input_signature = [
        tf.TensorSpec(
            shape=(None, IMG_SIZE, IMG_SIZE, 3),
            dtype=tf.float32,
            name=input_name
        )
    ]
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=17,
        output_path=ONNX_FP32
    )
    onnx_model = onnx.load(ONNX_FP32)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(ONNX_FP32) / 1e6
    print(f"         ✅ Exported. Size: {size_mb:.1f} MB")
except Exception as e:
    print(f"         ❌ ONNX export failed: {e}")
    print("         Install with: pip install tf2onnx onnx")
    raise

# ── 5. INT8 Post-Training Quantisation ────────────────────────────────────────
print(f"\n[Step 4] Applying INT8 PTQ → {ONNX_INT8} ...")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=ONNX_FP32,
        model_output=ONNX_INT8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    size_mb_int8 = os.path.getsize(ONNX_INT8) / 1e6
    print(f"         ✅ INT8 model saved. Size: {size_mb_int8:.1f} MB")
    size_reduction = (1 - size_mb_int8 / size_mb) * 100
    print(f"         Size reduction: {size_reduction:.1f}%")
except Exception as e:
    print(f"         ❌ Quantisation failed: {e}")
    raise

# ── 6. Benchmark helper ───────────────────────────────────────────────────────
def benchmark(fn, n_warmup, n_runs, label):
    """Run fn() n_warmup times, then time n_runs executions."""
    print(f"\n[Benchmark] {label}")
    print(f"            Warming up ({n_warmup} runs) ...")
    for _ in range(n_warmup):
        fn()

    times = []
    print(f"            Timing ({n_runs} runs) ...")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    mean_ms = np.mean(times)
    std_ms  = np.std(times)
    p50_ms  = np.percentile(times, 50)
    p95_ms  = np.percentile(times, 95)
    p99_ms  = np.percentile(times, 99)

    print(f"            Mean: {mean_ms:.2f} ms  "
          f"Std: {std_ms:.2f} ms  "
          f"P50: {p50_ms:.2f} ms  "
          f"P95: {p95_ms:.2f} ms  "
          f"P99: {p99_ms:.2f} ms")

    return {
        "backend":  label,
        "mean_ms":  round(mean_ms, 2),
        "std_ms":   round(std_ms,  2),
        "p50_ms":   round(p50_ms,  2),
        "p95_ms":   round(p95_ms,  2),
        "p99_ms":   round(p99_ms,  2),
        "n_runs":   n_runs,
    }

results = []

# ── 7. Benchmark 1: Keras GPU ─────────────────────────────────────────────────
if not args.skip_gpu and len(tf.config.list_physical_devices("GPU")) > 0:
    print("\n[Step 5] Benchmarking Keras (GPU) ...")
    @tf.function
    def keras_infer():
        return keras_model({input_name: dummy_input}, training=False)

    # Warmup to compile tf.function graph
    for _ in range(3):
        keras_infer()

    results.append(benchmark(keras_infer, N_WARMUP, N_RUNS,
                             "Keras TF GPU (FP32)"))
else:
    print("\n[Step 5] Skipping Keras GPU benchmark (no GPU / --skip_gpu)")
    results.append({
        "backend": "Keras TF GPU (FP32)", "mean_ms": "N/A",
        "std_ms": "N/A", "p50_ms": "N/A",
        "p95_ms": "N/A", "p99_ms": "N/A", "n_runs": 0
    })

# ── 8. Benchmark 2: ONNX Runtime GPU ─────────────────────────────────────────
import onnxruntime as ort

if not args.skip_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
    print("\n[Step 6] Benchmarking ONNX Runtime (GPU) ...")
    sess_gpu = ort.InferenceSession(
        ONNX_FP32,
        providers=["CUDAExecutionProvider"]
    )
    ort_input_name = sess_gpu.get_inputs()[0].name

    def onnx_gpu_infer():
        sess_gpu.run(None, {ort_input_name: dummy_input})

    results.append(benchmark(onnx_gpu_infer, N_WARMUP, N_RUNS,
                             "ONNX Runtime GPU (FP32)"))
else:
    print("\n[Step 6] Skipping ONNX GPU benchmark (no CUDA / --skip_gpu)")
    results.append({
        "backend": "ONNX Runtime GPU (FP32)", "mean_ms": "N/A",
        "std_ms": "N/A", "p50_ms": "N/A",
        "p95_ms": "N/A", "p99_ms": "N/A", "n_runs": 0
    })

# ── 9. Benchmark 3: ONNX Runtime CPU FP32 ────────────────────────────────────
print("\n[Step 7] Benchmarking ONNX Runtime (CPU FP32) ...")
sess_cpu_fp32 = ort.InferenceSession(
    ONNX_FP32,
    providers=["CPUExecutionProvider"]
)
ort_input_name_fp32 = sess_cpu_fp32.get_inputs()[0].name

def onnx_cpu_fp32_infer():
    sess_cpu_fp32.run(None, {ort_input_name_fp32: dummy_input})

results.append(benchmark(onnx_cpu_fp32_infer, N_WARMUP, N_RUNS,
                         "ONNX Runtime CPU (FP32)"))

# ── 10. Benchmark 4: ONNX Runtime CPU INT8 ───────────────────────────────────
print("\n[Step 8] Benchmarking ONNX Runtime (CPU INT8) ...")
sess_int8 = ort.InferenceSession(
    ONNX_INT8,
    providers=["CPUExecutionProvider"]
)
ort_input_name_int8 = sess_int8.get_inputs()[0].name

def onnx_int8_infer():
    sess_int8.run(None, {ort_input_name_int8: dummy_input})

results.append(benchmark(onnx_int8_infer, N_WARMUP, N_RUNS,
                         "ONNX Runtime CPU (INT8)"))

# ── 11. Accuracy check on INT8 model ─────────────────────────────────────────
print("\n[Step 9] Accuracy spot-check (FP32 vs INT8) ...")
fp32_out  = sess_cpu_fp32.run(None, {ort_input_name_fp32: dummy_input})[0]
int8_out  = sess_int8.run(None,     {ort_input_name_int8: dummy_input})[0]
max_delta = float(np.max(np.abs(fp32_out - int8_out)))
pred_fp32 = CLASSES[np.argmax(fp32_out[0])]
pred_int8 = CLASSES[np.argmax(int8_out[0])]
print(f"         FP32 prediction: {pred_fp32}  "
      f"conf={np.max(fp32_out[0]):.4f}")
print(f"         INT8 prediction: {pred_int8}  "
      f"conf={np.max(int8_out[0]):.4f}")
print(f"         Max logit delta: {max_delta:.6f}")
class_match = "✅ MATCH" if pred_fp32 == pred_int8 else "⚠️  MISMATCH"
print(f"         Prediction agreement: {class_match}")

# ── 12. Compute speedup ratios ────────────────────────────────────────────────
print("\n[Step 10] Computing speedup ratios ...")
cpu_fp32_mean = next(
    r["mean_ms"] for r in results
    if "CPU (FP32)" in r["backend"]
)
cpu_int8_mean = next(
    r["mean_ms"] for r in results
    if "INT8" in r["backend"]
)

if isinstance(cpu_fp32_mean, float) and isinstance(cpu_int8_mean, float):
    speedup = cpu_fp32_mean / cpu_int8_mean
    print(f"         CPU INT8 speedup over CPU FP32: {speedup:.2f}×")
else:
    speedup = None
    print("         Could not compute speedup (missing values)")

# ── 13. Model size summary ────────────────────────────────────────────────────
keras_size_mb = os.path.getsize(KERAS_PATH) / 1e6
fp32_size_mb  = os.path.getsize(ONNX_FP32)  / 1e6
int8_size_mb  = os.path.getsize(ONNX_INT8)  / 1e6

print("\n[Model Sizes]")
print(f"  Keras .keras  : {keras_size_mb:.1f} MB")
print(f"  ONNX FP32     : {fp32_size_mb:.1f} MB")
print(f"  ONNX INT8     : {int8_size_mb:.1f} MB  "
      f"({(1 - int8_size_mb/fp32_size_mb)*100:.1f}% smaller than FP32 ONNX)")

# ── 14. Save CSV ──────────────────────────────────────────────────────────────
csv_path = "benchmark_results.csv"
fieldnames = ["backend", "mean_ms", "std_ms", "p50_ms",
              "p95_ms", "p99_ms", "n_runs"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"\n[Saved] {csv_path}")

# ── 15. Save Markdown table ───────────────────────────────────────────────────
md_path = "benchmark_results.md"
with open(md_path, "w") as f:
    f.write("## Inference Latency Benchmark\n\n")
    f.write(f"**Hardware:** NVIDIA RTX 4060 (GPU) / Intel CPU  \n")
    f.write(f"**Batch size:** {BS}  \n")
    f.write(f"**Runs:** {N_RUNS} (after {N_WARMUP} warmup)  \n")
    f.write(f"**Input:** {IMG_SIZE}×{IMG_SIZE}×3  \n\n")
    f.write("| Backend | Mean (ms) | Std (ms) | P50 (ms) | "
            "P95 (ms) | P99 (ms) |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in results:
        f.write(f"| {r['backend']} | {r['mean_ms']} | {r['std_ms']} | "
                f"{r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} |\n")

    f.write("\n### Model Size Comparison\n\n")
    f.write("| Format | Size |\n")
    f.write("|---|---|\n")
    f.write(f"| Keras `.keras` (original) | {keras_size_mb:.1f} MB |\n")
    f.write(f"| ONNX FP32 | {fp32_size_mb:.1f} MB |\n")
    f.write(f"| ONNX INT8 (quantised) | {int8_size_mb:.1f} MB |\n")

    f.write("\n### Accuracy (FP32 vs INT8 on dummy input)\n\n")
    f.write(f"- FP32 prediction: **{pred_fp32}**  \n")
    f.write(f"- INT8 prediction: **{pred_int8}**  \n")
    f.write(f"- Max logit delta: `{max_delta:.6f}`  \n")
    f.write(f"- Prediction agreement: {class_match}  \n")

    if speedup:
        f.write(f"\n> INT8 is **{speedup:.2f}× faster** than FP32 on CPU "
                f"with `{max_delta:.4f}` max logit deviation.\n")

print(f"[Saved] {md_path}")

# ── 16. Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Keras model:    {KERAS_PATH}")
print(f"  ONNX FP32:      {ONNX_FP32}  ({fp32_size_mb:.1f} MB)")
print(f"  ONNX INT8:      {ONNX_INT8}  ({int8_size_mb:.1f} MB)")
print(f"  CSV results:    {csv_path}")
print(f"  MD table:       {md_path}")
if speedup:
    print(f"\n  🚀 CPU INT8 speedup: {speedup:.2f}× over CPU FP32")
print("=" * 60)
print("\nDone. Upload oct_retinal_int8.onnx to HuggingFace weights repo.")
