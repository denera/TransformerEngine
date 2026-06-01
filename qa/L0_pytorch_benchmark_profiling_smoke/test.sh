# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

fail() {
  echo "SMOKE_STATUS:FAIL:$1"
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

require_nonempty_file() {
  local file="$1"
  [ -s "$file" ] || fail "Expected non-empty artifact: $file"
}

require_positive_csv_metric() {
  local csv_path="$1"
  local column_name="$2"
  python3 - "$csv_path" "$column_name" <<'PY'
import csv
import sys

csv_path, column_name = sys.argv[1], sys.argv[2]
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("CSV has no rows")
for idx, row in enumerate(rows):
    value = float(row[column_name])
    if value <= 0:
        raise SystemExit(f"Row {idx} has non-positive value: {value}")
print(f"Validated {len(rows)} rows with positive {column_name}")
PY
}

: "${TE_PATH:=/opt/transformerengine}"
: "${XML_LOG_DIR:=/tmp/te-logs}"
: "${DLCLUSTER_CONTAINER_IMAGE:=nvcr.io/nvidia/pytorch:25.04-py3}"
: "${SMOKE_OUTPUT_DIR:=$XML_LOG_DIR/benchmark_profiling_smoke}"
: "${SMOKE_REQUIRED_GPU_ARCH:=sm90}"
: "${SMOKE_ENFORCE_REQUIRED_ARCH:=1}"
: "${SMOKE_TOKEN_DIM:=1024}"
: "${SMOKE_HIDDEN_DIM:=2048}"
: "${SMOKE_OUTPUT_DIM:=2048}"
: "${SMOKE_BENCHMARK_MIN_RUN_TIME:=1.0}"
: "${SMOKE_BENCHMARK_MICROBATCHES:=8}"
: "${SMOKE_PROFILE_WARMUP_STEPS:=3}"
: "${SMOKE_PROFILE_STEPS:=10}"

mkdir -p "$SMOKE_OUTPUT_DIR"

echo "SMOKE_CONTAINER_IMAGE:$DLCLUSTER_CONTAINER_IMAGE"
echo "SMOKE_REQUIRED_GPU_ARCH:$SMOKE_REQUIRED_GPU_ARCH"
echo "SMOKE_OUTPUT_DIR:$SMOKE_OUTPUT_DIR"

require_cmd python3
require_cmd nsys

GPU_ARCH=$(python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
major, minor = torch.cuda.get_device_capability()
print(f"sm{major}{minor}")
PY
)

echo "SMOKE_GPU_ARCH:$GPU_ARCH"
if [ "$SMOKE_ENFORCE_REQUIRED_ARCH" = "1" ] && [ "$GPU_ARCH" != "$SMOKE_REQUIRED_GPU_ARCH" ]; then
  fail "Expected $SMOKE_REQUIRED_GPU_ARCH for required gating pass, got $GPU_ARCH"
fi

cd "$TE_PATH"

REQUIRED_BENCH_CSV="$SMOKE_OUTPUT_DIR/linear_bf16_benchmark.csv"
REQUIRED_PROFILE_CSV="$SMOKE_OUTPUT_DIR/linear_bf16_profile.csv"
REQUIRED_PROFILE_BASE="$SMOKE_OUTPUT_DIR/linear_bf16_profile"
REQUIRED_PROFILE_REP="${REQUIRED_PROFILE_BASE}.nsys-rep"

python3 benchmarks/linear/benchmark_linear.py \
  --recipe bf16 \
  --strict-recipe \
  --token-dim "$SMOKE_TOKEN_DIM" \
  --hidden-dim "$SMOKE_HIDDEN_DIM" \
  --output-dim "$SMOKE_OUTPUT_DIM" \
  --benchmark-min-run-time "$SMOKE_BENCHMARK_MIN_RUN_TIME" \
  --benchmark-microbatches "$SMOKE_BENCHMARK_MICROBATCHES" \
  --output-file "$REQUIRED_BENCH_CSV" \
  > "$SMOKE_OUTPUT_DIR/linear_bf16_benchmark.log" 2>&1 || fail "Required bf16 benchmark run failed"

require_nonempty_file "$REQUIRED_BENCH_CSV"
require_positive_csv_metric "$REQUIRED_BENCH_CSV" "linear_fwd_bwd_time_ms"

nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --output="$REQUIRED_PROFILE_BASE" \
  --force-overwrite true \
  --trace=cuda,nvtx,cudnn,cublas \
  python3 benchmarks/linear/benchmark_linear.py \
    --profile \
    --recipe bf16 \
    --strict-recipe \
    --token-dim "$SMOKE_TOKEN_DIM" \
    --hidden-dim "$SMOKE_HIDDEN_DIM" \
    --output-dim "$SMOKE_OUTPUT_DIM" \
    --profile-warmup-steps "$SMOKE_PROFILE_WARMUP_STEPS" \
    --profile-steps "$SMOKE_PROFILE_STEPS" \
    --output-file "$REQUIRED_PROFILE_CSV" \
  > "$SMOKE_OUTPUT_DIR/linear_bf16_profile.log" 2>&1 || fail "Required bf16 profile run failed"

require_nonempty_file "$REQUIRED_PROFILE_REP"
require_nonempty_file "$REQUIRED_PROFILE_CSV"
require_positive_csv_metric "$REQUIRED_PROFILE_CSV" "linear_profile_step_time_ms"

for recipe in mxfp8 nvfp4; do
  OPTIONAL_LOG="$SMOKE_OUTPUT_DIR/linear_${recipe}_optional.log"
  OPTIONAL_CSV="$SMOKE_OUTPUT_DIR/linear_${recipe}_optional.csv"

  python3 benchmarks/linear/benchmark_linear.py \
    --recipe "$recipe" \
    --token-dim "$SMOKE_TOKEN_DIM" \
    --hidden-dim "$SMOKE_HIDDEN_DIM" \
    --output-dim "$SMOKE_OUTPUT_DIM" \
    --benchmark-min-run-time "$SMOKE_BENCHMARK_MIN_RUN_TIME" \
    --benchmark-microbatches "$SMOKE_BENCHMARK_MICROBATCHES" \
    --output-file "$OPTIONAL_CSV" \
    > "$OPTIONAL_LOG" 2>&1 || fail "Optional recipe command failed unexpectedly: $recipe"

  if grep -q "RECIPE_STATUS:${recipe}:SKIP_UNSUPPORTED" "$OPTIONAL_LOG"; then
    echo "SMOKE_OPTIONAL_RECIPE:${recipe}:SKIP_UNSUPPORTED"
    continue
  fi

  require_nonempty_file "$OPTIONAL_CSV"
  require_positive_csv_metric "$OPTIONAL_CSV" "linear_fwd_bwd_time_ms"
  echo "SMOKE_OPTIONAL_RECIPE:${recipe}:PASS"
done

echo "SMOKE_STATUS:PASS"
