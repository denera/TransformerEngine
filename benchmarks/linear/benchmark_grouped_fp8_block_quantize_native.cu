/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using transformer_engine::DType;
using transformer_engine::GroupedTensorWrapper;
using transformer_engine::QuantizationConfigWrapper;
using transformer_engine::TensorWrapper;

#define CHECK_CUDA(EXPR) CheckCuda((EXPR), #EXPR, __FILE__, __LINE__)

void CheckCuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << file << ":" << line << ": CUDA call failed: " << expr << ": "
        << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

size_t DivUp(size_t value, size_t divisor) { return (value + divisor - 1) / divisor; }

size_t RoundUp(size_t value, size_t multiple) { return DivUp(value, multiple) * multiple; }

std::string Trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
    value.pop_back();
  }
  size_t first = 0;
  while (first < value.size() && value[first] == ' ') {
    ++first;
  }
  return value.substr(first);
}

std::vector<std::string> SplitCsv(const std::string &value) {
  std::vector<std::string> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = Trim(item);
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

std::vector<int> SplitCsvInts(const std::string &value) {
  std::vector<int> out;
  for (const auto &item : SplitCsv(value)) {
    out.push_back(std::stoi(item));
  }
  return out;
}

std::vector<size_t> SplitCsvSizes(const std::string &value) {
  std::vector<size_t> out;
  for (const auto &item : SplitCsv(value)) {
    out.push_back(static_cast<size_t>(std::stoull(item)));
  }
  return out;
}

std::string JsonEscape(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec;
        } else {
          out << ch;
        }
    }
  }
  out << '"';
  return out.str();
}

void WriteNullableDouble(std::ostream &os, double value) {
  if (std::isfinite(value)) {
    os << std::setprecision(12) << value;
  } else {
    os << "null";
  }
}

template <typename T>
void WriteVector(std::ostream &os, const std::vector<T> &values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
}

void WriteStringVector(std::ostream &os, const std::vector<std::string> &values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << JsonEscape(values[i]);
  }
  os << "]";
}

std::string GetEnv(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::string RunCommand(const char *command) {
  FILE *pipe = popen(command, "r");
  if (pipe == nullptr) {
    return "unknown";
  }
  std::string output;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const int status = pclose(pipe);
  if (status != 0) {
    return "unknown";
  }
  output = Trim(output);
  return output.empty() ? "unknown" : output;
}

std::string QuoteArg(const char *arg) {
  std::string value(arg);
  const bool needs_quotes =
      value.find_first_of(" \t\n\"'\\$") != std::string::npos || value.empty();
  if (!needs_quotes) {
    return value;
  }
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out += ch;
    }
  }
  out += "'";
  return out;
}

std::string ReconstructCommand(int argc, char **argv) {
  std::ostringstream oss;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << QuoteArg(argv[i]);
  }
  return oss.str();
}

struct Options {
  std::vector<int> dims{1, 2};
  std::vector<std::string> output_modes{"rowwise", "columnwise", "both"};
  std::vector<std::string> layouts{"uniform", "jagged"};
  std::vector<int> num_groups{1, 2, 4, 8, 16};
  std::vector<size_t> rows_sweep{512, 1024, 2048, 4096, 8192, 16384};
  std::vector<size_t> cols{1024, 2048, 4096, 7168, 8192};
  std::vector<std::string> dtypes{"bf16"};
  int warmup = 10;
  int iterations = 30;
  int samples = 5;
  double min_sample_ms = 25.0;
  bool adaptive_iterations = true;
  bool shard_across_gpus = true;
  bool profile = false;
  bool validate = false;
  bool smoke = false;
  bool require_speed_of_light = true;
  double high_cv_threshold = 0.10;
  double drift_threshold = 0.20;
  std::string profile_case = "all";
  std::string output_path;
};

bool IsFlag(const std::string &arg, const char *name) { return arg == name; }

Options ParseOptions(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return std::string(argv[++i]);
    };

    if (IsFlag(arg, "--dims")) {
      opts.dims = SplitCsvInts(require_value("--dims"));
    } else if (IsFlag(arg, "--output-modes")) {
      opts.output_modes = SplitCsv(require_value("--output-modes"));
    } else if (IsFlag(arg, "--layouts")) {
      opts.layouts = SplitCsv(require_value("--layouts"));
    } else if (IsFlag(arg, "--num-groups")) {
      opts.num_groups = SplitCsvInts(require_value("--num-groups"));
    } else if (IsFlag(arg, "--rows-sweep")) {
      opts.rows_sweep = SplitCsvSizes(require_value("--rows-sweep"));
    } else if (IsFlag(arg, "--cols")) {
      opts.cols = SplitCsvSizes(require_value("--cols"));
    } else if (IsFlag(arg, "--dtype")) {
      opts.dtypes = SplitCsv(require_value("--dtype"));
    } else if (IsFlag(arg, "--dtypes")) {
      opts.dtypes = SplitCsv(require_value("--dtypes"));
    } else if (IsFlag(arg, "--warmup")) {
      opts.warmup = std::stoi(require_value("--warmup"));
    } else if (IsFlag(arg, "--iterations")) {
      opts.iterations = std::stoi(require_value("--iterations"));
    } else if (IsFlag(arg, "--samples")) {
      opts.samples = std::stoi(require_value("--samples"));
    } else if (IsFlag(arg, "--min-sample-ms")) {
      opts.min_sample_ms = std::stod(require_value("--min-sample-ms"));
    } else if (IsFlag(arg, "--adaptive-iterations")) {
      opts.adaptive_iterations = true;
    } else if (IsFlag(arg, "--no-adaptive-iterations")) {
      opts.adaptive_iterations = false;
    } else if (IsFlag(arg, "--shard-across-gpus")) {
      opts.shard_across_gpus = true;
    } else if (IsFlag(arg, "--no-shard-across-gpus")) {
      opts.shard_across_gpus = false;
    } else if (IsFlag(arg, "--profile")) {
      opts.profile = true;
    } else if (IsFlag(arg, "--profile-case")) {
      opts.profile_case = require_value("--profile-case");
    } else if (IsFlag(arg, "--validate")) {
      opts.validate = true;
    } else if (IsFlag(arg, "--smoke")) {
      opts.smoke = true;
    } else if (IsFlag(arg, "--require-speed-of-light")) {
      opts.require_speed_of_light = true;
    } else if (IsFlag(arg, "--no-require-speed-of-light")) {
      opts.require_speed_of_light = false;
    } else if (IsFlag(arg, "--high-cv-threshold")) {
      opts.high_cv_threshold = std::stod(require_value("--high-cv-threshold"));
    } else if (IsFlag(arg, "--drift-threshold")) {
      opts.drift_threshold = std::stod(require_value("--drift-threshold"));
    } else if (IsFlag(arg, "--output")) {
      opts.output_path = require_value("--output");
    } else if (IsFlag(arg, "--help") || IsFlag(arg, "-h")) {
      std::cout
          << "Native grouped FP8 block-scaling quantize benchmark\n"
          << "  --dims 1,2 --output-modes rowwise,columnwise,both --layouts uniform,jagged\n"
          << "  --num-groups 1,2,4,8,16 --rows-sweep 512,1024 --cols 1024,2048\n"
          << "  --dtypes bf16,fp16 --warmup 10 --iterations 30 --samples 5\n"
          << "  --output $ORCHESTRA_BENCHMARK_RAW_REPORT\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.output_path.empty()) {
    opts.output_path = GetEnv("ORCHESTRA_BENCHMARK_RAW_REPORT");
  }
  if (opts.output_path.empty()) {
    opts.output_path = "grouped_fp8_block_quantize_native_report.json";
  }

  if (opts.smoke) {
    opts.dims = {1, 2};
    opts.output_modes = {"rowwise", "both"};
    opts.layouts = {"uniform", "jagged"};
    opts.num_groups = {1, 2};
    opts.rows_sweep = {128, 129};
    opts.cols = {128};
    opts.dtypes = {"bf16", "fp16"};
    opts.warmup = 1;
    opts.iterations = 1;
    opts.samples = 1;
    opts.min_sample_ms = 0.0;
    opts.adaptive_iterations = false;
    opts.shard_across_gpus = false;
    opts.validate = true;
  }

  if (opts.profile) {
    opts.shard_across_gpus = false;
  }

  return opts;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { Reset(bytes); }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept { MoveFrom(std::move(other)); }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFree(ptr_);
      }
      MoveFrom(std::move(other));
    }
    return *this;
  }
  void Reset(size_t bytes) {
    if (ptr_ != nullptr) {
      CHECK_CUDA(cudaFree(ptr_));
      ptr_ = nullptr;
      bytes_ = 0;
    }
    bytes_ = bytes;
    if (bytes_ > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, bytes_));
      CHECK_CUDA(cudaMemset(ptr_, 0, bytes_));
    }
  }
  void *data() const { return ptr_; }
  size_t bytes() const { return bytes_; }

 private:
  void MoveFrom(DeviceBuffer &&other) {
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
};

template <typename T>
__device__ T FromFloat(float value);

template <>
__device__ half FromFloat<half>(float value) {
  return __float2half(value);
}

template <>
__device__ nv_bfloat16 FromFloat<nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__global__ void FillInputKernel(T *data, size_t n, uint64_t seed) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const uint64_t mixed = idx * 2862933555777941757ULL + seed * 3037000493ULL + 0x9e3779b97f4a7c15ULL;
  const int bucket = static_cast<int>((mixed >> 21) & 2047ULL) - 1024;
  const float scale = 1.0f + static_cast<float>((mixed >> 11) & 7ULL) * 0.125f;
  data[idx] = FromFloat<T>(static_cast<float>(bucket) * (scale / 256.0f));
}

void FillInput(void *ptr, DType dtype, size_t elements, cudaStream_t stream) {
  if (elements == 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>(DivUp(elements, static_cast<size_t>(threads)));
  if (dtype == DType::kBFloat16) {
    FillInputKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<nv_bfloat16 *>(ptr),
                                                    elements, 1234ULL);
  } else if (dtype == DType::kFloat16) {
    FillInputKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<half *>(ptr), elements,
                                                    4321ULL);
  } else {
    throw std::runtime_error("Unsupported input dtype for fill");
  }
  CHECK_CUDA(cudaGetLastError());
}

DType InputDTypeFromName(const std::string &name) {
  if (name == "bf16" || name == "bfloat16") {
    return DType::kBFloat16;
  }
  if (name == "fp16" || name == "float16" || name == "half") {
    return DType::kFloat16;
  }
  throw std::runtime_error("Unsupported dtype: " + name);
}

std::string DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kBFloat16:
      return "bf16";
    case DType::kFloat16:
      return "fp16";
    default:
      return "unknown";
  }
}

size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kBFloat16:
    case DType::kFloat16:
      return 2;
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
      return 1;
    case DType::kFloat32:
      return 4;
    case DType::kInt64:
      return 8;
    default:
      throw std::runtime_error("Unsupported dtype size request");
  }
}

bool WantsRowwise(const std::string &mode) { return mode == "rowwise" || mode == "both"; }

bool WantsColumnwise(const std::string &mode) { return mode == "columnwise" || mode == "both"; }

std::vector<size_t> MakeRows(const std::string &layout, int num_groups, size_t base_rows) {
  if (layout == "uniform") {
    return std::vector<size_t>(num_groups, base_rows);
  }
  if (layout != "jagged") {
    throw std::runtime_error("Unsupported layout: " + layout);
  }
  static constexpr int deltas[] = {0,   -1,  1,   -64, 64,  -127, 127, -128,
                                   128, -255, 255, -3,  3,   -17,  17,  33};
  std::vector<size_t> rows;
  rows.reserve(num_groups);
  for (int i = 0; i < num_groups; ++i) {
    const int64_t value = static_cast<int64_t>(base_rows) + deltas[i % 16];
    rows.push_back(static_cast<size_t>(std::max<int64_t>(1, value)));
  }
  if (num_groups >= 4) {
    rows[0] = std::max<size_t>(1, std::min<size_t>(rows[0], 127));
    rows[1] = std::max<size_t>(rows[1], 128);
    rows[2] = std::max<size_t>(rows[2], 129);
  }
  return rows;
}

std::vector<size_t> ScaleShape(size_t rows, size_t cols, int dim, bool columnwise) {
  if (dim == 2) {
    if (columnwise) {
      return {DivUp(cols, 128), RoundUp(DivUp(rows, 128), 4)};
    }
    return {DivUp(rows, 128), RoundUp(DivUp(cols, 128), 4)};
  }
  if (columnwise) {
    return {DivUp(rows, 128), RoundUp(cols, 4)};
  }
  return {DivUp(cols, 128), RoundUp(rows, 4)};
}

size_t Product(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                         std::multiplies<size_t>());
}

size_t ScaleElements(size_t rows, size_t cols, int dim, bool columnwise) {
  return Product(ScaleShape(rows, cols, dim, columnwise));
}

struct CaseSpec {
  int id = 0;
  int block_scaling_dim = 1;
  std::string output_mode = "both";
  std::string layout = "uniform";
  int num_groups = 1;
  size_t base_rows = 0;
  size_t cols = 0;
  std::string dtype_name = "bf16";
};

struct TimingStats {
  int warmup_count = 0;
  int requested_iterations = 0;
  int sample_count = 0;
  int launches_per_logical_iteration = 1;
  std::vector<int> timed_iterations_per_sample;
  std::vector<double> sample_ms_per_iteration;
  double mean_ms = std::numeric_limits<double>::quiet_NaN();
  double median_ms = std::numeric_limits<double>::quiet_NaN();
  double stdev_ms = 0.0;
  double cv = 0.0;
  double min_ms = std::numeric_limits<double>::quiet_NaN();
  double max_ms = std::numeric_limits<double>::quiet_NaN();
  int total_logical_iterations = 0;
  int total_kernel_requests = 0;
};

struct MethodResult {
  std::string label;
  std::string path;
  TimingStats timing;
  size_t useful_bytes = 0;
  size_t estimated_physical_bytes = 0;
  double bandwidth_GBps_actual_bytes = std::numeric_limits<double>::quiet_NaN();
  double roofline_fraction = std::numeric_limits<double>::quiet_NaN();
};

struct BenchmarkRecord {
  int case_id = 0;
  bool skipped = false;
  std::string skip_reason;
  int worker_id = 0;
  int cuda_device_ordinal = 0;
  std::string gpu_name;
  std::string input_dtype;
  int block_scaling_dim = 1;
  std::string output_mode;
  std::string layout;
  int num_groups = 1;
  size_t base_rows = 0;
  size_t cols = 0;
  std::vector<size_t> rows_per_tensor;
  std::vector<int64_t> tensor_offsets;
  size_t total_rows = 0;
  size_t total_elements = 0;
  size_t rowwise_scale_elements = 0;
  size_t columnwise_scale_elements = 0;
  size_t monolithic_rowwise_scale_elements = 0;
  size_t monolithic_columnwise_scale_elements = 0;
  size_t copy_calibration_bytes = 0;
  double copy_roofline_GBps_read_write = std::numeric_limits<double>::quiet_NaN();
  TimingStats copy_timing;
  MethodResult candidate;
  MethodResult manual_loop_baseline;
  MethodResult monolithic_reference;
  double candidate_speedup_over_manual_loop = std::numeric_limits<double>::quiet_NaN();
  double candidate_ratio_vs_monolithic = std::numeric_limits<double>::quiet_NaN();
  std::string monolithic_comparability;
  std::string monolithic_comparability_reason;
  bool validation_performed = false;
  bool validation_passed = false;
  std::string validation_message;
  bool roofline_invalid_alarm = false;
  bool baseline_noise_alarm = false;
  bool baseline_drift_alarm = false;
  bool adjacent_size_instability_alarm = false;
  double baseline_drift_fraction = std::numeric_limits<double>::quiet_NaN();
  size_t candidate_planned_total_ctas = 0;
  size_t candidate_useful_total_ctas = 0;
  double candidate_total_cta_overlaunch_factor = std::numeric_limits<double>::quiet_NaN();
};

struct WorkerInfo {
  int worker_id = 0;
  int cuda_device_ordinal = 0;
  std::string gpu_name;
  std::string worker_output_path;
  std::vector<int> assigned_case_ids;
};

TimingStats SummarizeSamples(std::vector<double> samples, std::vector<int> iterations,
                             int warmup_count, int requested_iterations,
                             int launches_per_iteration) {
  TimingStats stats;
  stats.warmup_count = warmup_count;
  stats.requested_iterations = requested_iterations;
  stats.sample_count = static_cast<int>(samples.size());
  stats.launches_per_logical_iteration = launches_per_iteration;
  stats.timed_iterations_per_sample = std::move(iterations);
  stats.sample_ms_per_iteration = std::move(samples);
  if (stats.sample_ms_per_iteration.empty()) {
    return stats;
  }
  stats.total_logical_iterations =
      std::accumulate(stats.timed_iterations_per_sample.begin(),
                      stats.timed_iterations_per_sample.end(), 0);
  stats.total_kernel_requests = stats.total_logical_iterations * launches_per_iteration;
  stats.mean_ms =
      std::accumulate(stats.sample_ms_per_iteration.begin(), stats.sample_ms_per_iteration.end(),
                      0.0) /
      static_cast<double>(stats.sample_ms_per_iteration.size());
  std::vector<double> sorted = stats.sample_ms_per_iteration;
  std::sort(sorted.begin(), sorted.end());
  stats.median_ms = sorted[sorted.size() / 2];
  if (sorted.size() % 2 == 0) {
    stats.median_ms = 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
  }
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  if (stats.sample_ms_per_iteration.size() > 1) {
    double accum = 0.0;
    for (double value : stats.sample_ms_per_iteration) {
      const double diff = value - stats.mean_ms;
      accum += diff * diff;
    }
    stats.stdev_ms =
        std::sqrt(accum / static_cast<double>(stats.sample_ms_per_iteration.size() - 1));
  }
  stats.cv = stats.mean_ms > 0.0 ? stats.stdev_ms / stats.mean_ms : 0.0;
  return stats;
}

double MeasureElapsedMs(const std::function<void(int)> &fn, int iterations, cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, stream));
  fn(iterations);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());
  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms);
}

TimingStats Measure(const Options &opts, int launches_per_iteration, cudaStream_t stream,
                    const std::function<void()> &call, bool enable_profile) {
  for (int i = 0; i < opts.warmup; ++i) {
    call();
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (enable_profile) {
    CHECK_CUDA(cudaProfilerStart());
  }

  std::vector<double> samples;
  std::vector<int> timed_iterations;
  samples.reserve(opts.samples);
  timed_iterations.reserve(opts.samples);

  auto loop_fn = [&](int loops) {
    for (int i = 0; i < loops; ++i) {
      call();
    }
  };

  for (int sample = 0; sample < opts.samples; ++sample) {
    int loops = std::max(1, opts.iterations);
    double elapsed_ms = 0.0;
    while (true) {
      elapsed_ms = MeasureElapsedMs(loop_fn, loops, stream);
      if (!opts.adaptive_iterations || opts.min_sample_ms <= 0.0 ||
          elapsed_ms >= opts.min_sample_ms || loops >= (1 << 20)) {
        break;
      }
      const double multiplier = std::ceil(opts.min_sample_ms / std::max(elapsed_ms, 0.001));
      loops = std::max(loops + 1, static_cast<int>(std::ceil(loops * multiplier)));
    }
    samples.push_back(elapsed_ms / static_cast<double>(loops));
    timed_iterations.push_back(loops);
  }

  if (enable_profile) {
    CHECK_CUDA(cudaProfilerStop());
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  return SummarizeSamples(std::move(samples), std::move(timed_iterations), opts.warmup,
                          opts.iterations, launches_per_iteration);
}

TimingStats MeasureCopy(const Options &opts, size_t physical_bytes, size_t *copy_bytes_out,
                        cudaStream_t stream) {
  const size_t one_way_bytes = std::max<size_t>(DivUp(physical_bytes, 2), 4 * 1024 * 1024);
  DeviceBuffer src(one_way_bytes);
  DeviceBuffer dst(one_way_bytes);
  *copy_bytes_out = 2 * one_way_bytes;
  Options copy_opts = opts;
  copy_opts.samples = std::min(opts.samples, 3);
  copy_opts.warmup = std::min(opts.warmup, 5);
  copy_opts.iterations = std::min(opts.iterations, 20);
  copy_opts.adaptive_iterations = true;
  auto call = [&]() {
    CHECK_CUDA(cudaMemcpyAsync(dst.data(), src.data(), one_way_bytes, cudaMemcpyDeviceToDevice,
                               stream));
  };
  return Measure(copy_opts, 1, stream, call, /*enable_profile=*/false);
}

struct SingleTensorState {
  size_t rows = 0;
  size_t data_offset_elements = 0;
  size_t rowwise_scale_offset_elements = 0;
  size_t columnwise_scale_offset_elements = 0;
  size_t rowwise_scale_elements = 0;
  size_t columnwise_scale_elements = 0;
  DeviceBuffer rowwise_output;
  DeviceBuffer columnwise_output;
  DeviceBuffer rowwise_scale_inv;
  DeviceBuffer columnwise_scale_inv;
  std::unique_ptr<TensorWrapper> input;
  std::unique_ptr<TensorWrapper> output;
};

struct PreparedCase {
  CaseSpec spec;
  DType input_dtype = DType::kBFloat16;
  DType output_dtype = DType::kFloat8E4M3;
  NVTEScalingMode scaling_mode = NVTE_BLOCK_SCALING_1D;
  bool rowwise = true;
  bool columnwise = true;
  std::vector<size_t> rows;
  std::vector<int64_t> offsets;
  size_t total_rows = 0;
  size_t total_elements = 0;
  size_t max_rows = 0;
  size_t rowwise_scale_elements = 0;
  size_t columnwise_scale_elements = 0;
  size_t monolithic_rowwise_scale_elements = 0;
  size_t monolithic_columnwise_scale_elements = 0;
  DeviceBuffer input_data;
  DeviceBuffer first_dims_device;
  DeviceBuffer offsets_device;
  DeviceBuffer candidate_rowwise_output;
  DeviceBuffer candidate_columnwise_output;
  DeviceBuffer candidate_rowwise_scale_inv;
  DeviceBuffer candidate_columnwise_scale_inv;
  DeviceBuffer monolithic_rowwise_output;
  DeviceBuffer monolithic_columnwise_output;
  DeviceBuffer monolithic_rowwise_scale_inv;
  DeviceBuffer monolithic_columnwise_scale_inv;
  std::unique_ptr<GroupedTensorWrapper> grouped_input;
  std::unique_ptr<GroupedTensorWrapper> grouped_output;
  std::unique_ptr<TensorWrapper> monolithic_input;
  std::unique_ptr<TensorWrapper> monolithic_output;
  std::vector<SingleTensorState> baseline;
  QuantizationConfigWrapper quant_config;
};

PreparedCase PrepareCase(const CaseSpec &spec, cudaStream_t stream) {
  PreparedCase prep;
  prep.spec = spec;
  prep.input_dtype = InputDTypeFromName(spec.dtype_name);
  prep.scaling_mode =
      spec.block_scaling_dim == 2 ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;
  prep.rowwise = WantsRowwise(spec.output_mode);
  prep.columnwise = WantsColumnwise(spec.output_mode);
  prep.rows = MakeRows(spec.layout, spec.num_groups, spec.base_rows);
  prep.total_rows = std::accumulate(prep.rows.begin(), prep.rows.end(), static_cast<size_t>(0));
  prep.max_rows = *std::max_element(prep.rows.begin(), prep.rows.end());
  prep.offsets.resize(prep.rows.size() + 1, 0);
  for (size_t i = 0; i < prep.rows.size(); ++i) {
    prep.offsets[i + 1] =
        prep.offsets[i] + static_cast<int64_t>(prep.rows[i] * spec.cols);
  }
  prep.total_elements = static_cast<size_t>(prep.offsets.back());

  const size_t input_bytes = prep.total_elements * DTypeSize(prep.input_dtype);
  prep.input_data.Reset(input_bytes);
  FillInput(prep.input_data.data(), prep.input_dtype, prep.total_elements, stream);

  std::vector<int64_t> first_dims(prep.rows.begin(), prep.rows.end());
  prep.first_dims_device.Reset(first_dims.size() * sizeof(int64_t));
  prep.offsets_device.Reset(prep.offsets.size() * sizeof(int64_t));
  CHECK_CUDA(cudaMemcpyAsync(prep.first_dims_device.data(), first_dims.data(),
                             first_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
                             stream));
  CHECK_CUDA(cudaMemcpyAsync(prep.offsets_device.data(), prep.offsets.data(),
                             prep.offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
                             stream));

  for (size_t rows : prep.rows) {
    if (prep.rowwise) {
      prep.rowwise_scale_elements +=
          ScaleElements(rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/false);
    }
    if (prep.columnwise) {
      prep.columnwise_scale_elements +=
          ScaleElements(rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/true);
    }
  }

  if (prep.rowwise) {
    prep.candidate_rowwise_output.Reset(prep.total_elements * DTypeSize(prep.output_dtype));
    prep.candidate_rowwise_scale_inv.Reset(prep.rowwise_scale_elements * sizeof(float));
    prep.monolithic_rowwise_output.Reset(prep.total_elements * DTypeSize(prep.output_dtype));
    prep.monolithic_rowwise_scale_elements =
        ScaleElements(prep.total_rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/false);
    prep.monolithic_rowwise_scale_inv.Reset(prep.monolithic_rowwise_scale_elements * sizeof(float));
  }
  if (prep.columnwise) {
    prep.candidate_columnwise_output.Reset(prep.total_elements * DTypeSize(prep.output_dtype));
    prep.candidate_columnwise_scale_inv.Reset(prep.columnwise_scale_elements * sizeof(float));
    prep.monolithic_columnwise_output.Reset(prep.total_elements * DTypeSize(prep.output_dtype));
    prep.monolithic_columnwise_scale_elements =
        ScaleElements(prep.total_rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/true);
    prep.monolithic_columnwise_scale_inv.Reset(prep.monolithic_columnwise_scale_elements *
                                               sizeof(float));
  }

  const std::vector<size_t> logical_shape{prep.total_rows, spec.cols};
  const std::vector<size_t> flat_data_shape{prep.total_elements};
  const std::vector<size_t> first_dims_shape{prep.rows.size()};
  const std::vector<size_t> offsets_shape{prep.offsets.size()};

  prep.grouped_input =
      std::make_unique<GroupedTensorWrapper>(prep.rows.size(), logical_shape,
                                             NVTE_DELAYED_TENSOR_SCALING);
  prep.grouped_input->set_rowwise_data(prep.input_data.data(), prep.input_dtype, flat_data_shape);
  prep.grouped_input->set_first_dims(prep.first_dims_device.data(), DType::kInt64,
                                     first_dims_shape);
  prep.grouped_input->set_tensor_offsets(prep.offsets_device.data(), DType::kInt64,
                                         offsets_shape);

  prep.grouped_output =
      std::make_unique<GroupedTensorWrapper>(prep.rows.size(), logical_shape, prep.scaling_mode);
  if (prep.rowwise) {
    prep.grouped_output->set_rowwise_data(prep.candidate_rowwise_output.data(), prep.output_dtype,
                                          flat_data_shape);
    prep.grouped_output->set_rowwise_scale_inv(prep.candidate_rowwise_scale_inv.data(),
                                               DType::kFloat32,
                                               std::vector<size_t>{prep.rowwise_scale_elements});
  }
  if (prep.columnwise) {
    prep.grouped_output->set_columnwise_data(prep.candidate_columnwise_output.data(),
                                             prep.output_dtype, flat_data_shape);
    prep.grouped_output->set_columnwise_scale_inv(
        prep.candidate_columnwise_scale_inv.data(), DType::kFloat32,
        std::vector<size_t>{prep.columnwise_scale_elements});
  }
  prep.grouped_output->set_first_dims(prep.first_dims_device.data(), DType::kInt64,
                                      first_dims_shape);
  prep.grouped_output->set_tensor_offsets(prep.offsets_device.data(), DType::kInt64,
                                          offsets_shape);

  prep.monolithic_input = std::make_unique<TensorWrapper>(NVTE_DELAYED_TENSOR_SCALING);
  prep.monolithic_input->set_rowwise_data(prep.input_data.data(), prep.input_dtype, logical_shape);
  prep.monolithic_output = std::make_unique<TensorWrapper>(prep.scaling_mode);
  if (prep.rowwise) {
    prep.monolithic_output->set_rowwise_data(prep.monolithic_rowwise_output.data(),
                                             prep.output_dtype, logical_shape);
    prep.monolithic_output->set_rowwise_scale_inv(
        prep.monolithic_rowwise_scale_inv.data(), DType::kFloat32,
        ScaleShape(prep.total_rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/false));
  }
  if (prep.columnwise) {
    prep.monolithic_output->set_columnwise_data(prep.monolithic_columnwise_output.data(),
                                                prep.output_dtype,
                                                std::vector<size_t>{spec.cols, prep.total_rows});
    prep.monolithic_output->set_columnwise_scale_inv(
        prep.monolithic_columnwise_scale_inv.data(), DType::kFloat32,
        ScaleShape(prep.total_rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/true));
  }

  prep.baseline.reserve(prep.rows.size());
  size_t row_scale_offset = 0;
  size_t col_scale_offset = 0;
  for (size_t i = 0; i < prep.rows.size(); ++i) {
    SingleTensorState tensor;
    tensor.rows = prep.rows[i];
    tensor.data_offset_elements = static_cast<size_t>(prep.offsets[i]);
    tensor.rowwise_scale_offset_elements = row_scale_offset;
    tensor.columnwise_scale_offset_elements = col_scale_offset;
    const size_t tensor_elements = tensor.rows * spec.cols;
    auto *input_ptr =
        static_cast<unsigned char *>(prep.input_data.data()) +
        tensor.data_offset_elements * DTypeSize(prep.input_dtype);

    tensor.input = std::make_unique<TensorWrapper>(NVTE_DELAYED_TENSOR_SCALING);
    tensor.input->set_rowwise_data(input_ptr, prep.input_dtype,
                                   std::vector<size_t>{tensor.rows, spec.cols});
    tensor.output = std::make_unique<TensorWrapper>(prep.scaling_mode);
    if (prep.rowwise) {
      tensor.rowwise_scale_elements =
          ScaleElements(tensor.rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/false);
      tensor.rowwise_output.Reset(tensor_elements * DTypeSize(prep.output_dtype));
      tensor.rowwise_scale_inv.Reset(tensor.rowwise_scale_elements * sizeof(float));
      tensor.output->set_rowwise_data(tensor.rowwise_output.data(), prep.output_dtype,
                                      std::vector<size_t>{tensor.rows, spec.cols});
      tensor.output->set_rowwise_scale_inv(
          tensor.rowwise_scale_inv.data(), DType::kFloat32,
          ScaleShape(tensor.rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/false));
      row_scale_offset += tensor.rowwise_scale_elements;
    }
    if (prep.columnwise) {
      tensor.columnwise_scale_elements =
          ScaleElements(tensor.rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/true);
      tensor.columnwise_output.Reset(tensor_elements * DTypeSize(prep.output_dtype));
      tensor.columnwise_scale_inv.Reset(tensor.columnwise_scale_elements * sizeof(float));
      tensor.output->set_columnwise_data(tensor.columnwise_output.data(), prep.output_dtype,
                                         std::vector<size_t>{spec.cols, tensor.rows});
      tensor.output->set_columnwise_scale_inv(
          tensor.columnwise_scale_inv.data(), DType::kFloat32,
          ScaleShape(tensor.rows, spec.cols, spec.block_scaling_dim, /*columnwise=*/true));
      col_scale_offset += tensor.columnwise_scale_elements;
    }
    prep.baseline.push_back(std::move(tensor));
  }

  prep.quant_config.set_force_pow_2_scales(true);
  prep.quant_config.set_amax_epsilon(0.0f);
  prep.quant_config.set_grouped_max_first_dim(prep.max_rows);

  CHECK_CUDA(cudaStreamSynchronize(stream));
  return prep;
}

void RunCandidate(PreparedCase *prep, cudaStream_t stream) {
  nvte_group_quantize(prep->grouped_input->data(), prep->grouped_output->data(),
                      prep->quant_config, stream);
}

void RunManualLoopBaseline(PreparedCase *prep, cudaStream_t stream) {
  for (auto &tensor : prep->baseline) {
    nvte_quantize_v2(tensor.input->data(), tensor.output->data(), prep->quant_config, stream);
  }
}

void RunMonolithicReference(PreparedCase *prep, cudaStream_t stream) {
  nvte_quantize_v2(prep->monolithic_input->data(), prep->monolithic_output->data(),
                   prep->quant_config, stream);
}

std::vector<unsigned char> CopyDeviceBytes(const void *ptr, size_t bytes) {
  std::vector<unsigned char> host(bytes);
  if (bytes != 0) {
    CHECK_CUDA(cudaMemcpy(host.data(), ptr, bytes, cudaMemcpyDeviceToHost));
  }
  return host;
}

bool CompareBytes(const std::string &name, const unsigned char *lhs, const unsigned char *rhs,
                  size_t bytes, std::string *message) {
  for (size_t i = 0; i < bytes; ++i) {
    if (lhs[i] != rhs[i]) {
      std::ostringstream oss;
      oss << name << " mismatch at byte " << i << ": candidate=" << static_cast<int>(lhs[i])
          << " baseline=" << static_cast<int>(rhs[i]);
      *message = oss.str();
      return false;
    }
  }
  return true;
}

void MemsetIfAllocated(DeviceBuffer *buffer, int value, cudaStream_t stream) {
  if (buffer->bytes() != 0) {
    CHECK_CUDA(cudaMemsetAsync(buffer->data(), value, buffer->bytes(), stream));
  }
}

bool ValidateCandidateAgainstManualLoop(PreparedCase *prep, cudaStream_t stream,
                                        std::string *message) {
  MemsetIfAllocated(&prep->candidate_rowwise_output, 0, stream);
  MemsetIfAllocated(&prep->candidate_columnwise_output, 0, stream);
  MemsetIfAllocated(&prep->candidate_rowwise_scale_inv, 0, stream);
  MemsetIfAllocated(&prep->candidate_columnwise_scale_inv, 0, stream);
  for (auto &tensor : prep->baseline) {
    MemsetIfAllocated(&tensor.rowwise_output, 0, stream);
    MemsetIfAllocated(&tensor.columnwise_output, 0, stream);
    MemsetIfAllocated(&tensor.rowwise_scale_inv, 0, stream);
    MemsetIfAllocated(&tensor.columnwise_scale_inv, 0, stream);
  }
  RunCandidate(prep, stream);
  RunManualLoopBaseline(prep, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  const auto candidate_row =
      CopyDeviceBytes(prep->candidate_rowwise_output.data(), prep->candidate_rowwise_output.bytes());
  const auto candidate_col = CopyDeviceBytes(prep->candidate_columnwise_output.data(),
                                            prep->candidate_columnwise_output.bytes());
  const auto candidate_row_scale = CopyDeviceBytes(prep->candidate_rowwise_scale_inv.data(),
                                                  prep->candidate_rowwise_scale_inv.bytes());
  const auto candidate_col_scale = CopyDeviceBytes(prep->candidate_columnwise_scale_inv.data(),
                                                  prep->candidate_columnwise_scale_inv.bytes());

  for (size_t i = 0; i < prep->baseline.size(); ++i) {
    const auto &tensor = prep->baseline[i];
    const size_t tensor_bytes = tensor.rows * prep->spec.cols * DTypeSize(prep->output_dtype);
    if (prep->rowwise) {
      const auto baseline_row = CopyDeviceBytes(tensor.rowwise_output.data(), tensor_bytes);
      if (!CompareBytes("rowwise_data[tensor=" + std::to_string(i) + "]",
                        candidate_row.data() + tensor.data_offset_elements, baseline_row.data(),
                        tensor_bytes, message)) {
        return false;
      }
      const auto baseline_row_scale =
          CopyDeviceBytes(tensor.rowwise_scale_inv.data(), tensor.rowwise_scale_inv.bytes());
      if (!CompareBytes("rowwise_scale_inv[tensor=" + std::to_string(i) + "]",
                        candidate_row_scale.data() +
                            tensor.rowwise_scale_offset_elements * sizeof(float),
                        baseline_row_scale.data(), tensor.rowwise_scale_inv.bytes(), message)) {
        return false;
      }
    }
    if (prep->columnwise) {
      const auto baseline_col = CopyDeviceBytes(tensor.columnwise_output.data(), tensor_bytes);
      if (!CompareBytes("columnwise_data[tensor=" + std::to_string(i) + "]",
                        candidate_col.data() + tensor.data_offset_elements, baseline_col.data(),
                        tensor_bytes, message)) {
        return false;
      }
      const auto baseline_col_scale =
          CopyDeviceBytes(tensor.columnwise_scale_inv.data(), tensor.columnwise_scale_inv.bytes());
      if (!CompareBytes("columnwise_scale_inv[tensor=" + std::to_string(i) + "]",
                        candidate_col_scale.data() +
                            tensor.columnwise_scale_offset_elements * sizeof(float),
                        baseline_col_scale.data(), tensor.columnwise_scale_inv.bytes(), message)) {
        return false;
      }
    }
  }
  *message = "candidate grouped output and scale buffers match manual-loop baseline";
  return true;
}

size_t UsefulBytesForGrouped(const PreparedCase &prep) {
  size_t bytes = prep.total_elements * DTypeSize(prep.input_dtype);
  if (prep.rowwise) {
    bytes += prep.total_elements * DTypeSize(prep.output_dtype);
    bytes += prep.rowwise_scale_elements * sizeof(float);
  }
  if (prep.columnwise) {
    bytes += prep.total_elements * DTypeSize(prep.output_dtype);
    bytes += prep.columnwise_scale_elements * sizeof(float);
  }
  return bytes;
}

size_t UsefulBytesForMonolithic(const PreparedCase &prep) {
  size_t bytes = prep.total_elements * DTypeSize(prep.input_dtype);
  if (prep.rowwise) {
    bytes += prep.total_elements * DTypeSize(prep.output_dtype);
    bytes += prep.monolithic_rowwise_scale_elements * sizeof(float);
  }
  if (prep.columnwise) {
    bytes += prep.total_elements * DTypeSize(prep.output_dtype);
    bytes += prep.monolithic_columnwise_scale_elements * sizeof(float);
  }
  return bytes;
}

std::pair<std::string, std::string> MonolithicComparability(const PreparedCase &prep) {
  const bool uniform = prep.spec.layout == "uniform";
  bool exact_blocks = prep.spec.cols % 128 == 0;
  for (size_t rows : prep.rows) {
    exact_blocks = exact_blocks && (rows % 128 == 0);
  }
  if (uniform && exact_blocks) {
    return {"comparable",
            "uniform tensors with row and column dimensions aligned to 128-element scale blocks"};
  }
  if (uniform) {
    return {"partially_comparable",
            "uniform tensors avoid jagged metadata effects but at least one dimension is not an "
            "exact 128-element scale-block multiple"};
  }
  return {"not_comparable",
          "jagged grouped boundaries require per-member scale-block isolation that a collapsed "
          "monolithic tensor does not preserve"};
}

double BandwidthGBps(size_t bytes, const TimingStats &stats) {
  if (!std::isfinite(stats.mean_ms) || stats.mean_ms <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(bytes) / (stats.mean_ms / 1000.0) / 1.0e9;
}

BenchmarkRecord MakeSkippedRecord(const CaseSpec &spec, int worker_id, int device,
                                  const std::string &gpu_name, const std::string &reason) {
  BenchmarkRecord record;
  record.case_id = spec.id;
  record.skipped = true;
  record.skip_reason = reason;
  record.worker_id = worker_id;
  record.cuda_device_ordinal = device;
  record.gpu_name = gpu_name;
  record.input_dtype = spec.dtype_name;
  record.block_scaling_dim = spec.block_scaling_dim;
  record.output_mode = spec.output_mode;
  record.layout = spec.layout;
  record.num_groups = spec.num_groups;
  record.base_rows = spec.base_rows;
  record.cols = spec.cols;
  record.rows_per_tensor = MakeRows(spec.layout, spec.num_groups, spec.base_rows);
  return record;
}

BenchmarkRecord RunCase(const Options &opts, const CaseSpec &spec, int worker_id, int device,
                        const std::string &gpu_name, cudaStream_t stream) {
  if (spec.block_scaling_dim == 2 && spec.output_mode == "columnwise") {
    return MakeSkippedRecord(
        spec, worker_id, device, gpu_name,
        "non_grouped_float8_block_dim2_columnwise_only_manual_loop_baseline_unsupported");
  }

  PreparedCase prep = PrepareCase(spec, stream);
  BenchmarkRecord record;
  record.case_id = spec.id;
  record.worker_id = worker_id;
  record.cuda_device_ordinal = device;
  record.gpu_name = gpu_name;
  record.input_dtype = DTypeName(prep.input_dtype);
  record.block_scaling_dim = spec.block_scaling_dim;
  record.output_mode = spec.output_mode;
  record.layout = spec.layout;
  record.num_groups = spec.num_groups;
  record.base_rows = spec.base_rows;
  record.cols = spec.cols;
  record.rows_per_tensor = prep.rows;
  record.tensor_offsets = prep.offsets;
  record.total_rows = prep.total_rows;
  record.total_elements = prep.total_elements;
  record.rowwise_scale_elements = prep.rowwise_scale_elements;
  record.columnwise_scale_elements = prep.columnwise_scale_elements;
  record.monolithic_rowwise_scale_elements = prep.monolithic_rowwise_scale_elements;
  record.monolithic_columnwise_scale_elements = prep.monolithic_columnwise_scale_elements;

  const auto comparability = MonolithicComparability(prep);
  record.monolithic_comparability = comparability.first;
  record.monolithic_comparability_reason = comparability.second;

  const size_t tiles_n = DivUp(spec.cols, 128);
  const size_t launch_row_tiles = DivUp(prep.max_rows, 128);
  size_t useful_row_tiles = 0;
  for (size_t rows : prep.rows) {
    useful_row_tiles += DivUp(rows, 128);
  }
  record.candidate_planned_total_ctas =
      tiles_n * launch_row_tiles * static_cast<size_t>(spec.num_groups);
  record.candidate_useful_total_ctas = tiles_n * useful_row_tiles;
  record.candidate_total_cta_overlaunch_factor =
      record.candidate_useful_total_ctas == 0
          ? std::numeric_limits<double>::quiet_NaN()
          : static_cast<double>(record.candidate_planned_total_ctas) /
                static_cast<double>(record.candidate_useful_total_ctas);

  if (opts.validate) {
    record.validation_performed = true;
    record.validation_passed = ValidateCandidateAgainstManualLoop(&prep, stream,
                                                                  &record.validation_message);
    if (!record.validation_passed) {
      throw std::runtime_error("Validation failed for case " + std::to_string(spec.id) + ": " +
                               record.validation_message);
    }
  }

  const size_t grouped_bytes = UsefulBytesForGrouped(prep);
  const size_t monolithic_bytes = UsefulBytesForMonolithic(prep);
  record.copy_timing = MeasureCopy(opts, grouped_bytes, &record.copy_calibration_bytes, stream);
  record.copy_roofline_GBps_read_write =
      BandwidthGBps(record.copy_calibration_bytes, record.copy_timing);

  auto candidate_call = [&]() { RunCandidate(&prep, stream); };
  auto baseline_call = [&]() { RunManualLoopBaseline(&prep, stream); };
  auto monolithic_call = [&]() { RunMonolithicReference(&prep, stream); };

  record.candidate.label = "candidate_grouped_fp8_block_quantize";
  record.candidate.path =
      "nvte_group_quantize -> fp8_blockwise::group_quantize<block_scaling_dim=" +
      std::to_string(spec.block_scaling_dim) + ">";
  record.candidate.useful_bytes = grouped_bytes;
  record.candidate.estimated_physical_bytes = grouped_bytes;
  record.candidate.timing = Measure(opts, /*launches_per_iteration=*/1, stream, candidate_call,
                                    opts.profile);
  record.candidate.bandwidth_GBps_actual_bytes =
      BandwidthGBps(record.candidate.estimated_physical_bytes, record.candidate.timing);

  record.manual_loop_baseline.label = "manual_loop_split_quantize_baseline";
  record.manual_loop_baseline.path =
      "for each tensor: nvte_quantize_v2 established non-grouped FP8 block-scaling path";
  record.manual_loop_baseline.useful_bytes = grouped_bytes;
  record.manual_loop_baseline.estimated_physical_bytes = grouped_bytes;
  record.manual_loop_baseline.timing =
      Measure(opts, /*launches_per_iteration=*/spec.num_groups, stream, baseline_call,
              opts.profile);
  record.manual_loop_baseline.bandwidth_GBps_actual_bytes =
      BandwidthGBps(record.manual_loop_baseline.estimated_physical_bytes,
                    record.manual_loop_baseline.timing);

  record.monolithic_reference.label = "monolithic_collapsed_tensor_reference";
  record.monolithic_reference.path =
      "single nvte_quantize_v2 invocation on collapsed contiguous tensor";
  record.monolithic_reference.useful_bytes = monolithic_bytes;
  record.monolithic_reference.estimated_physical_bytes = monolithic_bytes;
  record.monolithic_reference.timing =
      Measure(opts, /*launches_per_iteration=*/1, stream, monolithic_call, opts.profile);
  record.monolithic_reference.bandwidth_GBps_actual_bytes =
      BandwidthGBps(record.monolithic_reference.estimated_physical_bytes,
                    record.monolithic_reference.timing);

  const double roofline = record.copy_roofline_GBps_read_write;
  record.candidate.roofline_fraction = record.candidate.bandwidth_GBps_actual_bytes / roofline;
  record.manual_loop_baseline.roofline_fraction =
      record.manual_loop_baseline.bandwidth_GBps_actual_bytes / roofline;
  record.monolithic_reference.roofline_fraction =
      record.monolithic_reference.bandwidth_GBps_actual_bytes / roofline;
  record.candidate_speedup_over_manual_loop =
      record.candidate.bandwidth_GBps_actual_bytes /
      record.manual_loop_baseline.bandwidth_GBps_actual_bytes;
  if (record.monolithic_comparability == "comparable") {
    record.candidate_ratio_vs_monolithic =
        record.candidate.bandwidth_GBps_actual_bytes /
        record.monolithic_reference.bandwidth_GBps_actual_bytes;
  }
  record.roofline_invalid_alarm = record.candidate.roofline_fraction > 1.0 ||
                                  record.manual_loop_baseline.roofline_fraction > 1.0 ||
                                  record.monolithic_reference.roofline_fraction > 1.0;
  record.baseline_noise_alarm = record.manual_loop_baseline.timing.cv > opts.high_cv_threshold;
  if (std::isfinite(record.manual_loop_baseline.timing.mean_ms) &&
      record.manual_loop_baseline.timing.mean_ms > 0.0) {
    record.baseline_drift_fraction =
        (record.manual_loop_baseline.timing.max_ms - record.manual_loop_baseline.timing.min_ms) /
        record.manual_loop_baseline.timing.mean_ms;
    record.baseline_drift_alarm = record.baseline_drift_fraction > opts.drift_threshold;
  }
  return record;
}

void WriteTimingStats(std::ostream &os, const TimingStats &stats) {
  os << "{";
  os << "\"warmup_count\":" << stats.warmup_count << ",";
  os << "\"requested_iterations\":" << stats.requested_iterations << ",";
  os << "\"sample_count\":" << stats.sample_count << ",";
  os << "\"launches_per_logical_iteration\":" << stats.launches_per_logical_iteration << ",";
  os << "\"timed_iterations_per_sample\":";
  WriteVector(os, stats.timed_iterations_per_sample);
  os << ",";
  os << "\"sample_ms_per_iteration\":";
  WriteVector(os, stats.sample_ms_per_iteration);
  os << ",";
  os << "\"mean_ms\":";
  WriteNullableDouble(os, stats.mean_ms);
  os << ",\"median_ms\":";
  WriteNullableDouble(os, stats.median_ms);
  os << ",\"stdev_ms\":";
  WriteNullableDouble(os, stats.stdev_ms);
  os << ",\"coefficient_of_variation\":";
  WriteNullableDouble(os, stats.cv);
  os << ",\"min_ms\":";
  WriteNullableDouble(os, stats.min_ms);
  os << ",\"max_ms\":";
  WriteNullableDouble(os, stats.max_ms);
  os << ",\"total_logical_iterations\":" << stats.total_logical_iterations;
  os << ",\"total_kernel_requests\":" << stats.total_kernel_requests;
  os << "}";
}

void WriteMethodResult(std::ostream &os, const MethodResult &result) {
  os << "{";
  os << "\"label\":" << JsonEscape(result.label) << ",";
  os << "\"path\":" << JsonEscape(result.path) << ",";
  os << "\"useful_bytes\":" << result.useful_bytes << ",";
  os << "\"estimated_physical_bytes\":" << result.estimated_physical_bytes << ",";
  os << "\"bandwidth_GBps_actual_bytes\":";
  WriteNullableDouble(os, result.bandwidth_GBps_actual_bytes);
  os << ",\"roofline_fraction\":";
  WriteNullableDouble(os, result.roofline_fraction);
  os << ",\"timing\":";
  WriteTimingStats(os, result.timing);
  os << "}";
}

void WriteRecord(std::ostream &os, const BenchmarkRecord &record) {
  os << "{";
  os << "\"case_id\":" << record.case_id << ",";
  os << "\"skipped\":" << (record.skipped ? "true" : "false") << ",";
  os << "\"skip_reason\":" << JsonEscape(record.skip_reason) << ",";
  os << "\"primary_evidence_layer\":\"native_cxx_cuda_nvte_common_api\",";
  os << "\"framework_involvement\":\"none\",";
  os << "\"worker_id\":" << record.worker_id << ",";
  os << "\"cuda_device_ordinal\":" << record.cuda_device_ordinal << ",";
  os << "\"gpu_name\":" << JsonEscape(record.gpu_name) << ",";
  os << "\"input_dtype\":" << JsonEscape(record.input_dtype) << ",";
  os << "\"fp8_dtype\":\"float8e4m3\",";
  os << "\"block_scaling_dim\":" << record.block_scaling_dim << ",";
  os << "\"output_mode\":" << JsonEscape(record.output_mode) << ",";
  os << "\"layout\":" << JsonEscape(record.layout) << ",";
  os << "\"num_groups\":" << record.num_groups << ",";
  os << "\"base_rows\":" << record.base_rows << ",";
  os << "\"cols\":" << record.cols << ",";
  os << "\"rows_per_tensor\":";
  WriteVector(os, record.rows_per_tensor);
  os << ",\"tensor_offsets\":";
  WriteVector(os, record.tensor_offsets);
  os << ",\"total_rows\":" << record.total_rows << ",";
  os << "\"total_elements\":" << record.total_elements << ",";
  os << "\"rowwise_scale_elements\":" << record.rowwise_scale_elements << ",";
  os << "\"columnwise_scale_elements\":" << record.columnwise_scale_elements << ",";
  os << "\"monolithic_rowwise_scale_elements\":" << record.monolithic_rowwise_scale_elements
     << ",";
  os << "\"monolithic_columnwise_scale_elements\":" << record.monolithic_columnwise_scale_elements
     << ",";
  os << "\"launch_evidence\":{";
  os << "\"candidate_grouped_quantize_requests_per_logical_iteration\":1,";
  os << "\"manual_loop_single_tensor_quantize_requests_per_logical_iteration\":"
     << record.num_groups << ",";
  os << "\"monolithic_quantize_requests_per_logical_iteration\":1,";
  os << "\"candidate_planned_total_ctas\":" << record.candidate_planned_total_ctas << ",";
  os << "\"candidate_useful_total_ctas\":" << record.candidate_useful_total_ctas << ",";
  os << "\"candidate_total_cta_overlaunch_factor\":";
  WriteNullableDouble(os, record.candidate_total_cta_overlaunch_factor);
  os << ",\"explicit_first_dims_metadata\":true,";
  os << "\"grouped_max_first_dim_configured\":true";
  os << "},";
  os << "\"copy_calibration_bytes\":" << record.copy_calibration_bytes << ",";
  os << "\"copy_roofline_GBps_read_write\":";
  WriteNullableDouble(os, record.copy_roofline_GBps_read_write);
  os << ",\"copy_timing\":";
  WriteTimingStats(os, record.copy_timing);
  os << ",\"monolithic_comparability\":" << JsonEscape(record.monolithic_comparability) << ",";
  os << "\"monolithic_comparability_reason\":"
     << JsonEscape(record.monolithic_comparability_reason) << ",";
  os << "\"validation_performed\":" << (record.validation_performed ? "true" : "false") << ",";
  os << "\"validation_passed\":" << (record.validation_passed ? "true" : "false") << ",";
  os << "\"validation_message\":" << JsonEscape(record.validation_message) << ",";
  os << "\"candidate\":";
  WriteMethodResult(os, record.candidate);
  os << ",\"manual_loop_baseline\":";
  WriteMethodResult(os, record.manual_loop_baseline);
  os << ",\"monolithic_reference\":";
  WriteMethodResult(os, record.monolithic_reference);
  os << ",\"candidate_speedup_over_manual_loop\":";
  WriteNullableDouble(os, record.candidate_speedup_over_manual_loop);
  os << ",\"candidate_ratio_vs_monolithic\":";
  WriteNullableDouble(os, record.candidate_ratio_vs_monolithic);
  os << ",\"roofline_invalid_alarm\":" << (record.roofline_invalid_alarm ? "true" : "false");
  os << ",\"baseline_noise_alarm\":" << (record.baseline_noise_alarm ? "true" : "false");
  os << ",\"baseline_drift_alarm\":" << (record.baseline_drift_alarm ? "true" : "false");
  os << ",\"baseline_drift_fraction\":";
  WriteNullableDouble(os, record.baseline_drift_fraction);
  os << ",\"adjacent_size_instability_alarm\":"
     << (record.adjacent_size_instability_alarm ? "true" : "false");
  os << "}";
}

std::vector<CaseSpec> BuildCases(const Options &opts) {
  std::vector<CaseSpec> cases;
  int id = 0;
  for (const std::string &dtype : opts.dtypes) {
    for (int dim : opts.dims) {
      for (const std::string &mode : opts.output_modes) {
        for (const std::string &layout : opts.layouts) {
          for (int groups : opts.num_groups) {
            for (size_t rows : opts.rows_sweep) {
              for (size_t cols : opts.cols) {
                CaseSpec spec;
                spec.id = id++;
                spec.block_scaling_dim = dim;
                spec.output_mode = mode;
                spec.layout = layout;
                spec.num_groups = groups;
                spec.base_rows = rows;
                spec.cols = cols;
                spec.dtype_name = dtype;
                cases.push_back(std::move(spec));
              }
            }
          }
        }
      }
    }
  }
  return cases;
}

int ParseSlurmGpuCount(const std::string &value) {
  if (value.empty()) {
    return 0;
  }
  int count = 0;
  for (const auto &entry : SplitCsv(value)) {
    const size_t dash = entry.find('-');
    if (dash != std::string::npos) {
      const int begin = std::stoi(entry.substr(0, dash));
      const int end = std::stoi(entry.substr(dash + 1));
      count += std::max(0, end - begin + 1);
    } else {
      ++count;
    }
  }
  return count;
}

int ParsePositiveIntString(const std::string &value) {
  if (value.empty()) {
    return 0;
  }
  for (char ch : value) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) {
      return 0;
    }
  }
  return std::stoi(value);
}

std::vector<int> SelectDevices(const Options &opts) {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    throw std::runtime_error("No CUDA devices visible to benchmark process");
  }
  if (!opts.shard_across_gpus) {
    return {0};
  }

  int allocated_count = ParseSlurmGpuCount(GetEnv("SLURM_JOB_GPUS"));
  const std::string slurm_gpus_on_node = GetEnv("SLURM_GPUS_ON_NODE");
  if (allocated_count == 0) {
    allocated_count = ParsePositiveIntString(slurm_gpus_on_node);
  }
  if (allocated_count == 0) {
    allocated_count = device_count;
  }
  const int selected_count = std::max(1, std::min(device_count, allocated_count));
  std::vector<int> devices(selected_count);
  std::iota(devices.begin(), devices.end(), 0);
  return devices;
}

std::string DeviceName(int device) {
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  return prop.name;
}

void MarkAdjacentInstability(std::vector<BenchmarkRecord> *records) {
  std::vector<size_t> indices(records->size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
    const auto &a = (*records)[lhs];
    const auto &b = (*records)[rhs];
    return std::tie(a.input_dtype, a.block_scaling_dim, a.output_mode, a.layout, a.num_groups,
                    a.cols, a.base_rows) <
           std::tie(b.input_dtype, b.block_scaling_dim, b.output_mode, b.layout, b.num_groups,
                    b.cols, b.base_rows);
  });
  for (size_t k = 1; k < indices.size(); ++k) {
    auto &prev = (*records)[indices[k - 1]];
    auto &cur = (*records)[indices[k]];
    if (prev.skipped || cur.skipped) {
      continue;
    }
    const bool comparable = prev.input_dtype == cur.input_dtype &&
                            prev.block_scaling_dim == cur.block_scaling_dim &&
                            prev.output_mode == cur.output_mode && prev.layout == cur.layout &&
                            prev.num_groups == cur.num_groups && prev.cols == cur.cols;
    if (!comparable || prev.candidate.bandwidth_GBps_actual_bytes <= 0.0 ||
        cur.candidate.bandwidth_GBps_actual_bytes <= 0.0) {
      continue;
    }
    const double ratio =
        cur.candidate.bandwidth_GBps_actual_bytes / prev.candidate.bandwidth_GBps_actual_bytes;
    if (ratio > 2.0 || ratio < 0.5) {
      prev.adjacent_size_instability_alarm = true;
      cur.adjacent_size_instability_alarm = true;
    }
  }
}

void WriteWorkerJsonl(const std::string &path, const std::vector<BenchmarkRecord> &records) {
  const std::filesystem::path worker_path(path);
  if (worker_path.has_parent_path()) {
    std::filesystem::create_directories(worker_path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open worker report path: " + path);
  }
  for (const auto &record : records) {
    WriteRecord(out, record);
    out << "\n";
  }
}

void WriteFinalReport(const std::string &path, const Options &opts, const std::string &command,
                      const std::vector<WorkerInfo> &workers,
                      const std::vector<BenchmarkRecord> &records) {
  const std::filesystem::path report_path(path);
  if (report_path.has_parent_path()) {
    std::filesystem::create_directories(report_path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open output report path: " + path);
  }

  int runtime_version = 0;
  int driver_version = 0;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  CHECK_CUDA(cudaDriverGetVersion(&driver_version));

  size_t skipped = 0;
  size_t invalid_roofline = 0;
  size_t noisy_baseline = 0;
  size_t adjacent_alarm = 0;
  for (const auto &record : records) {
    skipped += record.skipped ? 1 : 0;
    invalid_roofline += record.roofline_invalid_alarm ? 1 : 0;
    noisy_baseline += record.baseline_noise_alarm || record.baseline_drift_alarm ? 1 : 0;
    adjacent_alarm += record.adjacent_size_instability_alarm ? 1 : 0;
  }

  out << "{";
  out << "\"schema_version\":\"grouped_fp8_block_quantize_native_benchmark/v1\",";
  out << "\"primary_evidence_layer\":\"native_cxx_cuda_nvte_common_api\",";
  out << "\"framework_involvement\":\"none\",";
  out << "\"excluded_primary_frameworks\":[\"python3\",\"pytest\",\"torch\","
         "\"transformer_engine.pytorch\",\"jax\",\"GroupedLinear\",\"autograd\","
         "\"training_loop\"],";
  out << "\"command\":" << JsonEscape(command) << ",";
  out << "\"profile_enabled\":" << (opts.profile ? "true" : "false") << ",";
  out << "\"profile_after_warmup\":true,";
  out << "\"te_commit\":" << JsonEscape(RunCommand("git rev-parse HEAD")) << ",";
  out << "\"te_branch\":" << JsonEscape(RunCommand("git rev-parse --abbrev-ref HEAD")) << ",";
  out << "\"build_mode\":"
#ifdef NDEBUG
      << "\"Release\",";
#else
      << "\"Debug\",";
#endif
  out << "\"cuda_runtime_version\":" << runtime_version << ",";
  out << "\"cuda_driver_version\":" << driver_version << ",";
  out << "\"container_image\":"
      << JsonEscape(!GetEnv("SLURM_CONTAINER_IMAGE").empty()
                        ? GetEnv("SLURM_CONTAINER_IMAGE")
                        : (!GetEnv("PYXIS_CONTAINER_IMAGE").empty() ? GetEnv("PYXIS_CONTAINER_IMAGE")
                                                                     : "unknown"))
      << ",";
  out << "\"environment\":{";
  out << "\"cuda_visible_devices\":" << JsonEscape(GetEnv("CUDA_VISIBLE_DEVICES")) << ",";
  out << "\"slurm_job_gpus\":" << JsonEscape(GetEnv("SLURM_JOB_GPUS")) << ",";
  out << "\"slurm_gpus_on_node\":" << JsonEscape(GetEnv("SLURM_GPUS_ON_NODE"));
  out << "},";
  out << "\"options\":{";
  out << "\"dims\":";
  WriteVector(out, opts.dims);
  out << ",\"output_modes\":";
  WriteStringVector(out, opts.output_modes);
  out << ",\"layouts\":";
  WriteStringVector(out, opts.layouts);
  out << ",\"num_groups\":";
  WriteVector(out, opts.num_groups);
  out << ",\"rows_sweep\":";
  WriteVector(out, opts.rows_sweep);
  out << ",\"cols\":";
  WriteVector(out, opts.cols);
  out << ",\"dtypes\":";
  WriteStringVector(out, opts.dtypes);
  out << ",\"warmup\":" << opts.warmup;
  out << ",\"iterations\":" << opts.iterations;
  out << ",\"samples\":" << opts.samples;
  out << ",\"min_sample_ms\":" << opts.min_sample_ms;
  out << ",\"adaptive_iterations\":" << (opts.adaptive_iterations ? "true" : "false");
  out << ",\"shard_across_gpus\":" << (opts.shard_across_gpus ? "true" : "false");
  out << "},";
  out << "\"sharding\":{";
  out << "\"scheduler_allocated_gpu_count\":"
      << std::max(ParseSlurmGpuCount(GetEnv("SLURM_JOB_GPUS")),
                  ParsePositiveIntString(GetEnv("SLURM_GPUS_ON_NODE")))
      << ",";
  out << "\"visible_gpu_count\":";
  int visible_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&visible_count));
  out << visible_count << ",";
  out << "\"selected_devices\":[";
  for (size_t i = 0; i < workers.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << workers[i].cuda_device_ordinal;
  }
  out << "],";
  out << "\"sharding_strategy\":\"round_robin_independent_cases_by_case_id\",";
  out << "\"merge_validation\":\"all worker records sorted by case_id into one JSON report\"";
  out << "},";
  out << "\"workers\":[";
  for (size_t i = 0; i < workers.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "{";
    out << "\"worker_id\":" << workers[i].worker_id << ",";
    out << "\"cuda_device_ordinal\":" << workers[i].cuda_device_ordinal << ",";
    out << "\"gpu_name\":" << JsonEscape(workers[i].gpu_name) << ",";
    out << "\"worker_output_path\":" << JsonEscape(workers[i].worker_output_path) << ",";
    out << "\"assigned_case_ids\":";
    WriteVector(out, workers[i].assigned_case_ids);
    out << "}";
  }
  out << "],";
  out << "\"summary\":{";
  out << "\"record_count\":" << records.size() << ",";
  out << "\"skipped_record_count\":" << skipped << ",";
  out << "\"roofline_invalid_alarm_count\":" << invalid_roofline << ",";
  out << "\"baseline_noise_or_drift_alarm_count\":" << noisy_baseline << ",";
  out << "\"adjacent_size_instability_alarm_count\":" << adjacent_alarm;
  out << "},";
  out << "\"records\":[";
  for (size_t i = 0; i < records.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    WriteRecord(out, records[i]);
  }
  out << "]";
  out << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options opts = ParseOptions(argc, argv);
    const std::string command = ReconstructCommand(argc, argv);
    std::vector<CaseSpec> cases = BuildCases(opts);
    const std::vector<int> devices = SelectDevices(opts);

    std::vector<WorkerInfo> workers(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
      workers[i].worker_id = static_cast<int>(i);
      workers[i].cuda_device_ordinal = devices[i];
      workers[i].gpu_name = DeviceName(devices[i]);
      workers[i].worker_output_path =
          opts.output_path + ".worker" + std::to_string(workers[i].worker_id) + ".jsonl";
    }

    std::vector<std::vector<CaseSpec>> cases_by_worker(devices.size());
    for (const auto &spec : cases) {
      const size_t worker = static_cast<size_t>(spec.id) % devices.size();
      cases_by_worker[worker].push_back(spec);
      workers[worker].assigned_case_ids.push_back(spec.id);
    }

    std::vector<std::vector<BenchmarkRecord>> records_by_worker(devices.size());
    std::vector<std::string> errors(devices.size());
    std::vector<std::thread> threads;
    threads.reserve(devices.size());
    for (size_t worker_idx = 0; worker_idx < devices.size(); ++worker_idx) {
      threads.emplace_back([&, worker_idx]() {
        try {
          CHECK_CUDA(cudaSetDevice(devices[worker_idx]));
          cudaStream_t stream = nullptr;
          CHECK_CUDA(cudaStreamCreate(&stream));
          for (const auto &spec : cases_by_worker[worker_idx]) {
            records_by_worker[worker_idx].push_back(
                RunCase(opts, spec, static_cast<int>(worker_idx), devices[worker_idx],
                        workers[worker_idx].gpu_name, stream));
          }
          CHECK_CUDA(cudaStreamDestroy(stream));
          WriteWorkerJsonl(workers[worker_idx].worker_output_path, records_by_worker[worker_idx]);
        } catch (const std::exception &ex) {
          errors[worker_idx] = ex.what();
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }
    for (size_t i = 0; i < errors.size(); ++i) {
      if (!errors[i].empty()) {
        throw std::runtime_error("Worker " + std::to_string(i) + " failed: " + errors[i]);
      }
    }

    std::vector<BenchmarkRecord> records;
    for (auto &worker_records : records_by_worker) {
      for (auto &record : worker_records) {
        records.push_back(std::move(record));
      }
    }
    std::sort(records.begin(), records.end(),
              [](const BenchmarkRecord &lhs, const BenchmarkRecord &rhs) {
                return lhs.case_id < rhs.case_id;
              });
    MarkAdjacentInstability(&records);
    WriteFinalReport(opts.output_path, opts, command, workers, records);
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "benchmark_grouped_fp8_block_quantize failed: " << ex.what() << "\n";
    return 1;
  }
}
