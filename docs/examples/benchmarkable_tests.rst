..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Benchmarkable Tests
===================

Transformer Engine benchmarkable tests let a correctness test and a
performance benchmark share the same setup, reference, evaluation, and
validation code. A test module exposes one or more
``benchmarks.benchmarkable.BenchmarkCase`` objects. Pytest calls
``case.run_check()`` for correctness, while the benchmark runner calls the
same case ``reference`` and ``evaluate`` functions for timing.

Contract
--------

A benchmarkable case must provide:

* ``case_id``: stable identifier that should not change across releases.
* ``framework``: ``pytorch`` or ``jax``.
* ``component`` and ``operation``: coarse grouping for dashboards.
* ``params``: JSON-serializable independent workload axes, such as
  ``M``, ``K``, ``N``, tensor count, sequence length, dtype, or top-k.
* ``setup()``: deterministic input construction.
* ``evaluate(state)``: Transformer Engine path under measurement.
* ``reference(state)``: optional reference or baseline path.
* ``check(state)``: correctness validation used by the unit test.
* ``metrics``: optional functions that derive throughput from the measured
  median time.

Example modules should expose either ``BENCHMARKABLE_CASES`` or
``iter_benchmark_cases()``. Discovery imports modules and reads those
registries; it does not execute pytest test functions.

Pytest Marker
-------------

PyTorch and JAX tests register a ``benchmarkable`` marker. A representative
test looks like:

.. code-block:: python

   @pytest.mark.benchmarkable
   @pytest.mark.parametrize("case", BENCHMARKABLE_CASES, ids=lambda case: case.case_id)
   def test_benchmarkable_example(case):
       try:
           case.run_check()
       except BenchmarkSkip as exc:
           pytest.skip(str(exc))

The skip path is important for GPU, architecture, or optional-extension
requirements. Unsupported hardware is a coverage skip, not an infrastructure
failure.

Runner
------

Run selected cases from the repository root:

.. code-block:: shell

   python3 -m benchmarks.benchmarkable.runner \
       --framework pytorch \
       --module tests.pytorch.test_benchmarkable_permutation \
       --output-dir /tmp/te-benchmarkable-pytorch \
       --warmup 5 \
       --iterations 20

Use ``--report-file`` when a harness expects the top-level JSON report at a
fixed path:

.. code-block:: shell

   python3 -m benchmarks.benchmarkable.runner \
       --framework pytorch \
       --module tests.pytorch.test_benchmarkable_permutation \
       --output-dir /tmp/te-benchmarkable-pytorch \
       --report-file "${ORCHESTRA_BENCHMARK_RAW_REPORT}" \
       --warmup 5 \
       --iterations 20

The JAX example uses the same runner:

.. code-block:: shell

   python3 -m benchmarks.benchmarkable.runner \
       --framework jax \
       --module tests.jax.test_benchmarkable_softmax \
       --output-dir /tmp/te-benchmarkable-jax \
       --warmup 5 \
       --iterations 20

The runner writes:

* ``benchmark_report.json``: full machine-readable report with metadata.
* ``benchmark_records.jsonl``: one record per case variant.
* ``benchmark_summary.csv``: compact table for quick inspection.

Each record includes framework, case id, operation, tags, params, variant
(``reference`` or ``evaluation``), timing statistics, samples, optional
throughput metrics, and skip/error status. The report includes the command,
git metadata, framework versions, device metadata, scheduler allocation
metadata, sharding strategy, worker commands, and merge validation.

Profiling
---------

Use ``--profile`` to start CUDA profiler collection after warmup iterations
and stop it immediately after timed iterations:

.. code-block:: shell

   python3 -m benchmarks.benchmarkable.runner \
       --framework pytorch \
       --module tests.pytorch.test_benchmarkable_permutation \
       --case-id pytorch.moe_permute.index.tokens1024.hidden256.experts8.topk1 \
       --output-dir /tmp/te-benchmarkable-profile \
       --warmup 10 \
       --iterations 20 \
       --profile \
       --shard-devices off

Do not wrap this command in Nsight tools when recording a benchmark request.
The cluster profiler wrapper supplies Nsight Systems or Nsight Compute and
uses the CUDA profiler API range emitted by the runner.

DLcluster And Weekly Runs
-------------------------

GPU validation and performance runs should use the configured DLcluster
container workflow through Orchestra cluster wrappers. The repository runner
does not require user-specific home-directory state beyond normal repository
authentication and the Orchestra-managed workspace configuration.

For independent single-GPU cases, ``--shard-devices auto`` is the default.
When Slurm or the container sets ``SLURM_STEP_GPUS``, ``SLURM_JOB_GPUS``, or
``CUDA_VISIBLE_DEVICES`` to multiple allocated devices, the runner starts one
worker per allocated GPU. Each worker receives an isolated
``CUDA_VISIBLE_DEVICES`` value, writes to ``worker_XXX/``, bounds common CPU
thread pools to one thread, and the parent deterministically merges records
into the top-level report. Profiling runs disable sharding by default so
Nsight attribution stays clear; use the unprofiled command for the full
sharded sweep.

Historical Comparison
---------------------

Compare a current report against a compatible prior report:

.. code-block:: shell

   python3 -m benchmarks.benchmarkable.compare \
       --baseline /path/to/previous/benchmark_report.json \
       --current /tmp/te-benchmarkable-pytorch/benchmark_report.json \
       --output /tmp/te-benchmarkable-pytorch/comparison.json \
       --fail-on-regression

Records match by framework, case id, operation, variant, and JSON params.
Device metadata must be compatible. The default regression rule flags current
median time greater than baseline by more than 5 percent and more than
0.01 ms. A case can override those thresholds through
``regression_threshold`` metadata.

Toolchain Decision
------------------

The benchmarkable-test runner is TE-owned because it must support both
PyTorch and JAX, explicit GPU synchronization, durable JSON artifacts,
DLcluster sharding metadata, and historical comparisons that are stable
outside pytest collection. ``torch.utils.benchmark`` remains useful inside
PyTorch-specific benchmark scripts, and existing scripts may continue to use
it. ``pytest-benchmark`` was not chosen as the core because it ties benchmark
execution to pytest collection and does not solve JAX asynchronous dispatch
or the required artifact schema. Airspeed Velocity is useful for long-term
CPU-style benchmarking, but its environment management does not fit TE's GPU
container and Orchestra artifact flow for this first benchmarkable-test API.

Existing Benchmark Survey
-------------------------

Current benchmark scripts remain valid and should migrate incrementally:

* ``benchmarks/gemm/benchmark_gemm.py`` now supports
  ``--standard-output-dir`` in default/``--shapes`` mode. The standardized
  report records independent ``M``, ``K``, and ``N`` axes, precision,
  timing backend, pre-quantized mode, TFLOP/s, and the ``2*M*N*K`` FLOP
  denominator. Model-config mode should be migrated in a follow-up because
  it reports derived forward, dgrad, and wgrad groups.
* ``benchmarks/linear/*.py`` already use ``torch.utils.benchmark`` and should
  be wrapped as PyTorch benchmarkable cases once their input setup and
  forward/backward validation are shared with tests.
* ``benchmarks/attention/benchmark_attention.py`` has operation-specific
  shape and backend choices. It should migrate after attention unit tests
  expose reusable setup and reference functions.
* ``benchmarks/benchmark_rht_cast*.py`` and
  ``benchmarks/profile_rht_cast_swizzle_fusion.py`` should first separate
  setup, validation, and timing so the runner can emit skip/error records
  for unsupported Blackwell/NVFP4 paths.

When adding new benchmarkable cases, vary the independent workload axes that
can affect performance. For grouped tensors this includes elements per tensor
and group count. For GEMM this means ``M``, ``K``, and ``N`` rather than only
total FLOPs. For attention or softmax this means batch, heads, query length,
key/value length, dtype, mask/fusion type, and forward/backward mode.
