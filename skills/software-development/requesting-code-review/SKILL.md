---
name: requesting-code-review
description: >
  Industrial-grade optimization and verification pipeline. Shifts from basic
  bug-fixing to high-performance auditing, focusing on VRAM efficiency for 
  NVIDIA GB10 (122GB), compute throughput, and rigorous verification loops.
version: 3.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [industrial-optimization, vram-efficiency, gb10, performance-audit, verification-loop, quant-libraries]
    related_skills: [subagent-driven-development, writing-plans, test-driven-development, github-code-review]
---

# Industrial Grade Optimization & Verification

High-performance verification pipeline designed for quant libraries and LLM kernels. 
Focuses on maximizing hardware utilization of NVIDIA GB10 (122GB VRAM) and ensuring 
mathematical correctness through rigorous verification.

**Core principle:** "Performance is a correctness issue." Sub-optimal VRAM usage 
or compute bottlenecks are treated as critical defects.

## Target Flow: Code Audit -> Resource Optimization -> Verification Loop -> Delivery

### Step 1 — High-Fidelity Diff & Context Extraction
Identify changes and extract the surrounding architectural context.
```bash
git diff --cached
```
For large changes, split by file. If the code involves CUDA/Triton/C++/PyTorch, 
extract the relevant kernel signatures and memory allocation calls.

### Step 2 — Industrial Audit (Performance & VRAM)
Analyze the diff specifically for "Industrial Grade" bottlenecks.

#### VRAM Efficiency (GB10 Focus)
- **Allocation Patterns:** Check for frequent `cudaMalloc`/`cudaFree` inside loops. Flag for use of memory pools or pre-allocation.
- **Tensors/Buffers:** Audit for redundant copies (`.clone()`, `.to(device)`). Ensure `inplace` operations are used where safe.
- **Dtype Precision:** Verify use of `bf16` or `fp8` for GB10 optimization. Flag `fp32` in paths where precision is not critical.
- **Memory Alignment:** Check for aligned memory access patterns to maximize coalescing.
- **VRAM Budgeting:** For GB10 (122GB), verify if large tensors can be sharded or if activation checkpointing is needed for larger batch sizes.

#### Compute Throughput
- **Kernel Fusion:** Look for sequences of element-wise operations that should be fused into a single Triton/CUDA kernel.
- **Parallelism:** Audit for GIL bottlenecks in Python or lack of `cudaStream` concurrency in C++.
- **Indexing:** Flag non-contiguous memory access (strided access) that degrades throughput.

### Step 3 — Mathematical & Logical Verification
Static check for correctness in high-performance code:
- **Numerical Stability:** Check for `eps` additions in divisions, `log-sum-exp` patterns, and overflow guards.
- **Shape Integrity:** Verify tensor shape assertions at boundaries of complex transformations.
- **Race Conditions:** Audit atomic operations and synchronization primitives in multi-GPU/multi-thread paths.

### Step 4 — Baseline Benchmarking & Regression
Run existing tests AND performance benchmarks.
1. **Functional Correctness:** Run existing test suites (pytest, cargo test, etc.).
2. **Performance Baseline:** Run `time` or specific micro-benchmarks on the current branch.
3. **VRAM Profiling:** Use `nvidia-smi` or PyTorch `memory_summarize` to capture peak VRAM usage.

**Success Criteria:** 
- No new functional regressions.
- VRAM usage $\le$ Baseline (or justified increase).
- Throughput (tokens/sec or ops/sec) $\ge$ Baseline.

### Step 5 — Independent Optimization Reviewer
Call `delegate_task` with a specialized "Performance Architect" persona.

```python
delegate_task(
    goal="""You are a High-Performance Computing (HPC) Architect specializing in NVIDIA GB10. 
Review the diff for 'Industrial Grade' optimization. 
Your goal is to maximize VRAM efficiency and compute throughput.

CRITICAL FAILURES (passed=false):
- VRAM Leaks: Memory not freed or growing linearly.
- Suboptimal Precision: Using fp32 where bf16/fp8 is standard for GB10.
- Synchronization Bottlenecks: Unnecessary cudaDeviceSynchronize() in hot paths.
- Redundant Memory Transfers: Host-to-Device transfers inside loops.
- Numerical Instability: Potential NaNs/Infs in quant kernels.

OPTIMIZATION OPPORTUNITIES (suggestions):
- Kernel Fusion: Opportunities to merge ops.
- Vectorization: Use of SIMD or tensor cores.
- Cache Locality: Improving L1/L2 cache hit rates.

Return ONLY JSON:
{
  "passed": bool,
  "critical_bottlenecks": [],
  "numerical_risks": [],
  "optimization_suggestions": [],
  "estimated_vram_impact": "description of impact on GB10's 122GB",
  "summary": "concise verdict"
}""",
    context="Industrial Performance Audit. Return only JSON.",
    toolsets=["terminal"]
)
```

### Step 6 — Verification Loop (The "Fix-Measure-Verify" Cycle)
If the audit fails or identifies critical bottlenecks, enter the loop:

1. **Fix:** Implement the optimization (e.g., replace `.clone()` with a pre-allocated buffer).
2. **Measure:** Re-run Step 4 (VRAM profile + Benchmark).
3. **Verify:** Re-run Step 5 (Reviewer audit).

**Maximum 3 cycles.** If the performance target is not met after 3 attempts, escalate to the user with a detailed performance report.

### Step 7 — Industrial Delivery
Upon passing all gates:
```bash
git add -A && git commit -m "[optimized] <description of performance gain>"
```
The `[optimized]` prefix signals that the code has passed the Industrial Grade Optimization pipeline.

## Reference: GB10 Optimization Patterns
- **Avoid:** `torch.cat` in loops $\rightarrow$ **Use:** Pre-allocated tensors + slicing.
- **Avoid:** `float32` for weights $\rightarrow$ **Use:** `bfloat16` or `int8/fp8` quant.
- **Avoid:** Small, frequent kernel launches $\rightarrow$ **Use:** Kernel fusion/Triton.
- **Avoid:** Synchronous `get()` calls $\rightarrow$ **Use:** Asynchronous streams.
