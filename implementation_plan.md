# SIREN — Implementation Plan

Build a production-grade repository demonstrating CSTE-compressed Apple Foundation Model inference targeting the Apple Neural Engine. The repo must demonstrate Apple ML engineer-level rigor: clean architecture, full benchmarks, scientific correctness, and Core ML awareness.

## Proposed Changes

### Core CSTE Engine (`siren/core/`)

#### [NEW] [circulant.py](file:///C:/Project/SIREN/siren/core/circulant.py)
- `BlockCirculantLinear(nn.Module)`: PyTorch module replacing `nn.Linear`
  - Stores only spectral coefficients `Λ ∈ C^{B×p}` (B blocks, p block size)
  - Forward: sign-flip → block partition → rfft → pointwise complex multiply → irfft → concat
  - Bernoulli diagonal sign-flip matrix `D` (fixed random ±1)
  - Learning rate scaling: `lr_eff = lr_base / sqrt(p)` exposed via `param_groups`
- `circulant_matvec()`: Standalone FFT-based matrix-vector multiply
- `dense_to_circulant()`: Convert dense weight matrix to block-circulant spectral coefficients

#### [NEW] [quantization.py](file:///C:/Project/SIREN/siren/core/quantization.py)
- `PhaseMagnitudeQuantizer`: Quantize spectral coefficients in polar form
  - Separate magnitude (4-bit log) and phase (4-bit uniform) quantization
  - STE via `x_hat = x + (quantize(x) - x).detach()` pattern
  - Support for 2/4/8-bit precision sweeps
- `STEQuantize(autograd.Function)`: Custom autograd for straight-through

#### [NEW] [fused_kernel.py](file:///C:/Project/SIREN/siren/core/fused_kernel.py)
- `FusedCirculantForward`: Combined FFT→multiply→IFFT→activation in single pass
- `IsingActivation`: Stochastic spin-flip activation `σ(x) = sign(x + τ·noise)`
- Temperature-scheduled `τ` with cosine annealing

---

### Model Architecture (`siren/models/`)

#### [NEW] [attention.py](file:///C:/Project/SIREN/siren/models/attention.py)
- `CirculantMultiHeadAttention`: Q/K/V projections as `BlockCirculantLinear`
- Rotary position embeddings (RoPE)
- Standard scaled dot-product attention (no circulant attention matrix)

#### [NEW] [feedforward.py](file:///C:/Project/SIREN/siren/models/feedforward.py)
- `CirculantFeedForward`: SwiGLU FFN with circulant gate/up/down projections

#### [NEW] [transformer.py](file:///C:/Project/SIREN/siren/models/transformer.py)
- `SIRENTransformer`: Full decoder-only transformer
  - Configurable depth, width, heads, block size
  - RMSNorm, residual connections
  - `from_dense()` classmethod for distillation from pretrained weights
  - `param_report()` for compression statistics

---

### Compression Pipeline (`siren/compression/`)

#### [NEW] [distillation.py](file:///C:/Project/SIREN/siren/compression/distillation.py)
- `FrobeniusDistiller`: Dense→CSTE weight conversion with auxiliary Frobenius loss
  - `L_total = L_task + λ · ||W_dense - W_cste||_F`
  - Progressive block size annealing (start p=64, end p=512)
- `compress_checkpoint()`: Convert any HuggingFace checkpoint to CSTE

#### [NEW] [profiler.py](file:///C:/Project/SIREN/siren/compression/profiler.py)
- `ModelProfiler`: Count parameters, FLOPs, memory for dense vs CSTE
- Output: compression ratio, FLOP reduction, memory per layer
- LaTeX table generation for paper-grade reporting

---

### ANE Targeting (`siren/ane/`)

#### [NEW] [power_model.py](file:///C:/Project/SIREN/siren/ane/power_model.py)
- `ANEPowerModel`: Estimate power draw based on op counts and memory access patterns
  - M5 Neural Engine: 16 cores, 35 TOPS, ~8W TDP
  - A19 Pro Neural Engine: 16 cores, 35 TOPS, ~5W TDP
  - Model: `P = P_compute × utilization + P_sram × sram_accesses + P_dram × dram_accesses`

#### [NEW] [sram_budget.py](file:///C:/Project/SIREN/siren/ane/sram_budget.py)
- `SRAMBudgetAnalyzer`: Map model weights to ANE SRAM tiers
  - L1 (per-core): ~48KB × 16 = 768KB
  - L2 (shared): ~4MB
  - Analysis: which layers fit in which tier, total DRAM spill

#### [NEW] [latency_model.py](file:///C:/Project/SIREN/siren/ane/latency_model.py)
- `ANELatencyModel`: Roofline model for inference latency
  - Compute-bound vs memory-bound analysis per layer
  - Total latency estimate for full forward pass

---

### Benchmarks ([benchmarks/](file:///C:/Project/March122026/v4_engine.py#371-385))

#### [NEW] [run_all.py](file:///C:/Project/SIREN/benchmarks/run_all.py)
- Orchestrator that runs all benchmarks and produces formatted report
- Generates markdown tables + ASCII charts

#### [NEW] [accuracy.py](file:///C:/Project/SIREN/benchmarks/accuracy.py)
- Perplexity comparison: dense baseline vs CSTE at p={128,256,512,1024}
- Frobenius norm tracking during distillation

#### [NEW] [throughput.py](file:///C:/Project/SIREN/benchmarks/throughput.py)
- Tokens/sec for dense vs CSTE on CPU/GPU
- Latency percentiles (p50, p95, p99)

#### [NEW] [memory.py](file:///C:/Project/SIREN/benchmarks/memory.py)
- Parameter count, checkpoint size, peak memory
- Compression ratio table across block sizes

#### [NEW] [power.py](file:///C:/Project/SIREN/benchmarks/power.py)
- ANE power model output for M5/A19
- TOPs/W comparison: dense vs CSTE

---

### Tests (`tests/`)

#### [NEW] [test_circulant.py](file:///C:/Project/SIREN/tests/test_circulant.py)
- Verify circulant matvec matches dense matvec (numerical tolerance)
- Verify sign-flip preserves gradient flow
- Verify block partitioning for non-square dimensions

#### [NEW] [test_quantization.py](file:///C:/Project/SIREN/tests/test_quantization.py)
- Verify STE gradient is identity
- Verify quantized values are in expected range
- Verify phase-magnitude decomposition is invertible

#### [NEW] [test_transformer.py](file:///C:/Project/SIREN/tests/test_transformer.py)
- Forward pass shape correctness
- Gradient flow through all layers
- `from_dense()` produces lower Frobenius error than random init

#### [NEW] [test_compression.py](file:///C:/Project/SIREN/tests/test_compression.py)
- Frobenius loss decreases during distillation
- Compression ratio matches theoretical prediction

---

### Project & Documentation

#### [NEW] [README.md](file:///C:/Project/SIREN/README.md)
- Apple-ML-engineer-grade professional README with architecture diagram, benchmarks, quickstart

#### [NEW] [docs/linkedin_article.md](file:///C:/Project/SIREN/docs/linkedin_article.md)
- Technical LinkedIn thought-leadership article on CSTE for on-device AI

---

## Verification Plan

### Automated Tests
```
cd C:\Project\SIREN
pip install -e ".[dev]"
python -m pytest tests/ -v
```

### Benchmark Execution
```
cd C:\Project\SIREN
python benchmarks/run_all.py
```
This will output compression ratios, throughput comparisons, ANE power estimates, and accuracy retention tables.

### Manual Verification
- Review README for professional quality
- Review LinkedIn article for technical accuracy and engagement quality
