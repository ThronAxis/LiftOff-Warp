# Changelog

## v1.0.0 (June 2026)

### Added
- **Core Layer**: `config.cuh`, `types.cuh`, `profile.cuh`
- **Primitive Layer**: 7 modules — intrinsics, reduce, scan, ballot, broadcast, sort, topk
- **Cooperative Groups**: `WarpTile<N>` abstraction with sort, scan, reduce, barrier
- **ML Kernels**: softmax, layernorm, RMS norm, dot product, attention, GELU, histogram
- **Composition Recipes**: fused LN+GELU, stream compaction, online softmax
- **Benchmark Suite**: 10 benchmarks with shared-memory baseline comparison
- **Correctness Tests**: 25 tests with edge cases (all-zero, all-same, uniform)
- **Kaggle Driver**: Python orchestration script with auto SM detection
- **Documentation**: Full project report, README

### Validated
- 25/25 correctness tests passed on Tesla T4
- 2.24× speedup over shared memory reductions
- 65:5 shuffle-to-shared PTX instruction ratio
- 622.7 GOPS peak throughput (GELU fused reduce)
- Zero sync barriers on 11/12 kernels

### Fixed
- Duplicate function definitions in compositions.cuh
- `thread_block_tile` default constructor (member initializer list)
- Shuffle participation bug (all lanes must execute `__shfl_xor_sync`)
- Unused `partner` variable warning in bitonic sort macro
