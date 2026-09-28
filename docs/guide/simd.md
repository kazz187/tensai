# SIMD Acceleration

tensai's fast kernels are AVX2 on amd64 and NEON (ARM's Advanced SIMD) on arm64, written with Go's experimental `simd/archsimd` package — still pure Go, no cgo, no assembly files.

```bash
GOEXPERIMENT=simd go build ./...
GOEXPERIMENT=simd go test -bench=Dot .
```

Requirements: amd64 with Go 1.26 or 1.27 (both `simd` API generations are supported via build tags), or arm64 with Go 1.27, whose `simd/archsimd` is the first to carry an arm64 half. Every other build — other architectures, older Go, or `GOEXPERIMENT` unset — uses the portable fallbacks automatically, with the same results up to rounding (and one exception in the matrix products' tiles, below). [Platforms](platforms.md) has the per-kernel breakdown.

## What is vectorized

Where the AVX2 kernels apply today, and where they still could:

- [x] Matmul (`Dot`/`DotInto`) — used by `Dense`, `Conv2D` (im2col product), `knn.Classifier` distances, and autograd `MatMul`; four rows by sixteen columns of the output stay in registers across the whole contraction, so a loaded row of the right operand feeds four output rows and the output is written once
- [x] ReLU / LeakyReLU forward & backward
- [x] Sigmoid / Tanh forward & backward (vectorized polynomial `exp`)
- [x] GELU forward & backward (vectorized `erf`)
- [x] LayerNorm forward & backward (vector row reductions)
- [x] Softmax / SoftmaxCrossEntropy exponentials and scaling
- [x] Adam / AdamW parameter update
- [x] SGD update (momentum form, same fused multiply-add loop as Adam)
- [x] Slice add & scale primitives (bias add, `Embedding` gradient scatter-add)
- [x] Transpose-free gradient matmuls (`DotTAInto`, `DotTBInto`) — `Dense`/`Conv2D` weight gradients no longer materialize `input^T` / `im2col^T`; on NEON the weight gradient runs in the same 4x16 tiles and the input gradient four rows against four weight rows at a time
- [x] Remaining transposes (`T`/`TInto`) — cache-blocked 32x32 tiles
- [x] Softmax backward row dot products (autograd) — fused AVX2 dot and Jacobian-vector accumulation
- [ ] MSE / BinaryCrossEntropy losses (BCE needs a vectorized `log`)
- [ ] Autograd element-wise backward passes (gradients accumulate with `+=`, so they need dedicated fused kernels)
- [ ] BatchNorm statistics (column-strided access needs a restructure)
- [ ] MaxPool2D window scan
- [ ] im2col / col2im gather-scatter (contiguous runs could use bulk copies)

The unchecked items are ordered roughly by expected impact; none of them show up prominently in training profiles today.

On finite inputs the tiles leave every result bit for bit as the row-at-a-time kernels did, up to the sign of a zero: each output element is still summed by the same fused multiply-adds in the same order, only alongside its neighbours. Infinities and NaNs are the exception, in the x·w and xᵀ·g products — their 4x16 tiles and the tall xᵀ·g register kernel. The row kernels, like the portable code, skip a zero in the left operand, where these kernels multiply it in, so a zero against an Inf or a NaN in the right operand makes the element NaN where the portable build leaves it finite — as the AVX2 kernels always have. The g·wᵀ dot tiles agree with the portable build here, since both multiply every pair. Past that, what the tiles change is the speed. On one Apple M5 core (`go test -bench GEMM -cpu 1`, 8192 tokens of width 128 into 128 and 512 and back):

| product | row kernels | tiles |
|---|---|---|
| `x * w` (`DotInto`) | 13 GFLOP/s | 79–84 GFLOP/s |
| `x^T * g` (`DotTAInto`) | 13 GFLOP/s | 72–78 GFLOP/s |
| `g * w^T` (`DotTBInto`) | 33–36 GFLOP/s | 68–86 GFLOP/s |

Shapes too small for the tiles — a product of fewer than four rows, the weight gradient of a single sample, an input gradient over fewer than twelve elements — go through the row kernels instead.

The int8/int4 quantized matmuls have their own AVX2 paths built on the 256-bit u8 x s8 pairwise multiply-add — see [Quantization](quantization.md).

## A live benchmark

`_example/plasma` animates a demoscene-style plasma in the terminal where the plasma function is a randomly weighted network (a CPPN) evaluated for every pixel of every frame as one batch. The status line shows the per-frame network time: 120x90 pixels runs at ~32 fps on the portable build and ~100 fps with `GOEXPERIMENT=simd` on the same machine.

```bash
go run ./_example/plasma
GOEXPERIMENT=simd go run ./_example/plasma
```
