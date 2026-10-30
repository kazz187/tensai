//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
)

// gemmBenchShapes are (m, k, n): the products of a small transformer
// (width 128, four heads, FFN 512) over a batch of m token rows -- the QKV
// projection, the output projection, and the two FFN products -- and a
// square one.
var gemmBenchShapes = [][3]int{
	{1792, 128, 384}, {1792, 128, 128}, {1792, 128, 512}, {1792, 512, 128},
	{7168, 128, 384}, {7168, 128, 128}, {7168, 128, 512}, {7168, 512, 128},
	{28672, 128, 384}, {28672, 128, 128}, {28672, 128, 512}, {28672, 512, 128},
	{2048, 2048, 2048},
}

// gemmVariant switches the kernel choice for one benchmark run: "old" is
// matmulWGSL's kernels, "new" (and "fused", which differs only in the
// caller) the tiled ones as stridedMatMul picks them, and a tile entry
// name forces that tiling.
func gemmVariant(name string) func() {
	switch name {
	case "old":
		gemmOff = true
	case "new", "fused":
	default:
		for i, c := range gemmTiles {
			if c.entry() == name {
				gemmForce = i
			}
		}
	}
	return func() { gemmOff, gemmForce = false, -1 }
}

func gemmBenchVariants() []string {
	v := []string{"old", "new"}
	if os.Getenv("TENSAI_GEMM_BENCH_TILES") != "" {
		for _, c := range gemmTiles {
			v = append(v, c.entry())
		}
	}
	return v
}

// BenchmarkGemmKernel times device-resident products alone: no upload, no
// download but a four-element one to wait for the last product. Products
// are recorded sixteen to a submission, so the device is not kept waiting
// on the host, and the variants alternate round by round in one process,
// so a device shared with other work drifts under all of them alike.
// Compare medians over the rounds.
func BenchmarkGemmKernel(b *testing.B) {
	// With TENSAI_GEMM_BENCH_CHECKED set the tiled module keeps wgpu's
	// runtime checks, to see what dropping them buys.
	gemmChecked = os.Getenv("TENSAI_GEMM_BENCH_CHECKED") != ""
	defer func() { gemmChecked = false }()
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	b.Logf("adapter: %s", g.Adapter())
	rng := rand.New(rand.NewPCG(41, 0))
	rounds := 3
	for _, s := range gemmBenchShapes {
		m, k, n := s[0], s[1], s[2]
		x, err := g.Upload(randTensor(rng, m, k))
		if err != nil {
			b.Fatal(err)
		}
		w, err := g.Upload(randTensor(rng, k, n))
		if err != nil {
			b.Fatal(err)
		}
		flops := 2 * float64(m) * float64(k) * float64(n)
		for round := 0; round < rounds; round++ {
			for _, v := range gemmBenchVariants() {
				b.Run(fmt.Sprintf("m%dk%dn%d/%s/r%d", m, k, n, v, round), func(b *testing.B) {
					defer gemmVariant(v)()
					var last *Tensor
					for i := 0; i < b.N; i++ {
						if i%16 == 0 {
							if err := g.Flush(); err != nil {
								b.Fatal(err)
							}
							if err := g.BeginBatch(); err != nil {
								b.Fatal(err)
							}
						}
						if last != nil {
							last.Free()
						}
						if last, err = x.MatMul(w); err != nil {
							b.Fatal(err)
						}
					}
					if _, err := last.DownloadRange(0, 4); err != nil {
						b.Fatal(err)
					}
					last.Free()
					b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
				})
			}
		}
		x.Free()
		w.Free()
	}
}

// ffn runs x*W1, ReLU, *W2 on resident tensors: as three dispatches, or
// for "fused" as two Linear calls with the ReLU in the first one's
// epilogue.
func ffn(v string, x, w1, w2 *Tensor) (*Tensor, error) {
	if v == "fused" {
		h, err := x.Linear(w1, nil, ActReLU)
		if err != nil {
			return nil, err
		}
		defer h.Free()
		return h.Linear(w2, nil)
	}
	h, err := x.MatMul(w1)
	if err != nil {
		return nil, err
	}
	defer h.Free()
	a, err := h.Activate(ActReLU)
	if err != nil {
		return nil, err
	}
	defer a.Free()
	return a.MatMul(w2)
}

// BenchmarkGemmFFN times the transformer FFN on m rows the way an
// inference batch runs it: upload the rows, x*W1, ReLU, *W2, download,
// with the weights resident. rows/s is what the batch delivers.
func BenchmarkGemmFFN(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	b.Logf("adapter: %s", g.Adapter())
	rng := rand.New(rand.NewPCG(43, 0))
	w1, err := g.Upload(randTensor(rng, 128, 512))
	if err != nil {
		b.Fatal(err)
	}
	defer w1.Free()
	w2, err := g.Upload(randTensor(rng, 512, 128))
	if err != nil {
		b.Fatal(err)
	}
	defer w2.Free()
	for _, m := range []int{1600, 6400, 25600, 51200} {
		x := randTensor(rng, m, 128)
		for round := 0; round < 3; round++ {
			for _, v := range []string{"old", "new", "fused"} {
				b.Run(fmt.Sprintf("m%d/%s/r%d", m, v, round), func(b *testing.B) {
					defer gemmVariant(v)()
					for i := 0; i < b.N; i++ {
						X, err := g.Upload(x)
						if err != nil {
							b.Fatal(err)
						}
						Y, err := ffn(v, X, w1, w2)
						if err != nil {
							b.Fatal(err)
						}
						if _, err := Y.Download(); err != nil {
							b.Fatal(err)
						}
						X.Free()
						Y.Free()
					}
					b.ReportMetric(float64(m)*float64(b.N)/b.Elapsed().Seconds(), "rows/s")
				})
			}
		}
	}
}

// BenchmarkGemmFFNParts splits BenchmarkGemmFFN's time: the upload of the
// rows, the three kernels on resident data, and the download of the
// result, each waited for on its own.
func BenchmarkGemmFFNParts(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewPCG(47, 0))
	w1, _ := g.Upload(randTensor(rng, 128, 512))
	w2, _ := g.Upload(randTensor(rng, 512, 128))
	defer w1.Free()
	defer w2.Free()
	for _, m := range []int{6400, 25600} {
		x := randTensor(rng, m, 128)
		X, err := g.Upload(x)
		if err != nil {
			b.Fatal(err)
		}
		rows := func(b *testing.B) {
			b.ReportMetric(float64(m)*float64(b.N)/b.Elapsed().Seconds(), "rows/s")
		}
		b.Run(fmt.Sprintf("m%d/upload", m), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				U, err := g.Upload(x)
				if err != nil {
					b.Fatal(err)
				}
				if _, err := U.DownloadRange(0, 4); err != nil {
					b.Fatal(err)
				}
				U.Free()
			}
			rows(b)
		})
		for round := 0; round < 3; round++ {
			for _, v := range []string{"old", "new", "fused"} {
				b.Run(fmt.Sprintf("m%d/kernels/%s/r%d", m, v, round), func(b *testing.B) {
					defer gemmVariant(v)()
					for i := 0; i < b.N; i++ {
						Y, err := ffn(v, X, w1, w2)
						if err != nil {
							b.Fatal(err)
						}
						if _, err := Y.DownloadRange(0, 4); err != nil {
							b.Fatal(err)
						}
						Y.Free()
					}
					rows(b)
				})
			}
		}
		Y, _ := g.Upload(x) // the (m, 128) shape of the FFN's output
		b.Run(fmt.Sprintf("m%d/download", m), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := Y.Download(); err != nil {
					b.Fatal(err)
				}
			}
			rows(b)
		})
		Y.Free()
		X.Free()
	}
}
