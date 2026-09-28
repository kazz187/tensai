package tensai

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

func benchmarkDot(b *testing.B, size int) {
	rng := rand.New(rand.NewPCG(1, 0))
	x := RandomMatrix(size, size, rng)
	y := RandomMatrix(size, size, rng)
	b.SetBytes(int64(size * size * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Dot(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDot128(b *testing.B) { benchmarkDot(b, 128) }

func BenchmarkDot512(b *testing.B) { benchmarkDot(b, 512) }

func BenchmarkTranspose1024(b *testing.B) {
	src := NewMatrix(1024, 1024)
	dst := NewMatrix(1024, 1024)
	b.SetBytes(int64(len(src.Data) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TInto(dst, src); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGEMM runs the three products a training step spends its time
// in -- the forward x*w, the weight gradient x^T*g and the input gradient
// g*w^T -- over a small transformer's projections: 8192 tokens (64
// sequences of 128) of width 128, into 128 and into the 512 of its
// feed-forward and back. Run it with -cpu 1 for one core's rate, which is
// what a kernel change moves; the parallel split only multiplies it.
func BenchmarkGEMM(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sh := range [][3]int{{8192, 128, 128}, {8192, 128, 512}, {8192, 512, 128}, {512, 512, 512}} {
		m, k, n := sh[0], sh[1], sh[2]
		x, w := RandomMatrix(m, k, rng), RandomMatrix(k, n, rng)
		y, g := NewMatrix(m, n), RandomMatrix(m, n, rng)
		gw, gx := NewMatrix(k, n), NewMatrix(m, k)
		flops := 2 * float64(m) * float64(k) * float64(n)
		for _, p := range []struct {
			name string
			fn   func() error
		}{
			{"NN", func() error { return DotInto(y, x, w) }},
			{"TN", func() error { return DotTAInto(gw, x, g) }},
			{"NT", func() error { return DotTBInto(gx, g, w) }},
		} {
			b.Run(fmt.Sprintf("%s/%dx%dx%d", p.name, m, k, n), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := p.fn(); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}
