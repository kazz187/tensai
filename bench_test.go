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
		p := gemmProducts(rng, m, k, n)
		for _, name := range []string{"NN", "TN", "NT"} {
			benchGEMM(b, name, m, k, n, p[name])
		}
	}
}

// BenchmarkGEMMSmall runs the products where the register tiles have
// little to work with: the forward product of one to three rows -- a
// single position at inference, or the rows a tile leaves -- the weight
// gradient of a single sample, whose contraction is one row long, and the
// input gradient over a contraction of 16. Run it with -cpu 1 as well.
func BenchmarkGEMMSmall(b *testing.B) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, c := range []struct {
		name    string
		m, k, n int
	}{
		{"NN", 1, 128, 128}, {"NN", 1, 512, 512}, {"NN", 3, 512, 512},
		{"TN", 1, 128, 128}, {"TN", 1, 512, 512}, {"TN", 1, 64, 1024},
		{"NT", 4, 16, 16}, {"NT", 16, 16, 16},
	} {
		benchGEMM(b, c.name, c.m, c.k, c.n, gemmProducts(rng, c.m, c.k, c.n)[c.name])
	}
}

// gemmProducts returns the three products BenchmarkGEMM names, over x of
// m x k, w of k x n and g of m x n: NN is x*w, TN x^T*g and NT g*w^T, so
// m is the contraction of TN and n that of NT.
func gemmProducts(rng *rand.Rand, m, k, n int) map[string]func() error {
	x, w := RandomMatrix(m, k, rng), RandomMatrix(k, n, rng)
	y, g := NewMatrix(m, n), RandomMatrix(m, n, rng)
	gw, gx := NewMatrix(k, n), NewMatrix(m, k)
	return map[string]func() error{
		"NN": func() error { return DotInto(y, x, w) },
		"TN": func() error { return DotTAInto(gw, x, g) },
		"NT": func() error { return DotTBInto(gx, g, w) },
	}
}

// benchGEMM runs one product as a sub-benchmark named by its shape.
func benchGEMM(b *testing.B, name string, m, k, n int, fn func() error) {
	b.Run(fmt.Sprintf("%s/%dx%dx%d", name, m, k, n), func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := fn(); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
