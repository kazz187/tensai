//go:build goexperiment.simd && (amd64 || (arm64 && go1.27))

package tensai

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// The tiled kernel accumulates in the same order as the row kernel, so on
// finite inputs the two agree to the bit, including the column and row
// tails the tiles leave to the row kernel and inputs with zeros in a. The
// depths straddle both builds' dotTileK, so a tile also picks up a
// partial sum from the output.
func TestDotRowsTiledMatchesAxpy(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sh := range [][3]int{
		{4, 16, 16}, {7, 33, 17}, {12, 300, 48}, {9, 600, 70}, {32, 1, 64}, {5, 257, 31}, {64, 512, 144},
		{8, 1100, 32}, {13, 513, 50},
		// Every short tile (3, 2 and 1 rows) over depths that leave each
		// remainder of four, and past dotTileK.
		{1, 3, 16}, {2, 37, 48}, {3, 1029, 32}, {6, 5, 16}, {58, 387, 128}, {58, 512, 130},
	} {
		m, k, n := sh[0], sh[1], sh[2]
		a, b := NewMatrix(m, k), NewMatrix(k, n)
		for i := range a.Data {
			if rng.IntN(4) > 0 { // leave zeros for the row kernel to skip
				a.Data[i] = Float(rng.NormFloat64())
			}
		}
		for i := range b.Data {
			b.Data[i] = Float(rng.NormFloat64())
		}
		got, want := NewMatrix(m, n), NewMatrix(m, n)
		dotRows(got, a, b, 0, m)
		dotRowsAxpy(want, a, b, 0, m, 0)
		for i := range got.Data {
			if math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i]) && got.Data[i] != want.Data[i] {
				t.Fatalf("%dx%dx%d: element %d = %v, row kernel %v", m, k, n, i, got.Data[i], want.Data[i])
			}
		}
	}
}

func BenchmarkDotTall(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sh := range [][3]int{{428, 896, 4864}, {428, 4864, 896}, {428, 896, 896}} {
		m, k, n := sh[0], sh[1], sh[2]
		x, w, out := NewMatrix(m, k), NewMatrix(k, n), NewMatrix(m, n)
		for i := range x.Data {
			x.Data[i] = Float(rng.NormFloat64())
		}
		for i := range w.Data {
			w.Data[i] = Float(rng.NormFloat64())
		}
		b.Run(fmt.Sprintf("NN/%dx%dx%d", m, k, n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := DotInto(out, x, w); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		// The input gradient of x*w: out * w^T, back to x's shape.
		g := NewMatrix(m, k)
		b.Run(fmt.Sprintf("NT/%dx%dx%d", m, n, k), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := DotTBInto(g, out, w); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// A matrix whose data is shorter than its shape -- Matrix's fields are
// exported, and the shape checks compare only Rows and Cols -- must panic
// before a tile touches memory past the slice, including when the shape is
// large enough that the index arithmetic would wrap.
func TestDotRowsTiledShortDataPanics(t *testing.T) {
	for _, c := range []struct {
		name   string
		out, a *Matrix
		b      *Matrix
	}{
		{"short out", &Matrix{Rows: 4, Cols: 32, Data: make([]float32, 100)}, NewMatrix(4, 8), NewMatrix(8, 32)},
		{"short b", NewMatrix(4, 32), NewMatrix(4, 8), &Matrix{Rows: 8, Cols: 32, Data: make([]float32, 200)}},
		{"wrapping cols", &Matrix{Rows: 4, Cols: 1<<62 + 16, Data: make([]float32, 64)}, NewMatrix(4, 4),
			&Matrix{Rows: 4, Cols: 1<<62 + 16, Data: make([]float32, 64)}},
		{"wrapping rows", &Matrix{Rows: 1<<60 + 1, Cols: 16, Data: make([]float32, 16)},
			&Matrix{Rows: 1<<60 + 1, Cols: 16, Data: make([]float32, 16)}, NewMatrix(16, 16)},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", c.name)
				}
			}()
			dotRows(c.out, c.a, c.b, 0, c.a.Rows)
		}()
	}
}
