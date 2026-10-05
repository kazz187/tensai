package tensai

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkGEMMInfer runs the serial forward products of a small policy
// transformer's inference on one position: 58 tokens (a typical position)
// and 146 (every slot filled) of width 128, through the input embedding
// from 387 features, the 128x128 projections and the 512-wide feed-forward.
// Self-play runs one such forward per game on each core, so this is the
// single-thread rate of DotIntoSerial at these shapes.
func BenchmarkGEMMInfer(b *testing.B) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, m := range []int{58, 146} {
		for _, kn := range [][2]int{{387, 128}, {128, 128}, {128, 512}, {512, 128}} {
			k, n := kn[0], kn[1]
			x, w, y := RandomMatrix(m, k, rng), RandomMatrix(k, n, rng), NewMatrix(m, n)
			b.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := DotIntoSerial(y, x, w); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}
