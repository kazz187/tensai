//go:build goexperiment.simd && arm64 && go1.27

package tensai

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The weight-gradient tiles accumulate in the order of a's rows, as the
// row kernel does, so the two agree to the bit on finite inputs: over
// output rows short of a tile, a column tail short of 16, a contraction
// deeper than dotTileK, zeros in a, and a worker's slice of the output
// rows that starts off a multiple of four.
func TestDotTARowsTiledMatchesAxpy(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, sh := range [][5]int{
		// rows of a and b, output rows, output columns, and the worker's lo..hi
		{64, 4, 16, 0, 4}, {33, 7, 17, 0, 7}, {300, 12, 48, 0, 12}, {1100, 9, 70, 0, 9},
		{1, 32, 64, 0, 32}, {257, 5, 31, 0, 5}, {600, 128, 144, 0, 128}, {700, 40, 40, 6, 31},
	} {
		r, i, j, lo, hi := sh[0], sh[1], sh[2], sh[3], sh[4]
		a, b := NewMatrix(r, i), NewMatrix(r, j)
		for k := range a.Data {
			if rng.IntN(4) > 0 { // leave zeros for the row kernel to skip
				a.Data[k] = Float(rng.NormFloat64())
			}
		}
		for k := range b.Data {
			b.Data[k] = Float(rng.NormFloat64())
		}
		// Both kernels add into what out holds.
		got, want := RandomMatrix(i, j, rng), NewMatrix(i, j)
		copy(want.Data, got.Data)
		dotTARows(got, a, b, lo, hi)
		dotTARowsAxpy(want, a, b, lo, hi, 0)
		for k := range got.Data {
			if math.Float32bits(got.Data[k]) != math.Float32bits(want.Data[k]) && got.Data[k] != want.Data[k] {
				t.Fatalf("%v: element %d = %v, row kernel %v", sh, k, got.Data[k], want.Data[k])
			}
		}
	}
}
