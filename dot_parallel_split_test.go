package tensai

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestDotIntoParallelSplitMatchesSerial checks that splitting a product across workers gives the serial result bit
// for bit, at row counts where rounding the chunk up leaves the last workers no rows (17 rows over 8 workers is
// chunks of 3). The rows are mostly zeros, which sends the leftover rows to the row kernel: before the empty
// ranges were skipped, a worker with lo > hi recomputed another worker's last rows at the same time on arm64.
func TestDotIntoParallelSplitMatchesSerial(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const k, n = 512, 128
	b := NewMatrix(k, n)
	for i := range b.Data {
		b.Data[i] = Float(rng.NormFloat64())
	}
	bt := NewMatrix(n, k)
	transposeData(bt.Data, b.Data, k, n)
	for _, rows := range []int{17, 25, 26, 27, 33, 34, 41} {
		a := NewMatrix(rows, k)
		for i := range a.Data {
			if rng.Float64() < 0.05 {
				a.Data[i] = Float(rng.NormFloat64())
			}
		}
		want, got := NewMatrix(rows, n), NewMatrix(rows, n)
		if err := DotIntoSerial(want, a, b); err != nil {
			t.Fatal(err)
		}
		wantTB, gotTB := NewMatrix(rows, n), NewMatrix(rows, n)
		if err := DotTBIntoSerial(wantTB, a, bt); err != nil {
			t.Fatal(err)
		}
		for rep := 0; rep < 200; rep++ {
			if err := DotInto(got, a, b); err != nil {
				t.Fatal(err)
			}
			if err := DotTBInto(gotTB, a, bt); err != nil {
				t.Fatal(err)
			}
			for i := range want.Data {
				if math.Float32bits(float32(got.Data[i])) != math.Float32bits(float32(want.Data[i])) {
					t.Fatalf("rows %d rep %d: DotInto element %d (row %d) = %v, serial %v", rows, rep, i, i/n, got.Data[i], want.Data[i])
				}
				if math.Float32bits(float32(gotTB.Data[i])) != math.Float32bits(float32(wantTB.Data[i])) {
					t.Fatalf("rows %d rep %d: DotTBInto element %d = %v, serial %v", rows, rep, i, gotTB.Data[i], wantTB.Data[i])
				}
			}
		}
	}
}
