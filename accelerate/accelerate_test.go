package accelerate

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai"
)

func randMatrix(rng *rand.Rand, rows, cols int) *tensai.Matrix {
	m := tensai.NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return m
}

// shapes are the products of a small transformer at one position (58 rows of width 128, FFN 512, a 387-wide input)
// plus edge shapes.
var shapes = [][3]int{{58, 387, 128}, {58, 128, 384}, {58, 128, 128}, {58, 128, 512}, {58, 512, 128}, {8, 128, 128},
	{1, 128, 1}, {3, 5, 7}, {146, 32, 146}, {1, 1, 1}}

func TestDotIntoMatchesTensai(t *testing.T) {
	if !Available() {
		t.Logf("Accelerate unavailable (%v); checking the fallback", Err())
	}
	rng := rand.New(rand.NewPCG(3, 4)) //nolint:gosec // test data
	for _, s := range shapes {
		m, k, n := s[0], s[1], s[2]
		a, b, bt := randMatrix(rng, m, k), randMatrix(rng, k, n), randMatrix(rng, n, k)
		for _, tb := range []bool{false, true} {
			got, want := tensai.NewMatrix(m, n), tensai.NewMatrix(m, n)
			var err error
			if tb {
				err = DotTBInto(got, a, bt)
				_ = tensai.DotTBIntoSerial(want, a, bt)
			} else {
				err = DotInto(got, a, b)
				_ = tensai.DotIntoSerial(want, a, b)
			}
			if err != nil {
				t.Fatal(err)
			}
			for i := range want.Data {
				tol := 1e-5 * math.Sqrt(float64(k)) * (1 + math.Abs(float64(want.Data[i])))
				if d := math.Abs(float64(got.Data[i] - want.Data[i])); d > tol || math.IsNaN(float64(got.Data[i])) {
					t.Fatalf("%dx%dx%d transB=%v: element %d = %v, tensai %v", m, k, n, tb, i, got.Data[i], want.Data[i])
				}
			}
		}
	}
}

// The same product gives the same bits on every call (the collector replays seeded games).
func TestDotIntoDeterministic(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	a, b := randMatrix(rng, 58, 512), randMatrix(rng, 512, 128)
	first := tensai.NewMatrix(58, 128)
	if err := DotInto(first, a, b); err != nil {
		t.Fatal(err)
	}
	for rep := 0; rep < 200; rep++ {
		got := tensai.NewMatrix(58, 128)
		if err := DotInto(got, a, b); err != nil {
			t.Fatal(err)
		}
		for i := range got.Data {
			if math.Float32bits(float32(got.Data[i])) != math.Float32bits(float32(first.Data[i])) {
				t.Fatalf("rep %d: element %d = %v, first call %v", rep, i, got.Data[i], first.Data[i])
			}
		}
	}
}

func TestDotIntoEmptyAndMismatch(t *testing.T) {
	out := tensai.NewMatrix(2, 3)
	out.Data[0] = 9
	if err := DotInto(out, tensai.NewMatrix(2, 0), tensai.NewMatrix(0, 3)); err != nil {
		t.Fatal(err)
	}
	for _, v := range out.Data {
		if v != 0 {
			t.Fatalf("empty contraction left %v", out.Data)
		}
	}
	if err := DotInto(tensai.NewMatrix(2, 3), tensai.NewMatrix(2, 4), tensai.NewMatrix(5, 3)); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

func BenchmarkGEMM(b *testing.B) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, s := range shapes[:6] {
		m, k, n := s[0], s[1], s[2]
		a, w, out := randMatrix(rng, m, k), randMatrix(rng, k, n), tensai.NewMatrix(m, n)
		name := func(p string) string { return p + "/" + itoa(m) + "x" + itoa(k) + "x" + itoa(n) }
		b.Run(name("accelerate"), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = DotInto(out, a, w)
			}
			b.ReportMetric(float64(2*m*k*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
		b.Run(name("tensai"), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = tensai.DotIntoSerial(out, a, w)
			}
			b.ReportMetric(float64(2*m*k*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b []byte
	for ; v > 0; v /= 10 {
		b = append([]byte{byte('0' + v%10)}, b...)
	}
	return string(b)
}
