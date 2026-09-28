package tensai

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestDotShape(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(3, 2)
	for i := range a.Data {
		a.Data[i] = Float(i + 1)
	}
	for i := range b.Data {
		b.Data[i] = Float(i + 1)
	}
	out, err := Dot(a, b)
	if err != nil {
		t.Fatalf("Dot error: %v", err)
	}
	if out.Rows != 2 || out.Cols != 2 {
		t.Fatalf("expected 2x2, got %dx%d", out.Rows, out.Cols)
	}
	// a = [[1,2,3],[4,5,6]], b = [[1,2],[3,4],[5,6]]
	// a*b = [[22,28],[49,64]]
	checks := []struct {
		r, c int
		v    Float
	}{
		{0, 0, 22}, {0, 1, 28}, {1, 0, 49}, {1, 1, 64},
	}
	for _, ch := range checks {
		if got := out.At(ch.r, ch.c); got != ch.v {
			t.Errorf("At(%d,%d) = %g, want %g", ch.r, ch.c, got, ch.v)
		}
	}
}

func TestDotTAInto(t *testing.T) {
	rng := rand.New(rand.NewPCG(89, 0))
	// The last three are tall and narrow, which is the shape a
	// convolution's weight gradient has and the register-accumulating
	// kernel takes: a k that is not a multiple of four, a b that is not a
	// multiple of eight columns, and one wide enough to need two passes.
	for _, dims := range [][3]int{{5, 3, 4}, {8, 8, 8}, {17, 9, 13}, {64, 33, 5},
		{1024, 9, 8}, {600, 6, 5}, {1000, 7, 20}} {
		r, i, j := dims[0], dims[1], dims[2]
		a := RandomMatrix(r, i, rng)
		b := RandomMatrix(r, j, rng)
		// Sprinkle zeros to exercise the skip path.
		for k := 0; k < len(a.Data); k += 3 {
			a.Data[k] = 0
		}
		want, err := Dot(a.T(), b)
		if err != nil {
			t.Fatal(err)
		}
		got := NewMatrix(i, j)
		if err := DotTAInto(got, a, b); err != nil {
			t.Fatal(err)
		}
		for k := range want.Data {
			if diff := got.Data[k] - want.Data[k]; diff > 1e-5 || diff < -1e-5 {
				t.Fatalf("dims %v: element %d differs: %g vs %g", dims, k, got.Data[k], want.Data[k])
			}
		}
	}
	if err := DotTAInto(NewMatrix(2, 2), NewMatrix(3, 2), NewMatrix(4, 2)); err == nil {
		t.Error("row mismatch should be rejected")
	}
	if err := DotTAInto(NewMatrix(2, 2), NewMatrix(3, 2), NewMatrix(3, 5)); err == nil {
		t.Error("output shape mismatch should be rejected")
	}
}

// TestGEMMModes checks the three products against a float64 reference on
// every build -- portable, AVX2, NEON -- over shapes that reach each
// kernel's register tiles and the tails beside them: rows short of a
// tile, columns short of 16, a contraction deeper than a tile pass, and
// products big enough to be split across workers.
func TestGEMMModes(t *testing.T) {
	rng := rand.New(rand.NewPCG(97, 0))
	for _, sh := range [][3]int{
		{1, 5, 3}, {4, 16, 16}, {7, 33, 17}, {9, 1, 33}, {13, 513, 50},
		{64, 1100, 40}, {130, 70, 190}, {256, 128, 128},
	} {
		m, k, n := sh[0], sh[1], sh[2]
		a, b := RandomMatrix(m, k, rng), RandomMatrix(k, n, rng)
		for i := 0; i < len(a.Data); i += 5 {
			a.Data[i] = 0
		}
		want := make([]float64, m*n)
		bound := make([]float64, m*n) // sum of |a||b|, what rounding scales with
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				for p := 0; p < k; p++ {
					x, y := float64(a.Data[i*k+p]), float64(b.Data[p*n+j])
					want[i*n+j] += x * y
					bound[i*n+j] += math.Abs(x * y)
				}
			}
		}
		check := func(mode string, got *Matrix) {
			t.Helper()
			for i, w := range want {
				if diff := math.Abs(float64(got.Data[i]) - w); diff > 1e-5*bound[i]+1e-6 {
					t.Fatalf("%s %dx%dx%d: element %d = %v, want %v", mode, m, k, n, i, got.Data[i], w)
				}
			}
		}
		nn := NewMatrix(m, n)
		if err := DotInto(nn, a, b); err != nil {
			t.Fatal(err)
		}
		check("NN", nn)
		tn := NewMatrix(m, n)
		if err := DotTAInto(tn, a.T(), b); err != nil {
			t.Fatal(err)
		}
		check("TN", tn)
		nt := NewMatrix(m, n)
		if err := DotTBInto(nt, a, b.T()); err != nil {
			t.Fatal(err)
		}
		check("NT", nt)
	}
}

// TestGEMMEmptyContraction checks the three products over a contraction of
// length zero, which have nothing to add up: every element of the output
// is zero, whatever the buffer held before -- a tape hands out buffers the
// last step wrote. Some shapes reach the register tiles, whose loop over
// the contraction never runs, and the rest only the row kernels.
func TestGEMMEmptyContraction(t *testing.T) {
	nan := Float(math.NaN())
	for _, sh := range [][2]int{{4, 16}, {8, 32}, {5, 17}, {16, 16}, {1, 3}, {3, 40}} {
		m, n := sh[0], sh[1]
		for _, p := range []struct {
			name string
			fn   func(out *Matrix) error
		}{
			{"NN", func(out *Matrix) error { return DotInto(out, NewMatrix(m, 0), NewMatrix(0, n)) }},
			{"NN serial", func(out *Matrix) error { return DotIntoSerial(out, NewMatrix(m, 0), NewMatrix(0, n)) }},
			{"TN", func(out *Matrix) error { return DotTAInto(out, NewMatrix(0, m), NewMatrix(0, n)) }},
			{"NT", func(out *Matrix) error { return DotTBInto(out, NewMatrix(m, 0), NewMatrix(n, 0)) }},
		} {
			out := NewMatrix(m, n)
			for i := range out.Data {
				out.Data[i] = nan
			}
			if err := p.fn(out); err != nil {
				t.Fatal(err)
			}
			for i, v := range out.Data {
				if v != 0 {
					t.Fatalf("%s %dx0x%d: element %d = %v, want 0", p.name, m, n, i, v)
				}
			}
		}
	}
}

// TestGEMMEmptyOutput runs the three products into outputs with no rows or
// no columns, which have nothing to compute and must not trip over the
// operands they do get.
func TestGEMMEmptyOutput(t *testing.T) {
	for _, sh := range [][3]int{{0, 5, 16}, {4, 5, 0}, {0, 0, 0}, {0, 600, 3}, {4, 600, 0}} {
		m, k, n := sh[0], sh[1], sh[2]
		if err := DotInto(NewMatrix(m, n), NewMatrix(m, k), NewMatrix(k, n)); err != nil {
			t.Fatalf("NN %dx%dx%d: %v", m, k, n, err)
		}
		if err := DotTAInto(NewMatrix(m, n), NewMatrix(k, m), NewMatrix(k, n)); err != nil {
			t.Fatalf("TN %dx%dx%d: %v", m, k, n, err)
		}
		if err := DotTBInto(NewMatrix(m, n), NewMatrix(m, k), NewMatrix(n, k)); err != nil {
			t.Fatalf("NT %dx%dx%d: %v", m, k, n, err)
		}
	}
}

// TestDotTAIntoTallOverwrites checks the tall, narrow x^T*g path (a.Rows >=
// 512 with a result small enough for the register kernel) overwrites its
// output: the portable kernel accumulates, so a stale output -- a tape
// buffer reused from the step before, or the same product run twice --
// used to be added into the gradient.
func TestDotTAIntoTallOverwrites(t *testing.T) {
	rng := rand.New(rand.NewPCG(83, 0))
	nan := Float(math.NaN())
	for _, sh := range [][3]int{{600, 8, 8}, {512, 9, 4}, {1024, 16, 16}, {513, 3, 5}} {
		r, i, j := sh[0], sh[1], sh[2]
		a := RandomMatrix(r, i, rng)
		b := RandomMatrix(r, j, rng)
		out := NewMatrix(i, j)
		for k := range out.Data {
			out.Data[k] = nan
		}
		for run := 0; run < 2; run++ {
			if err := DotTAInto(out, a, b); err != nil {
				t.Fatal(err)
			}
			for p := 0; p < i; p++ {
				for q := 0; q < j; q++ {
					var want float64
					for s := 0; s < r; s++ {
						want += float64(a.Data[s*i+p]) * float64(b.Data[s*j+q])
					}
					got := float64(out.Data[p*j+q])
					if math.IsNaN(got) || math.Abs(got-want) > 1e-3*(1+math.Abs(want)) {
						t.Fatalf("%dx%d^T*%dx%d run %d: out[%d,%d] = %v, want %v", r, i, r, j, run, p, q, got, want)
					}
				}
			}
		}
	}
}

func TestDotVecAxpy(t *testing.T) {
	rng := rand.New(rand.NewPCG(81, 0))
	for _, n := range []int{0, 1, 7, 8, 15, 16, 64, 127, 1000} {
		a := make([]Float, n)
		b := make([]Float, n)
		for i := range a {
			a[i] = Float(rng.NormFloat64())
			b[i] = Float(rng.NormFloat64())
		}
		var want float64
		for i := range a {
			want += float64(a[i]) * float64(b[i])
		}
		got := float64(DotVec(a, b))
		if diff := math.Abs(got - want); diff > 1e-3*(1+math.Abs(want)) {
			t.Fatalf("DotVec n=%d: got %v want %v", n, got, want)
		}

		y := make([]Float, n)
		copy(y, b)
		Axpy(0.5, a, y)
		for i := range y {
			want := b[i] + 0.5*a[i]
			if diff := math.Abs(float64(y[i] - want)); diff > 1e-5 {
				t.Fatalf("Axpy n=%d elem %d: got %v want %v", n, i, y[i], want)
			}
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	DotVec(make([]Float, 3), make([]Float, 4))
}

func TestNewMatrixFromInts(t *testing.T) {
	m, err := NewMatrixFromInts(2, 2, []int{0, 1, 65535, 3})
	if err != nil {
		t.Fatal(err)
	}
	if m.At(1, 0) != 65535 {
		t.Fatalf("got %g", m.At(1, 0))
	}
	// 1<<25 + 1 is the first odd integer float32 cannot represent.
	if _, err := NewMatrixFromInts(1, 1, []int{1<<25 + 1}); err == nil {
		t.Fatal("expected exactness error")
	}
	if _, err := NewMatrixFromInts(2, 2, []int{1}); err == nil {
		t.Fatal("expected length error")
	}
}
