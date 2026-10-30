//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai"
)

// gemmShapes are products the tiled kernels take (k and n multiples of
// four) with every edge the tiles have: m, n, and k short of, just past,
// and far from a block or slab boundary, m of one, and k of four.
var gemmShapes = [][2][]int{
	{{1, 4}, {4, 4}},
	{{1, 128}, {128, 512}},
	{{3, 8}, {8, 12}},
	{{17, 20}, {20, 36}},
	{{64, 16}, {16, 64}},
	{{65, 132}, {132, 68}},
	{{129, 64}, {64, 132}},
	{{127, 512}, {512, 128}},
	{{200, 128}, {128, 384}},
	{{130, 36}, {36, 260}},
	{{257, 48}, {48, 196}},
	{{2, 33, 16}, {2, 16, 20}},  // batched
	{{3, 70, 24}, {24, 72}},     // b broadcast over the batch
	{{40, 12}, {2, 12, 136}},    // a broadcast over the batch
	{{2, 1, 37, 8}, {3, 8, 28}}, // both broadcast
}

func checkProduct(t *testing.T, what string, got, want *tensai.Tensor, k int) {
	t.Helper()
	if len(got.Data) != len(want.Data) {
		t.Fatalf("%s: %d elements, want %d", what, len(got.Data), len(want.Data))
	}
	// f32 sums over k in a different order: allow for the rounding.
	tol := 1e-5 * math.Sqrt(float64(k)) * 8
	for i := range want.Data {
		w := float64(want.Data[i])
		if diff := math.Abs(float64(got.Data[i]) - w); diff > tol*(1+math.Abs(w)) {
			t.Fatalf("%s element %d: gpu=%v cpu=%v", what, i, got.Data[i], want.Data[i])
		}
	}
}

// TestGPUGemmTiles runs every tiling of the tiled kernels, and the
// default choice among them, on shapes with ragged edges, and checks the
// products against the CPU.
func TestGPUGemmTiles(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	if g.pipes.gemm == nil {
		t.Skip("tiled GEMM module did not compile on this device")
	}
	rng := rand.New(rand.NewPCG(31, 0))
	for tile := -1; tile < len(gemmTiles); tile++ {
		name := "auto"
		if tile >= 0 {
			name = gemmTiles[tile].entry()
		}
		gemmForce = tile
		for _, c := range gemmShapes {
			a, b := randTensor(rng, c[0]...), randTensor(rng, c[1]...)
			want, err := tensai.MatMul(a, b)
			if err != nil {
				t.Fatal(err)
			}
			got, err := g.MatMul(a, b)
			if err != nil {
				t.Fatalf("%s %v*%v: %v", name, c[0], c[1], err)
			}
			checkProduct(t, fmt.Sprintf("%s %v*%v", name, c[0], c[1]), got, want, c[0][len(c[0])-1])
		}
	}
	gemmForce = -1
}

// TestGPUGemmFallback checks the products the tiled kernels must leave to
// matmulWGSL's -- k or n not a multiple of four, k under four -- next to
// the ones they take.
func TestGPUGemmFallback(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(37, 0))
	for _, c := range [][2][]int{
		{{1, 1}, {1, 1}},
		{{5, 1}, {1, 8}},
		{{70, 2}, {2, 70}},
		{{130, 3}, {3, 132}},
		{{1, 3}, {3, 5}},
		{{129, 128}, {128, 130}}, // n not a multiple of four
		{{129, 130}, {130, 128}}, // k not a multiple of four
		{{256, 7}, {7, 512}},
	} {
		a, b := randTensor(rng, c[0]...), randTensor(rng, c[1]...)
		want, err := tensai.MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}
		got, err := g.MatMul(a, b)
		if err != nil {
			t.Fatalf("%v*%v: %v", c[0], c[1], err)
		}
		checkProduct(t, fmt.Sprintf("%v*%v", c[0], c[1]), got, want, c[0][len(c[0])-1])
	}
}

// TestGPULinear checks Linear's fused epilogue -- bias, each activation,
// neither -- against the CPU, on products the tiled kernels take and on
// ones they leave to MatMul, Binary, and Activate.
func TestGPULinear(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(53, 0))
	acts := []struct {
		name string
		act  []Act
		f    func(float64) float64
	}{
		{"none", nil, func(x float64) float64 { return x }},
		{"relu", []Act{ActReLU}, func(x float64) float64 { return math.Max(x, 0) }},
		{"tanh", []Act{ActTanh}, math.Tanh},
		{"sigmoid", []Act{ActSigmoid}, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }},
		{"gelu", []Act{ActGELU}, func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }},
	}
	for _, c := range []struct {
		x, w, bias []int // bias nil: no bias
	}{
		{[]int{200, 128}, []int{128, 512}, []int{512}},
		{[]int{77, 512}, []int{512, 128}, []int{1, 128}},
		{[]int{2, 33, 12}, []int{12, 20}, []int{20}},
		{[]int{1, 8}, []int{8, 4}, nil},
		{[]int{19, 7}, []int{7, 5}, []int{5}},     // falls back: k, n not multiples of four
		{[]int{70, 3}, []int{3, 132}, []int{132}}, // falls back: k under four
	} {
		x, w := randTensor(rng, c.x...), randTensor(rng, c.w...)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatal(err)
		}
		gw, err := g.Upload(w)
		if err != nil {
			t.Fatal(err)
		}
		var bias *tensai.Tensor
		var gb *Tensor
		if c.bias != nil {
			bias = randTensor(rng, c.bias...)
			if gb, err = g.Upload(bias); err != nil {
				t.Fatal(err)
			}
		}
		prod, err := tensai.MatMul(x, w)
		if err != nil {
			t.Fatal(err)
		}
		n := c.w[len(c.w)-1]
		for _, a := range acts {
			want := tensai.NewTensor(prod.Shape...)
			for i, v := range prod.Data {
				s := float64(v)
				if bias != nil {
					s += float64(bias.Data[i%n])
				}
				want.Data[i] = tensai.Float(a.f(s))
			}
			gy, err := gx.Linear(gw, gb, a.act...)
			if err != nil {
				t.Fatalf("%v*%v %s: %v", c.x, c.w, a.name, err)
			}
			got, err := gy.Download()
			if err != nil {
				t.Fatal(err)
			}
			gy.Free()
			checkProduct(t, fmt.Sprintf("Linear %v*%v+%v %s", c.x, c.w, c.bias, a.name), got, want, c.x[len(c.x)-1])
		}
		gx.Free()
		gw.Free()
		if gb != nil {
			gb.Free()
		}
	}

	x, _ := g.Upload(randTensor(rng, 4, 8))
	w, _ := g.Upload(randTensor(rng, 8, 12))
	b, _ := g.Upload(randTensor(rng, 8))
	defer x.Free()
	defer w.Free()
	defer b.Free()
	if _, err := x.Linear(w, b); err == nil {
		t.Fatal("Linear accepted a bias of 8 for 12 columns")
	}
	if _, err := x.Linear(w, nil, ActReLU, ActTanh); err == nil {
		t.Fatal("Linear accepted two activations")
	}
}
