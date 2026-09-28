//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/dims"
	"github.com/mattn/tensai/internal/kernels"
)

// checkClose compares a downloaded device result against a CPU reference.
func checkClose(t *testing.T, name string, got *tensai.Tensor, want []tensai.Float, tol float64) {
	t.Helper()
	if len(got.Data) != len(want) {
		t.Fatalf("%s: got %d elements, want %d", name, len(got.Data), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got.Data[i] - want[i])); diff > tol*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s element %d: gpu=%v cpu=%v", name, i, got.Data[i], want[i])
		}
	}
}

// TestGPUBinaryOps checks the element-wise arithmetic, including the
// cyclic broadcast a bias row uses.
func TestGPUBinaryOps(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(31, 0))

	x := randTensor(rng, 6, 8)
	y := randTensor(rng, 6, 8)
	row := randTensor(rng, 1, 8) // broadcasts over the rows
	for i := range row.Data {
		row.Data[i] += 2 // keep division well conditioned
	}
	gx, gy, grow := upload3(t, g, x, y, row)
	defer gx.Free()
	defer gy.Free()
	defer grow.Free()

	ops := []struct {
		name string
		op   BinOp
		fn   func(a, b tensai.Float) tensai.Float
	}{
		{"add", OpAdd, func(a, b tensai.Float) tensai.Float { return a + b }},
		{"sub", OpSub, func(a, b tensai.Float) tensai.Float { return a - b }},
		{"mul", OpMul, func(a, b tensai.Float) tensai.Float { return a * b }},
		{"div", OpDiv, func(a, b tensai.Float) tensai.Float { return a / b }},
	}
	for _, o := range ops {
		for _, rhs := range []struct {
			name string
			gt   *Tensor
			ct   *tensai.Tensor
		}{{"same", gy, y}, {"broadcast", grow, row}} {
			out, err := gx.Binary(o.op, rhs.gt)
			if err != nil {
				t.Fatalf("%s/%s: %v", o.name, rhs.name, err)
			}
			got, err := out.Download()
			out.Free()
			if err != nil {
				t.Fatal(err)
			}
			want := make([]tensai.Float, len(x.Data))
			for i := range want {
				want[i] = o.fn(x.Data[i], rhs.ct.Data[i%len(rhs.ct.Data)])
			}
			checkClose(t, o.name+"/"+rhs.name, got, want, 1e-5)
		}
	}

	// A shape that does not divide the output cannot be broadcast.
	odd, err := g.Upload(randTensor(rng, 5))
	if err != nil {
		t.Fatal(err)
	}
	defer odd.Free()
	if _, err := gx.Binary(OpAdd, odd); err == nil {
		t.Error("expected a broadcast error")
	}
}

// broadcastPairs are shape pairs NumPy broadcasts, chosen to cover each
// way an operand can be stretched: a trailing block, an axis of length 1
// in the middle (the pairs that used to wrap cyclically and come out
// wrong without a word), both sides at once, the left side only, and a
// rank-six walk that merges to six axes.
var broadcastPairs = [][2][]int{
	{{6, 8}, {6, 8}},
	{{4, 6, 8}, {8}},
	{{4, 6, 8}, {1, 1, 8}},
	{{6, 8}, {1}},
	{{2, 3, 4, 4}, {3, 4, 4}},
	{{2, 3, 4, 4}, {3, 1, 4}},    // (heads, batch, seq, seq) + (batch, 1, seq)
	{{3, 2, 4, 4}, {3, 1, 1, 4}}, // a key-padding mask over (batch, heads, seq, seq)
	{{5, 4}, {5, 1}},             // a column: (batch, options) - (batch, 1)
	{{5, 1}, {1, 4}},             // both sides stretch
	{{1, 4}, {5, 1}},
	{{4}, {3, 4}}, // only the left side stretches
	{{3, 1}, {3, 5}},
	{{3, 1, 5}, {1, 4, 1}},
	{{2, 1, 3, 1}, {1, 5, 1, 4}},
	{{7, 1, 1, 9}, {1, 3, 2, 1}},
	{{2, 3, 1, 5, 1, 2}, {3, 4, 1, 6, 1}},
	{{300, 1, 17}, {1, 5, 17}}, // more than one workgroup
}

// TestGPUBinaryBroadcast checks every op over shape pairs NumPy broadcasts
// against the CPU's broadcasting arithmetic, element for element.
func TestGPUBinaryBroadcast(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(33, 0))

	ops := []struct {
		name string
		op   BinOp
		cpu  func(a, b *tensai.Tensor) (*tensai.Tensor, error)
	}{
		{"add", OpAdd, (*tensai.Tensor).Add},
		{"sub", OpSub, (*tensai.Tensor).Sub},
		{"mul", OpMul, (*tensai.Tensor).Mul},
		{"div", OpDiv, (*tensai.Tensor).Div},
	}
	for _, pair := range broadcastPairs {
		a, b := randTensor(rng, pair[0]...), randTensor(rng, pair[1]...)
		for i := range b.Data {
			b.Data[i] += 3 // keep division well conditioned
		}
		ga, gb := upload2(t, g, a, b)
		for _, o := range ops {
			want, err := o.cpu(a, b)
			if err != nil {
				t.Fatalf("cpu %s %v %v: %v", o.name, pair[0], pair[1], err)
			}
			out, err := ga.Binary(o.op, gb)
			if err != nil {
				t.Fatalf("%s %v %v: %v", o.name, pair[0], pair[1], err)
			}
			if !dims.Same(out.Shape(), want.Shape) {
				t.Fatalf("%s %v %v: shape %v, want %v", o.name, pair[0], pair[1], out.Shape(), want.Shape)
			}
			got, err := out.Download()
			out.Free()
			if err != nil {
				t.Fatal(err)
			}
			checkClose(t, fmt.Sprintf("%s %v %v", o.name, pair[0], pair[1]), got, want.Data, 1e-6)
		}
		ga.Free()
		gb.Free()
	}

	// Shapes NumPy cannot broadcast are refused -- including ones whose
	// sizes divide, which the cyclic kernel alone would have taken.
	for _, pair := range [][2][]int{{{6, 8}, {5}}, {{6, 8}, {2, 4}}, {{6, 8}, {8, 1}}} {
		ga, gb := upload2(t, g, randTensor(rng, pair[0]...), randTensor(rng, pair[1]...))
		if out, err := ga.Binary(OpAdd, gb); err == nil {
			out.Free()
			t.Errorf("%v + %v: expected a broadcast error", pair[0], pair[1])
		}
		ga.Free()
		gb.Free()
	}
	// A walk longer than the kernels take is an error too, not a guess.
	long := [2][]int{{2, 1, 2, 1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2, 1}}
	ga, gb := upload2(t, g, randTensor(rng, long[0]...), randTensor(rng, long[1]...))
	defer ga.Free()
	defer gb.Free()
	if out, err := ga.Binary(OpAdd, gb); err == nil {
		out.Free()
		t.Errorf("%v + %v merges to nine axes: expected an error", long[0], long[1])
	}
}

// sumToRef sums x down to shape on the host, in float64.
func sumToRef(x *tensai.Tensor, shape []int) []tensai.Float {
	strides := dims.BroadcastStrides(shape, x.Shape)
	sums := make([]float64, dims.Prod(shape))
	idx := make([]int, len(x.Shape))
	for _, v := range x.Data {
		off := 0
		for d, i := range idx {
			off += i * strides[d]
		}
		sums[off] += float64(v)
		for d := len(idx) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < x.Shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	out := make([]tensai.Float, len(sums))
	for i, v := range sums {
		out[i] = tensai.Float(v)
	}
	return out
}

// TestGPUSumTo checks the reduction a broadcast operand's gradient takes,
// for the operand side of every pair TestGPUBinaryBroadcast runs and a few
// more: a block of rows, everything into one element, and no reduction.
func TestGPUSumTo(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(35, 0))

	cases := [][2][]int{
		{{4, 6, 8}, {1, 6, 1}},
		{{4, 6, 8}, {1}},
		{{4, 6, 8}, {1, 1, 1}},
		{{4, 6, 8}, {4, 6, 8}},
		{{1000, 3}, {3}},
	}
	for _, pair := range broadcastPairs {
		out, err := dims.Broadcast(pair[0], pair[1])
		if err != nil {
			t.Fatal(err)
		}
		cases = append(cases, [2][]int{out, pair[0]}, [2][]int{out, pair[1]})
	}
	for _, c := range cases {
		x := randTensor(rng, c[0]...)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatal(err)
		}
		out, err := gx.SumTo(c[1]...)
		gx.Free()
		if err != nil {
			t.Fatalf("%v to %v: %v", c[0], c[1], err)
		}
		if !dims.Same(out.Shape(), c[1]) {
			t.Fatalf("%v to %v: shape %v", c[0], c[1], out.Shape())
		}
		got, err := out.Download()
		out.Free()
		if err != nil {
			t.Fatal(err)
		}
		checkClose(t, fmt.Sprintf("sum %v to %v", c[0], c[1]), got, sumToRef(x, c[1]), 1e-4)
	}

	gx, err := g.Upload(randTensor(rng, 4, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer gx.Free()
	if out, err := gx.SumTo(4, 1, 6); err == nil {
		out.Free()
		t.Error("summing (4, 6) to a larger shape should fail")
	}
	if out, err := gx.SumTo(3); err == nil {
		out.Free()
		t.Error("summing (4, 6) to (3) should fail")
	}
}

// TestGPUActivations checks each activation and its gradient against the
// CPU kernels the rest of tensai uses.
func TestGPUActivations(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(37, 0))

	x := randTensor(rng, 4, 16)
	grad := randTensor(rng, 4, 16)
	gx, ggrad := upload2(t, g, x, grad)
	defer gx.Free()
	defer ggrad.Free()

	acts := []struct {
		name string
		act  Act
		fwd  func(f tensai.Float) tensai.Float
		bwd  func(f tensai.Float) tensai.Float
	}{
		{"relu", ActReLU,
			func(f tensai.Float) tensai.Float {
				if f > 0 {
					return f
				}
				return 0
			},
			func(f tensai.Float) tensai.Float {
				if f > 0 {
					return 1
				}
				return 0
			}},
		{"tanh", ActTanh,
			kernels.TanhF,
			func(f tensai.Float) tensai.Float { y := kernels.TanhF(f); return 1 - y*y }},
		{"sigmoid", ActSigmoid,
			func(f tensai.Float) tensai.Float { return 1 / (1 + kernels.ExpF(-f)) },
			func(f tensai.Float) tensai.Float {
				y := 1 / (1 + kernels.ExpF(-f))
				return y * (1 - y)
			}},
		{"gelu", ActGELU, kernels.GeluF, kernels.GeluGrad},
	}
	for _, a := range acts {
		out, err := gx.Activate(a.act)
		if err != nil {
			t.Fatalf("%s forward: %v", a.name, err)
		}
		got, err := out.Download()
		out.Free()
		if err != nil {
			t.Fatal(err)
		}
		want := make([]tensai.Float, len(x.Data))
		for i, v := range x.Data {
			want[i] = a.fwd(v)
		}
		checkClose(t, a.name+" forward", got, want, 1e-5)

		dOut, err := gx.ActivateGrad(a.act, ggrad)
		if err != nil {
			t.Fatalf("%s backward: %v", a.name, err)
		}
		gotG, err := dOut.Download()
		dOut.Free()
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range x.Data {
			want[i] = grad.Data[i] * a.bwd(v)
		}
		checkClose(t, a.name+" backward", gotG, want, 1e-5)
	}
}

// TestGPUSumCols checks the reduction a broadcast operand's gradient uses.
func TestGPUSumCols(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(41, 0))

	x := randTensor(rng, 33, 12) // rows not a multiple of the workgroup
	gx, err := g.Upload(x)
	if err != nil {
		t.Fatal(err)
	}
	defer gx.Free()
	out, err := gx.SumCols()
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape[0] != 1 || got.Shape[1] != 12 {
		t.Fatalf("shape %v, want [1 12]", got.Shape)
	}
	want := make([]tensai.Float, 12)
	for r := 0; r < 33; r++ {
		for c := 0; c < 12; c++ {
			want[c] += x.Data[r*12+c]
		}
	}
	checkClose(t, "sumcols", got, want, 1e-5)
}

// TestGPUAdamStep runs several updates on the device and on the CPU kernel
// and checks the weights stay together.
func TestGPUAdamStep(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(43, 0))

	const n = 40
	w := randTensor(rng, n)
	grad := randTensor(rng, n)
	cpuW := append([]tensai.Float(nil), w.Data...)
	cpuM := make([]tensai.Float, n)
	cpuV := make([]tensai.Float, n)

	gw, err := g.Upload(w)
	if err != nil {
		t.Fatal(err)
	}
	defer gw.Free()
	ggrad, err := g.Upload(grad)
	if err != nil {
		t.Fatal(err)
	}
	defer ggrad.Free()
	gm, err := g.Upload(tensai.NewTensor(n))
	if err != nil {
		t.Fatal(err)
	}
	defer gm.Free()
	gv, err := g.Upload(tensai.NewTensor(n))
	if err != nil {
		t.Fatal(err)
	}
	defer gv.Free()

	const beta1, beta2, lr, eps = 0.9, 0.999, 0.01, 1e-8
	for step := 1; step <= 5; step++ {
		rc1 := 1 / (1 - kernels.PowF(beta1, tensai.Float(step)))
		rc2 := 1 / (1 - kernels.PowF(beta2, tensai.Float(step)))
		if err := gw.AdamStep(ggrad, gm, gv, lr, beta1, beta2, rc1, rc2, eps, 0); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		kernels.AdamStep(cpuW, grad.Data, cpuM, cpuV, beta1, beta2, rc1, rc2, lr, eps, 0)
	}
	got, err := gw.Download()
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "adam", got, cpuW, 1e-5)
}

func upload2(t *testing.T, g *Device, a, b *tensai.Tensor) (*Tensor, *Tensor) {
	t.Helper()
	ga, err := g.Upload(a)
	if err != nil {
		t.Fatal(err)
	}
	gb, err := g.Upload(b)
	if err != nil {
		t.Fatal(err)
	}
	return ga, gb
}

func upload3(t *testing.T, g *Device, a, b, c *tensai.Tensor) (*Tensor, *Tensor, *Tensor) {
	t.Helper()
	ga, gb := upload2(t, g, a, b)
	gc, err := g.Upload(c)
	if err != nil {
		t.Fatal(err)
	}
	return ga, gb, gc
}

// TestGPULayerNorm checks the normalization and both of its gradients
// against the CPU kernels the layer package uses.
func TestGPULayerNorm(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(53, 0))

	const rows, cols = 5, 40
	x := randTensor(rng, rows, cols)
	grad := randTensor(rng, rows, cols)
	gain := randTensor(rng, cols)
	bias := randTensor(rng, cols)
	const eps = 1e-5

	gx, ggrad := upload2(t, g, x, grad)
	defer gx.Free()
	defer ggrad.Free()
	ggain, gbias := upload2(t, g, gain, bias)
	defer ggain.Free()
	defer gbias.Free()

	// Forward, against LnFwdRow.
	out, err := gx.LayerNorm(ggain, gbias, eps)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]tensai.Float, rows*cols)
	xhat := make([]tensai.Float, rows*cols)
	invStd := make([]tensai.Float, rows)
	for r := 0; r < rows; r++ {
		lo, hi := r*cols, (r+1)*cols
		invStd[r] = kernels.LnFwdRow(want[lo:hi], xhat[lo:hi], x.Data[lo:hi], gain.Data, bias.Data, eps)
	}
	checkClose(t, "layernorm", got, want, 1e-4)

	// The normalized values on their own.
	xh, err := gx.LayerNormXhat(eps)
	if err != nil {
		t.Fatal(err)
	}
	defer xh.Free()
	gotXhat, err := xh.Download()
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "layernorm xhat", gotXhat, xhat, 1e-4)

	// Backward, against LnBwdRow.
	dx, err := gx.LayerNormGrad(ggrad, ggain, eps)
	if err != nil {
		t.Fatal(err)
	}
	defer dx.Free()
	gotDx, err := dx.Download()
	if err != nil {
		t.Fatal(err)
	}
	wantDx := make([]tensai.Float, rows*cols)
	gradGamma, gradBeta := make([]tensai.Float, cols), make([]tensai.Float, cols)
	for r := 0; r < rows; r++ {
		lo, hi := r*cols, (r+1)*cols
		kernels.LnBwdRow(wantDx[lo:hi], grad.Data[lo:hi], xhat[lo:hi], gain.Data, gradGamma, gradBeta, invStd[r])
	}
	checkClose(t, "layernorm backward", gotDx, wantDx, 1e-4)
}

// TestGPUSoftmaxGrad checks the softmax backward pass against the CPU
// kernel, on a 4-D stack the way attention produces it.
func TestGPUSoftmaxGrad(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(59, 0))

	const b, h, seq = 2, 3, 12
	scores := randTensor(rng, b, h, seq, seq)
	gscores, err := g.Upload(scores)
	if err != nil {
		t.Fatal(err)
	}
	defer gscores.Free()
	y, err := gscores.Softmax()
	if err != nil {
		t.Fatal(err)
	}
	defer y.Free()
	hostY, err := y.Download()
	if err != nil {
		t.Fatal(err)
	}

	grad := randTensor(rng, b, h, seq, seq)
	ggrad, err := g.Upload(grad)
	if err != nil {
		t.Fatal(err)
	}
	defer ggrad.Free()
	dx, err := y.SoftmaxGrad(ggrad)
	if err != nil {
		t.Fatal(err)
	}
	defer dx.Free()
	got, err := dx.Download()
	if err != nil {
		t.Fatal(err)
	}

	want := make([]tensai.Float, len(grad.Data))
	for pos := 0; pos < len(want); pos += seq {
		kernels.SoftmaxBwdAdd(want[pos:pos+seq], grad.Data[pos:pos+seq], hostY.Data[pos:pos+seq])
	}
	checkClose(t, "softmax backward", got, want, 1e-4)
}

// TestGPUPermute checks the axis permutation against the CPU transpose,
// including the (batch, seq, head, dim) -> (batch, head, seq, dim) move
// attention makes.
func TestGPUPermute(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(61, 0))

	cases := []struct {
		shape []int
		perm  []int
	}{
		{[]int{3, 5}, []int{1, 0}},
		{[]int{2, 3, 4}, []int{1, 0, 2}},
		{[]int{2, 6, 3, 8}, []int{0, 2, 1, 3}},
		{[]int{2, 6, 3, 8}, []int{3, 1, 0, 2}},
	}
	for _, c := range cases {
		x := randTensor(rng, c.shape...)
		gx, err := g.Upload(x)
		if err != nil {
			t.Fatal(err)
		}
		out, err := gx.Permute(c.perm...)
		if err != nil {
			t.Fatalf("permute %v by %v: %v", c.shape, c.perm, err)
		}
		got, err := out.Download()
		gx.Free()
		out.Free()
		if err != nil {
			t.Fatal(err)
		}
		want, err := x.Transpose(c.perm...)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.Shape {
			if got.Shape[i] != want.Shape[i] {
				t.Fatalf("permute %v by %v: shape %v, want %v", c.shape, c.perm, got.Shape, want.Shape)
			}
		}
		checkClose(t, "permute", got, want.Data, 1e-6)
	}

	// Rank five is beyond what the kernel expresses.
	big, err := g.Upload(randTensor(rng, 2, 2, 2, 2, 2))
	if err != nil {
		t.Fatal(err)
	}
	defer big.Free()
	if _, err := big.Permute(0, 1, 2, 4, 3); err == nil {
		t.Error("expected an unsupported-rank error")
	}
}

// TestGPUEmbed checks the lookup and its scatter-add, with an index that
// repeats so the atomic path is exercised -- and the same scatter through
// the kernel a backend without those atomics runs instead.
func TestGPUEmbed(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(67, 0))

	const vocab, dim = 7, 12
	table := randTensor(rng, vocab, dim)
	ids := []int{3, 0, 6, 0, 1, 5, 0}
	gtable, err := g.Upload(table)
	if err != nil {
		t.Fatal(err)
	}
	defer gtable.Free()
	gids, err := g.UploadIndices(ids)
	if err != nil {
		t.Fatal(err)
	}
	defer gids.Free()

	out, err := gtable.Embed(gids)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	got, err := out.Download()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]tensai.Float, len(ids)*dim)
	for i, id := range ids {
		copy(want[i*dim:(i+1)*dim], table.Data[id*dim:(id+1)*dim])
	}
	checkClose(t, "embed", got, want, 0)

	// The scatter adds into whatever the gradient buffer already holds.
	grad := randTensor(rng, len(ids), dim)
	ggrad, err := g.Upload(grad)
	if err != nil {
		t.Fatal(err)
	}
	defer ggrad.Free()
	start := randTensor(rng, vocab, dim)
	wantGrad := append([]tensai.Float(nil), start.Data...)
	for i, id := range ids {
		for j := 0; j < dim; j++ {
			wantGrad[id*dim+j] += grad.Data[i*dim+j]
		}
	}
	atomic := g.pipes.embedScatter
	defer func() { g.pipes.embedScatter = atomic }()
	for _, kernel := range []string{"atomic", "columns"} {
		if kernel == "atomic" && atomic == 0 {
			t.Logf("no atomic scatter on %s", g.Name())
			continue
		}
		if kernel == "columns" {
			g.pipes.embedScatter = 0 // what Open leaves when it cannot build it
		}
		gdst, err := g.Upload(start)
		if err != nil {
			t.Fatal(err)
		}
		if err := gdst.EmbedGrad(ggrad, gids); err != nil {
			t.Fatalf("%s: %v", kernel, err)
		}
		gotGrad, err := gdst.Download()
		gdst.Free()
		if err != nil {
			t.Fatal(err)
		}
		checkClose(t, "embed scatter ("+kernel+")", gotGrad, wantGrad, 1e-5)
	}
}
