//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/mattn/tensai"
)

// segLayout is a packed batch: segment s is rows [off[s], off[s+1]),
// rows past the last offset are padding, and flags holds every row's
// RowFlag. cls and opts list, per segment, the rows a last layer reads
// (its first row and its trailing option rows).
type segLayout struct {
	rows  int
	off   []int
	flags []RowFlag
	outs  []int // cls and option rows of every segment, then a padding row when there is one
}

// randLayout packs segs segments of 1..maxLen rows the way a format-v2
// batch looks: a first (CLS) row, a run of gated effect rows that see
// every key, trailing option rows, some of which -- and sometimes the CLS
// row -- are query-only, plus a few empty segments and pad padding rows.
func randLayout(rng *rand.Rand, segs, maxLen, pad int) segLayout {
	var l segLayout
	l.off = append(l.off, 0)
	for s := range segs {
		n := 1 + rng.IntN(maxLen)
		if s%17 == 5 {
			n = 0 // an empty segment
		}
		base := l.rows
		l.rows += n
		l.off = append(l.off, l.rows)
		for range n {
			l.flags = append(l.flags, 0)
		}
		if n == 0 {
			continue
		}
		if rng.IntN(5) == 0 {
			l.flags[base] |= RowQueryOnly
		}
		nOpt := 0
		if n > 2 {
			nOpt = rng.IntN(min(n-1, 12))
		}
		for j := base + n - nOpt; j < base+n; j++ {
			if rng.IntN(3) == 0 {
				l.flags[j] |= RowQueryOnly
			}
		}
		if mid := n - 1 - nOpt; mid > 1 && rng.IntN(4) != 0 {
			g0 := base + 1 + rng.IntN(mid)
			g1 := g0 + 1 + rng.IntN(base+1+mid-g0)
			for j := g0; j < g1; j++ {
				l.flags[j] |= RowGated | RowSeesAll
			}
		}
		l.outs = append(l.outs, base)
		for j := base + n - nOpt; j < base+n; j++ {
			l.outs = append(l.outs, j)
		}
	}
	for range pad {
		l.flags = append(l.flags, 0)
	}
	l.rows += pad
	if pad > 0 {
		l.outs = append(l.outs, l.rows-1)
	}
	return l
}

// refSegAttn is SegAttn's definition in float64: every query attends to
// the keys of its segment that are not query-only; under a gate a query
// that does not see all keys gets softmax-attention over the ungated keys
// plus gate[h] times softmax-attention over the gated ones. q, k, v are
// row-major with row widths qw, kw, vw and heads starting at the columns
// qc, kc, vc. A group without keys contributes nothing.
func refSegAttn(l segLayout, q, k, v []float32, qw, kw, vw, qc, kc, vc, heads, dim int, qrows []int, gate []float32, scale float64) []float32 {
	segOf := make([]int, l.rows)
	for r := range segOf {
		segOf[r] = -1
	}
	for s := 0; s+1 < len(l.off); s++ {
		for r := l.off[s]; r < l.off[s+1]; r++ {
			segOf[r] = s
		}
	}
	nq := l.rows
	if qrows != nil {
		nq = len(qrows)
	}
	dh := dim / heads
	out := make([]float32, nq*dim)
	att := func(i, h int, keys []int) []float64 {
		o := make([]float64, dh)
		if len(keys) == 0 {
			return o
		}
		sc := make([]float64, len(keys))
		top := math.Inf(-1)
		for n, j := range keys {
			var x float64
			for c := range dh {
				x += float64(q[i*qw+qc+h*dh+c]) * float64(k[j*kw+kc+h*dh+c])
			}
			sc[n] = x * scale
			top = max(top, sc[n])
		}
		var sum float64
		for n := range sc {
			sc[n] = math.Exp(sc[n] - top)
			sum += sc[n]
		}
		for n, j := range keys {
			for c := range dh {
				o[c] += sc[n] / sum * float64(v[j*vw+vc+h*dh+c])
			}
		}
		return o
	}
	for i := range nq {
		r := i
		if qrows != nil {
			r = qrows[i]
		}
		if r >= l.rows || segOf[r] < 0 {
			continue
		}
		s := segOf[r]
		split := gate != nil && l.flags[r]&RowSeesAll == 0
		var plain, gated []int
		for j := l.off[s]; j < l.off[s+1]; j++ {
			switch {
			case l.flags[j]&RowQueryOnly != 0:
			case split && l.flags[j]&RowGated != 0:
				gated = append(gated, j)
			default:
				plain = append(plain, j)
			}
		}
		for h := range heads {
			o0, o1 := att(i, h, plain), att(i, h, gated)
			for c := range dh {
				x := o0[c]
				if split {
					x += float64(gate[h]) * o1[c]
				}
				out[i*dim+h*dh+c] = float32(x)
			}
		}
	}
	return out
}

func randFloats(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

func uploadFloats(t *testing.T, g *Device, data []float32, shape ...int) *Tensor {
	t.Helper()
	x := tensai.NewTensor(shape...)
	copy(x.Data, data)
	gx, err := g.Upload(x)
	if err != nil {
		t.Fatal(err)
	}
	return gx
}

func checkFloats(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d elements, want %d", what, len(got), len(want))
	}
	for i := range want {
		w := float64(want[i])
		if diff := math.Abs(float64(got[i]) - w); !(diff <= tol*(1+math.Abs(w))) { // NaN fails too
			t.Fatalf("%s element %d: gpu=%v cpu=%v", what, i, got[i], want[i])
		}
	}
}

// segAttnRun uploads the operands of one SegmentAttention, runs it, and
// returns the downloaded output.
type segAttnRun struct {
	l                segLayout
	heads, dim       int
	fused            bool // q, k and v as one q|k|v tensor; otherwise three tensors with a column offset each
	subset           bool // queries are l.outs
	gate             []float32
	scale            float32
	q, k, v          []float32
	qw, kw, vw       int
	qc, kc, vc, nq   int
	qrows            []int
	gq, gk, gv, gidx *Tensor
	seg              *Segments
}

func newSegAttnRun(t *testing.T, g *Device, rng *rand.Rand, l segLayout, heads, dim int, fused, subset bool, gate []float32) *segAttnRun {
	t.Helper()
	r := &segAttnRun{l: l, heads: heads, dim: dim, fused: fused, subset: subset, gate: gate}
	seg, err := g.NewSegments(l.rows, l.off, l.flags)
	if err != nil {
		t.Fatal(err)
	}
	r.seg = seg
	r.nq = l.rows
	if subset {
		r.qrows = l.outs
		r.nq = len(l.outs)
		if r.gidx, err = g.UploadIndices(r.qrows); err != nil {
			t.Fatal(err)
		}
	}
	if fused && !subset {
		r.qw, r.kw, r.vw = 3*dim, 3*dim, 3*dim
		r.qc, r.kc, r.vc = 0, dim, 2*dim
		qkv := randFloats(rng, l.rows*3*dim)
		r.q, r.k, r.v = qkv, qkv, qkv
		r.gq = uploadFloats(t, g, qkv, l.rows, 3*dim)
		r.gk, r.gv = r.gq, r.gq
		return r
	}
	// Separate tensors, each with columns around the heads.
	r.qw, r.qc = dim+8, 4
	r.kw, r.kc = dim+4, 4
	r.vw, r.vc = dim+12, 8
	if fused { // k|v fused, q on its own (a last layer)
		r.kw, r.kc, r.vw, r.vc = 2*dim, 0, 2*dim, dim
	}
	r.q = randFloats(rng, r.nq*r.qw)
	r.gq = uploadFloats(t, g, r.q, r.nq, r.qw)
	r.k = randFloats(rng, l.rows*r.kw)
	r.gk = uploadFloats(t, g, r.k, l.rows, r.kw)
	if fused {
		r.v, r.gv = r.k, r.gk
	} else {
		r.v = randFloats(rng, l.rows*r.vw)
		r.gv = uploadFloats(t, g, r.v, l.rows, r.vw)
	}
	return r
}

func (r *segAttnRun) args() SegAttn {
	return SegAttn{
		Seg: r.seg, Heads: r.heads, Dim: r.dim,
		Q: r.gq, K: r.gk, V: r.gv, QCol: r.qc, KCol: r.kc, VCol: r.vc,
		QRows: r.gidx, Gate: r.gate, Scale: r.scale,
	}
}

func (r *segAttnRun) want() []float32 {
	scale := float64(r.scale)
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(r.dim/r.heads))
	}
	return refSegAttn(r.l, r.q, r.k, r.v, r.qw, r.kw, r.vw, r.qc, r.kc, r.vc, r.heads, r.dim, r.qrows, r.gate, scale)
}

func (r *segAttnRun) free() {
	r.seg.Free()
	for _, x := range []*Tensor{r.gq, r.gk, r.gv, r.gidx} {
		if x != nil {
			x.Free()
		}
	}
}

func runSegAttn(t *testing.T, g *Device, r *segAttnRun) []float32 {
	t.Helper()
	out, err := g.SegmentAttention(r.args())
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if s := out.Shape(); !slices.Equal(s, []int{r.nq, r.dim}) {
		t.Fatalf("output shape %v, want [%d %d]", s, r.nq, r.dim)
	}
	got := make([]float32, r.nq*r.dim)
	if err := out.DownloadInto(got); err != nil {
		t.Fatal(err)
	}
	return got
}

func randGate(rng *rand.Rand, heads int) []float32 {
	gate := make([]float32, heads)
	for h := range gate {
		gate[h] = float32(math.Tanh(rng.NormFloat64()))
	}
	gate[heads/2] = 0 // a head whose gate has not opened yet
	return gate
}

// TestGPUSegmentAttention checks SegmentAttention against refSegAttn on
// ragged batches (segments of 1 to 96 rows, empty segments, padding),
// for every supported head size, with and without a gate, over all rows
// and over the rows a last layer reads, on fused and separate operands.
func TestGPUSegmentAttention(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(41, 0))
	for _, c := range []struct{ heads, dh int }{
		{1, 4}, {2, 8}, {4, 16}, {4, 32}, {8, 32}, {2, 64}, {1, 128}, {3, 12},
	} {
		dim := c.heads * c.dh
		for _, gated := range []bool{false, true} {
			for _, subset := range []bool{false, true} {
				for _, fused := range []bool{false, true} {
					name := fmt.Sprintf("heads=%d dh=%d gate=%v subset=%v fused=%v", c.heads, c.dh, gated, subset, fused)
					l := randLayout(rng, 60, 96, rng.IntN(3)*7)
					var gate []float32
					if gated {
						gate = randGate(rng, c.heads)
					}
					r := newSegAttnRun(t, g, rng, l, c.heads, dim, fused, subset, gate)
					checkFloats(t, name, runSegAttn(t, g, r), r.want(), 2e-5)
					r.free()
				}
			}
		}
	}
}

// TestGPUSegmentAttentionPaths runs every workgroup through the tiles of
// workgroup memory, then every one straight from the buffers, on full and
// subset queries: whichever path a workgroup takes, the result is the
// same.
func TestGPUSegmentAttentionPaths(t *testing.T) {
	defer func(v int) { segAttnSpan = v }(segAttnSpan)
	for _, span := range []int{1 << 30, 0} {
		segAttnSpan = span
		g := openTestGPU(t)
		rng := rand.New(rand.NewPCG(37, 0))
		for _, subset := range []bool{false, true} {
			for _, gated := range []bool{false, true} {
				l := randLayout(rng, 50, 96, 4)
				var gate []float32
				if gated {
					gate = randGate(rng, 4)
				}
				r := newSegAttnRun(t, g, rng, l, 4, 128, true, subset, gate)
				checkFloats(t, fmt.Sprintf("span=%d subset=%v gate=%v", span, subset, gated), runSegAttn(t, g, r), r.want(), 2e-5)
				r.free()
			}
		}
		g.Close()
	}
}

// TestGPUSegmentAttentionChecked runs the same kernel compiled with wgpu's
// runtime checks, as a module without them must give the same result.
func TestGPUSegmentAttentionChecked(t *testing.T) {
	gemmChecked = true
	defer func() { gemmChecked = false }()
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(43, 0))
	l := randLayout(rng, 40, 96, 5)
	r := newSegAttnRun(t, g, rng, l, 4, 128, true, false, randGate(rng, 4))
	defer r.free()
	checkFloats(t, "checked", runSegAttn(t, g, r), r.want(), 2e-5)
}

// TestGPUSegmentAttentionManySegments packs thousands of segments of the
// lengths a batch of game positions has, with the query count past one
// row of workgroups' worth of grid in the full case.
func TestGPUSegmentAttentionManySegments(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(47, 0))
	l := randLayout(rng, 2000, 90, 0)
	for _, subset := range []bool{false, true} {
		r := newSegAttnRun(t, g, rng, l, 4, 128, true, subset, randGate(rng, 4))
		checkFloats(t, fmt.Sprintf("subset=%v", subset), runSegAttn(t, g, r), r.want(), 2e-5)
		r.free()
	}
}

// mlxGatedAttention is model_v2.py's _gated_attention for one segment of
// rows (q, k, v already projected; heads of dh in the packed layout) in
// float64: additive masks of -1e9, a softmax over the ungated keys, one
// over the gated keys switched off when the segment has none, and a full
// one for the gated queries.
func mlxGatedAttention(q, k, v []float32, n, heads, dh int, keyOK, gk, gq []bool, gamma []float32) []float64 {
	const neg = -1e9
	d := heads * dh
	out := make([]float64, n*d)
	has := slices.Contains(gk, true)
	sdpa := func(i, h int, mask func(j int) float64) []float64 {
		sc := make([]float64, n)
		top := math.Inf(-1)
		for j := range n {
			var x float64
			for c := range dh {
				x += float64(q[i*d+h*dh+c]) * float64(k[j*d+h*dh+c])
			}
			sc[j] = x/math.Sqrt(float64(dh)) + mask(j)
			top = max(top, sc[j])
		}
		var sum float64
		for j := range sc {
			sc[j] = math.Exp(sc[j] - top)
			sum += sc[j]
		}
		o := make([]float64, dh)
		for j := range n {
			for c := range dh {
				o[c] += sc[j] / sum * float64(v[j*d+h*dh+c])
			}
		}
		return o
	}
	b2f := func(b bool) float64 {
		if b {
			return 1
		}
		return 0
	}
	for i := range n {
		for h := range heads {
			base := sdpa(i, h, func(j int) float64 { return (1-b2f(keyOK[j]))*neg + b2f(gk[j])*neg })
			eff := sdpa(i, h, func(j int) float64 { return (1-b2f(keyOK[j]))*neg + (1-b2f(gk[j]))*neg })
			full := sdpa(i, h, func(j int) float64 { return (1 - b2f(keyOK[j])) * neg })
			for c := range dh {
				o := base[c] + math.Tanh(float64(gamma[h]))*b2f(has)*eff[c]
				if gq[i] {
					o = full[c]
				}
				out[i*d+h*dh+c] = o
			}
		}
	}
	return out
}

// TestGPUSegmentAttentionMatchesMLX checks a gated layer against the
// training model's formula (model_v2.py _gated_attention: gate keys and
// gate queries are the effect rows, the gate weight is tanh(alpha)) on
// segments that always keep an ungated key.
func TestGPUSegmentAttentionMatchesMLX(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(53, 0))
	const heads, dh = 4, 32
	const d = heads * dh
	l := randLayout(rng, 80, 89, 0)
	for s := 0; s+1 < len(l.off); s++ {
		if l.off[s] < l.off[s+1] {
			l.flags[l.off[s]] &^= RowQueryOnly // the CLS row stays a key
		}
	}
	alpha := []float32{0.7, -1.3, 0, 2.1}
	gate := make([]float32, heads)
	for h, a := range alpha {
		gate[h] = float32(math.Tanh(float64(a)))
	}
	qkv := randFloats(rng, l.rows*3*d)
	gqkv := uploadFloats(t, g, qkv, l.rows, 3*d)
	defer gqkv.Free()
	seg, err := g.NewSegments(l.rows, l.off, l.flags)
	if err != nil {
		t.Fatal(err)
	}
	defer seg.Free()
	out, err := g.SegmentAttention(SegAttn{Seg: seg, Heads: heads, Dim: d, Q: gqkv, K: gqkv, V: gqkv, KCol: d, VCol: 2 * d, Gate: gate})
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	got := make([]float32, l.rows*d)
	if err := out.DownloadInto(got); err != nil {
		t.Fatal(err)
	}
	var want []float32
	for s := 0; s+1 < len(l.off); s++ {
		o, n := l.off[s], l.off[s+1]-l.off[s]
		q, k, v := make([]float32, n*d), make([]float32, n*d), make([]float32, n*d)
		keyOK, gk, gq := make([]bool, n), make([]bool, n), make([]bool, n)
		for i := range n {
			row := qkv[(o+i)*3*d:]
			copy(q[i*d:], row[:d])
			copy(k[i*d:], row[d:2*d])
			copy(v[i*d:], row[2*d:3*d])
			f := l.flags[o+i]
			keyOK[i], gk[i], gq[i] = f&RowQueryOnly == 0, f&RowGated != 0, f&RowSeesAll != 0
		}
		for _, x := range mlxGatedAttention(q, k, v, n, heads, dh, keyOK, gk, gq, alpha) {
			want = append(want, float32(x))
		}
	}
	checkFloats(t, "mlx", got, want, 2e-5)
}

// TestGPUSegmentAttentionGateZero checks that a gate weight of 0 leaves a
// query that does not see all keys exactly where it would be without the
// gated rows: the same as running ungated with those rows query-only.
func TestGPUSegmentAttentionGateZero(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(59, 0))
	l := randLayout(rng, 50, 96, 0)
	r := newSegAttnRun(t, g, rng, l, 4, 64, true, true, make([]float32, 4))
	defer r.free()
	got := runSegAttn(t, g, r)
	hidden := slices.Clone(l.flags)
	for j, f := range hidden {
		if f&RowGated != 0 {
			hidden[j] |= RowQueryOnly
		}
	}
	seg, err := g.NewSegments(l.rows, l.off, hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer seg.Free()
	a := r.args()
	a.Seg, a.Gate = seg, nil
	out, err := g.SegmentAttention(a)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	want := make([]float32, r.nq*r.dim)
	if err := out.DownloadInto(want); err != nil {
		t.Fatal(err)
	}
	for i, row := range r.qrows {
		if l.flags[row]&RowSeesAll != 0 {
			continue
		}
		checkFloats(t, fmt.Sprintf("query %d", i), got[i*r.dim:(i+1)*r.dim], want[i*r.dim:(i+1)*r.dim], 1e-6)
	}
}

// TestGPUSegmentAttentionEdges covers segments a query gets nothing from:
// every key query-only, only gated keys under a gate, one row, rows outside
// every segment (before the first offset and after the last).
func TestGPUSegmentAttentionEdges(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(61, 0))
	l := segLayout{
		rows: 16,
		off:  []int{2, 5, 6, 6, 10, 13},
		flags: []RowFlag{
			0, 0, // before the first segment
			RowQueryOnly, RowQueryOnly, RowQueryOnly, // no key at all
			0,                                                       // one row
			RowGated | RowSeesAll, RowGated, RowGated, RowQueryOnly, // only gated keys
			0, RowGated, RowQueryOnly | RowSeesAll,
			0, 0, 0, // after the last segment
		},
	}
	l.outs = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	for _, gate := range [][]float32{nil, {0.5, -0.25}} {
		for _, subset := range []bool{false, true} {
			r := newSegAttnRun(t, g, rng, l, 2, 8, true, subset, gate)
			got := runSegAttn(t, g, r)
			checkFloats(t, fmt.Sprintf("gate=%v subset=%v", gate, subset), got, r.want(), 2e-5)
			for _, row := range []int{0, 1, 2, 3, 4, 13, 14, 15} {
				for c := range r.dim {
					if got[row*r.dim+c] != 0 {
						t.Fatalf("gate=%v subset=%v: row %d column %d = %v, want 0", gate, subset, row, c, got[row*r.dim+c])
					}
				}
			}
			r.free()
		}
	}
}

// TestGPUSegmentAttentionBatch chains two attentions and the projections
// around them inside BeginBatch, as a model forward records them, and
// checks the result after Flush.
func TestGPUSegmentAttentionBatch(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(67, 0))
	const heads, d = 4, 64
	l := randLayout(rng, 30, 96, 3)
	gate := randGate(rng, heads)
	x := randFloats(rng, l.rows*d)
	w := randFloats(rng, d*3*d)
	for i := range w {
		w[i] /= 8 // unit-scale activations: a peaked softmax would amplify the products' f32 rounding
	}
	gx := uploadFloats(t, g, x, l.rows, d)
	defer gx.Free()
	gw := uploadFloats(t, g, w, d, 3*d)
	defer gw.Free()
	seg, err := g.NewSegments(l.rows, l.off, l.flags)
	if err != nil {
		t.Fatal(err)
	}
	defer seg.Free()
	if err := g.BeginBatch(); err != nil {
		t.Fatal(err)
	}
	h := gx
	for range 2 {
		qkv, err := h.Linear(gw, nil)
		if err != nil {
			t.Fatal(err)
		}
		att, err := g.SegmentAttention(SegAttn{Seg: seg, Heads: heads, Q: qkv, K: qkv, V: qkv, Dim: d, KCol: d, VCol: 2 * d, Gate: gate})
		if err != nil {
			t.Fatal(err)
		}
		qkv.Free()
		if h != gx {
			h.Free()
		}
		h = att
	}
	if err := g.Flush(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, l.rows*d)
	if err := h.DownloadInto(got); err != nil {
		t.Fatal(err)
	}
	h.Free()

	want := x
	for range 2 {
		qkv := make([]float32, l.rows*3*d)
		for r := range l.rows {
			for c := range 3 * d {
				var s float64
				for k := range d {
					s += float64(want[r*d+k]) * float64(w[k*3*d+c])
				}
				qkv[r*3*d+c] = float32(s)
			}
		}
		want = refSegAttn(l, qkv, qkv, qkv, 3*d, 3*d, 3*d, 0, d, 2*d, heads, d, nil, gate, 1/math.Sqrt(d/heads))
	}
	checkFloats(t, "batched", got, want, 1e-4)
}

// TestGPUSegmentAttentionErrors checks the host-side checks that keep the
// unchecked kernel inside its buffers.
func TestGPUSegmentAttentionErrors(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(71, 0))
	for _, c := range []struct {
		name  string
		rows  int
		off   []int
		flags []RowFlag
	}{
		{"no offsets", 4, nil, nil},
		{"decreasing", 4, []int{0, 3, 2, 4}, nil},
		{"past the rows", 4, []int{0, 5}, nil},
		{"negative", 4, []int{-1, 4}, nil},
		{"flag count", 4, []int{0, 4}, make([]RowFlag, 3)},
		{"unknown flag", 2, []int{0, 2}, []RowFlag{0, 8}},
	} {
		if s, err := g.NewSegments(c.rows, c.off, c.flags); err == nil {
			s.Free()
			t.Errorf("NewSegments %s: no error", c.name)
		}
	}
	l := randLayout(rng, 4, 8, 0)
	seg, err := g.NewSegments(l.rows, l.off, l.flags)
	if err != nil {
		t.Fatal(err)
	}
	defer seg.Free()
	qkv := uploadFloats(t, g, randFloats(rng, l.rows*96), l.rows, 96)
	defer qkv.Free()
	few := uploadFloats(t, g, randFloats(rng, (l.rows-1)*32), l.rows-1, 32)
	defer few.Free()
	wide := uploadFloats(t, g, randFloats(rng, l.rows*264), l.rows, 264)
	defer wide.Free()
	ok := SegAttn{Seg: seg, Heads: 2, Dim: 32, Q: qkv, K: qkv, V: qkv, KCol: 32, VCol: 64}
	if out, err := g.SegmentAttention(ok); err != nil {
		t.Fatal(err)
	} else {
		out.Free()
	}
	for _, c := range []struct {
		name string
		edit func(a *SegAttn)
	}{
		{"no segments", func(a *SegAttn) { a.Seg = nil }},
		{"heads do not divide", func(a *SegAttn) { a.Heads = 3 }},
		{"head of 2", func(a *SegAttn) { a.Heads, a.Dim = 4, 8 }},
		{"head past 128", func(a *SegAttn) { a.Q, a.K, a.V, a.Heads, a.Dim, a.KCol, a.VCol = wide, wide, wide, 2, 264, 0, 0 }},
		{"column not a vec4", func(a *SegAttn) { a.KCol = 30 }},
		{"columns past the row", func(a *SegAttn) { a.VCol = 68 }},
		{"k short of rows", func(a *SegAttn) { a.K, a.KCol = few, 0 }},
		{"q short of queries", func(a *SegAttn) { a.Q = few }},
		{"gate weights", func(a *SegAttn) { a.Gate = []float32{1} }},
		{"33 gated heads", func(a *SegAttn) {
			a.Q, a.K, a.V, a.Heads, a.Dim, a.KCol, a.VCol, a.Gate = wide, wide, wide, 33, 132, 0, 0, make([]float32, 33)
		}},
	} {
		a := ok
		c.edit(&a)
		if out, err := g.SegmentAttention(a); err == nil {
			out.Free()
			t.Errorf("%s: no error", c.name)
		}
	}
	// 33 heads are fine without a gate.
	out, err := g.SegmentAttention(SegAttn{Seg: seg, Heads: 33, Dim: 132, Q: wide, K: wide, V: wide})
	if err != nil {
		t.Fatalf("33 heads: %v", err)
	}
	out.Free()
}
