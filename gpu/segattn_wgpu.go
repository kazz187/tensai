//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"unsafe"
)

// Segments splits the packed rows of SegmentAttention into segments and
// carries every row's RowFlag. Build one per batch with NewSegments, pass
// it to every layer's attention, and Free it.
type Segments struct {
	// t holds u32 words: the segs+1 offsets, the last offset again (the
	// empty range of rows that lie outside every segment), then one word
	// per row: its segment << 3 | its flags.
	t          *Tensor
	rows, segs int
}

// maxSegments keeps a segment index within the 29 bits a row word has.
const maxSegments = 1<<29 - 2

// NewSegments puts a segmentation of rows packed rows on the device:
// segment s is rows [off[s], off[s+1]), so off holds one more entry than
// there are segments and never decreases. Rows before off[0] or from the
// last offset on belong to no segment -- padding that rounds a batch up
// to a size the buffer pool already holds. flags holds one RowFlag per
// row, or is nil for none.
func (g *Device) NewSegments(rows int, off []int, flags []RowFlag) (*Segments, error) {
	if len(off) < 1 {
		return nil, errors.New("tensai: segments need at least one offset")
	}
	segs := len(off) - 1
	if segs > maxSegments {
		return nil, fmt.Errorf("tensai: %d segments exceed the limit of %d", segs, maxSegments)
	}
	if rows < 1 || uint64(rows)+uint64(segs)+2 > math.MaxUint32 {
		return nil, fmt.Errorf("tensai: segments over %d rows", rows)
	}
	if flags != nil && len(flags) != rows {
		return nil, fmt.Errorf("tensai: %d row flags for %d rows", len(flags), rows)
	}
	if off[0] < 0 || off[segs] > rows {
		return nil, fmt.Errorf("tensai: segment offsets [%d, %d] fall outside %d rows", off[0], off[segs], rows)
	}
	words := make([]uint32, segs+2+rows)
	for s := range off {
		if s > 0 && off[s] < off[s-1] {
			return nil, fmt.Errorf("tensai: segment offset %d (%d) is below offset %d (%d)", s, off[s], s-1, off[s-1])
		}
		words[s] = uint32(off[s])
	}
	words[segs+1] = uint32(off[segs])
	row := words[segs+2:]
	for r := range row {
		row[r] = uint32(segs) << 3 // outside every segment until the loop below
	}
	for s := range segs {
		for r := off[s]; r < off[s+1]; r++ {
			row[r] = uint32(s) << 3
		}
	}
	for r, f := range flags {
		if f&^(RowQueryOnly|RowGated|RowSeesAll) != 0 {
			return nil, fmt.Errorf("tensai: row %d has unknown flags %#x", r, f)
		}
		row[r] |= uint32(f)
	}
	t, err := g.uploadWords(words, []int{len(words)})
	if err != nil {
		return nil, err
	}
	return &Segments{t: t, rows: rows, segs: segs}, nil
}

// Rows reports how many packed rows the segmentation covers.
func (s *Segments) Rows() int { return s.rows }

// Len reports the number of segments.
func (s *Segments) Len() int { return s.segs }

// Free releases the device buffer. Calling Free again is a no-op.
func (s *Segments) Free() { s.t.Free() }

// segAttnParams mirrors SegAttnParams in segAttnWGSL; strides and columns
// count vec4s.
type segAttnParams struct {
	nq, heads, rows, segs         uint32
	qStride, qCol, kStride, kCol  uint32
	vStride, vCol, oStride, flags uint32
	scale                         float32
	_                             [3]uint32
	gate                          [segAttnMaxGateHeads]float32
}

const (
	segAttnParamBytes   = 192
	segAttnMaxGateHeads = 32
	segAttnMaxDH        = 128
	segAttnWG           = 64
	segAttnTileVec4s    = 256 // vec4s of a key (and a value) tile in workgroup memory: 4KiB each
)

// segAttnSpan is the longest run of key rows a workgroup stages in tiles:
// 64 consecutive queries of segments up to 96 rows long stay under it.
// Tests change it, before a device compiles the module, to run every
// workgroup down one path.
var segAttnSpan = 256

// segAttnWGSL is the module of seg_attn (no gate) and seg_attn_g (a gate)
// for heads of dh4 vec4s. One thread computes one (query, head) with an
// online softmax per key group -- the ungated keys and, under a gate, the
// gated ones -- so the scores never exist, and the query, the
// accumulators and the softmax state live in registers (the generator
// unrolls every loop over the head's vec4s). A group's running sum and
// accumulator are rescaled only when its maximum moves, a few times per
// query.
//
// The 64 threads of a workgroup take 64 consecutive queries of one head.
// Their segments cover one run of key rows, a little over 64 when every
// row is a query; the workgroup stages that run's keys, values and flags
// through workgroup memory a tile at a time, so each key row is read from
// the buffer once per workgroup instead of once per query, and each
// thread walks the part of every tile that lies in its own segment. When
// the run is long -- queries picked a few per segment by QRows -- most of
// a tile would belong to other threads' segments, and every thread reads
// its own keys from the buffers instead.
//
// The module is compiled without wgpu's runtime checks where the library
// allows: SegmentAttention checks every operand's extent on the host, the
// row words come from NewSegments, and the kernel checks the one index the
// host cannot see, a QRows entry.
func segAttnWGSL(dh4 int) string {
	tk := min(segAttnTileVec4s/dh4, segAttnWG)
	var b strings.Builder
	w := func(format string, args ...any) { fmt.Fprintf(&b, format, args...) }
	each := func(indent, format string) {
		for c := range dh4 {
			w(indent+format, c)
		}
	}
	w(`
struct SegAttnParams {
    nq: u32, heads: u32, rows: u32, segs: u32,
    q_stride: u32, q_col: u32, k_stride: u32, k_col: u32,
    v_stride: u32, v_col: u32, o_stride: u32, flags: u32,
    scale: f32, pad0: u32, pad1: u32, pad2: u32,
    gate: array<vec4<f32>, 8>,
}
@group(0) @binding(0) var<uniform> sp: SegAttnParams;
@group(0) @binding(1) var<storage, read> sq: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sk: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sseg: array<u32>;
@group(0) @binding(5) var<storage, read> sqrow: array<u32>;
@group(0) @binding(6) var<storage, read_write> sout: array<vec4<f32>>;

const TK = %[1]du;   // key rows per tile
const WG = %[2]du;   // queries per workgroup
const SPAN = %[3]du; // longest run of key rows staged in tiles
var<workgroup> tk_k: array<vec4<f32>, %[4]d>;
var<workgroup> tk_v: array<vec4<f32>, %[4]d>;
var<workgroup> tk_f: array<u32, TK>;
var<workgroup> wj: array<vec2<u32>, WG>;
var<workgroup> wspan: vec2<u32>;
`, tk, segAttnWG, segAttnSpan, tk*dh4)
	for _, gated := range []bool{false, true} {
		entry := "seg_attn"
		if gated {
			entry = "seg_attn_g"
		}
		w(`
@compute @workgroup_size(%[1]d, 1, 1)
fn %[2]s(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x * WG + lid.x;
    let h = wid.y;
    let hc = h * %[3]du;
    // The query's row word and key run; nothing when it has no query or
    // its row lies outside every segment.
    var rw = 0u;
    var j0 = 0u;
    var j1 = 0u;
    if (i < sp.nq) {
        var r = i;
        if ((sp.flags & 1u) != 0u) {
            r = sqrow[i];
        }
        if (r < sp.rows) {
            rw = sseg[sp.segs + 2u + r];
            j0 = sseg[rw >> 3u];
            j1 = sseg[(rw >> 3u) + 1u];
        }
    }
    wj[lid.x] = vec2<u32>(j0, j1);
    workgroupBarrier();
    if (lid.x == 0u) {
        var lo = 0xffffffffu;
        var hi = 0u;
        for (var t = 0u; t < WG; t = t + 1u) {
            let e = wj[t];
            if (e.y > e.x) {
                lo = min(lo, e.x);
                hi = max(hi, e.y);
            }
        }
        wspan = vec2<u32>(lo, max(lo, hi));
    }
    let span = workgroupUniformLoad(&wspan);
    var q0 = vec4<f32>(0.0);
`, segAttnWG, entry, dh4)
		for c := 1; c < dh4; c++ {
			w("    var q%d = vec4<f32>(0.0);\n", c)
		}
		w(`    if (j1 > j0) {
        let qb = i * sp.q_stride + sp.q_col + hc;
`)
		each("        ", "q%[1]d = sq[qb + %[1]du] * sp.scale;\n")
		w("    }\n")
		each("    ", "var a%[1]d = vec4<f32>(0.0);\n")
		w("    var m0 = -1e30;\n    var l0 = 0.0;\n")
		if gated {
			each("    ", "var g%[1]d = vec4<f32>(0.0);\n")
			w("    var m1 = -1e30;\n    var l1 = 0.0;\n")
		}
		// key writes the work of one key: its score against the query,
		// then the online softmax step of its group. kf, k(c) and v(c)
		// name its flags and its c-th key and value vec4s.
		key := func(indent, kf string, k, v func(c int) string) {
			w(indent+"if ((%s & 1u) == 0u) {\n", kf)
			in := indent + "    "
			w(in+"var x = dot(q0, %s);\n", k(0))
			for c := 1; c < dh4; c++ {
				w(in+"x = x + dot(q%d, %s);\n", c, k(c))
			}
			update := func(in, acc, m, l string) {
				w(in+"if (x > %[1]s) {\n", m)
				w(in+"    let f = exp(%[1]s - x);\n", m)
				w(in+"    %[1]s = %[1]s * f;\n", l)
				for c := range dh4 {
					w(in+"    %[1]s%[2]d = %[1]s%[2]d * f;\n", acc, c)
				}
				w(in+"    %[1]s = x;\n", m)
				w(in + "}\n")
				w(in+"let p = exp(x - %[1]s);\n", m)
				w(in+"%[1]s = %[1]s + p;\n", l)
				for c := range dh4 {
					w(in+"%[1]s%[2]d = %[1]s%[2]d + p * %[3]s;\n", acc, c, v(c))
				}
			}
			if gated {
				w(in+"if ((%s & 2u) != 0u) {\n", kf)
				update(in+"    ", "g", "m1", "l1")
				w(in + "} else {\n")
				update(in+"    ", "a", "m0", "l0")
				w(in + "}\n")
			} else {
				update(in, "a", "m0", "l0")
			}
			w(indent + "}\n")
		}
		w(`    if (span.y - span.x <= SPAN) {
        for (var t0 = span.x; t0 < span.y; t0 = t0 + TK) {
            let n = min(TK, span.y - t0);
            for (var e = lid.x; e < n * %[1]du; e = e + WG) {
                let row = t0 + e / %[1]du;
                let c = e %% %[1]du;
                tk_k[e] = sk[row * sp.k_stride + sp.k_col + hc + c];
                tk_v[e] = sv[row * sp.v_stride + sp.v_col + hc + c];
            }
            for (var e = lid.x; e < n; e = e + WG) {
                tk_f[e] = sseg[sp.segs + 2u + t0 + e];
            }
            workgroupBarrier();
            let e1 = min(j1, t0 + n);
            for (var j = max(j0, t0); j < e1; j = j + 1u) {
                let e = j - t0;
                let kb = e * %[1]du;
                let kf = tk_f[e];
`, dh4)
		key("                ", "kf",
			func(c int) string { return fmt.Sprintf("tk_k[kb + %du]", c) },
			func(c int) string { return fmt.Sprintf("tk_v[kb + %du]", c) })
		w(`            }
            workgroupBarrier();
        }
    } else {
        for (var j = j0; j < j1; j = j + 1u) {
            let kb = j * sp.k_stride + sp.k_col + hc;
            let vb = j * sp.v_stride + sp.v_col + hc;
            let kf = sseg[sp.segs + 2u + j];
`)
		key("            ", "kf",
			func(c int) string { return fmt.Sprintf("sk[kb + %du]", c) },
			func(c int) string { return fmt.Sprintf("sv[vb + %du]", c) })
		w(`        }
    }
    if (i >= sp.nq) {
        return;
    }
    let ob = i * sp.o_stride + hc;
`)
		if !gated {
			w("    var f0 = 0.0;\n    if (l0 > 0.0) {\n        f0 = 1.0 / l0;\n    }\n")
			each("    ", "sout[ob + %[1]du] = a%[1]d * f0;\n")
			w("}\n")
			continue
		}
		w(`    var f0 = 0.0;
    var f1 = 0.0;
    if ((rw & 4u) != 0u) {
        let m = max(m0, m1);
        let e0 = exp(m0 - m);
        let e1 = exp(m1 - m);
        let l = l0 * e0 + l1 * e1;
        if (l > 0.0) {
            f0 = e0 / l;
            f1 = e1 / l;
        }
    } else {
        if (l0 > 0.0) {
            f0 = 1.0 / l0;
        }
        if (l1 > 0.0) {
            f1 = sp.gate[h / 4u][h %% 4u] / l1;
        }
    }
`)
		each("    ", "sout[ob + %[1]du] = a%[1]d * f0 + g%[1]d * f1;\n")
		w("}\n")
	}
	return b.String()
}

// lazyPipe returns the pipeline of entry in the module src builds,
// compiling the module the first time mod is asked for and the pipeline
// the first time entry is: modules only a few models use stay out of
// Open. The module is compiled without wgpu's runtime checks where the
// library allows (makeModuleTrusted). A module or pipeline that failed
// once is not tried again. The caller holds g.mu and wgpuMu.
func (g *Device) lazyPipe(mod, entry string, src func() string) (gemmPipe, error) {
	key := mod + "." + entry
	if p, ok := g.pipes.lazy[key]; ok {
		if p.pipe == 0 {
			return p, fmt.Errorf("tensai: wgpu pipeline %q did not compile", entry)
		}
		return p, nil
	}
	if g.pipes.lazy == nil {
		g.pipes.lazy = make(map[string]gemmPipe)
		g.pipes.lazyMods = make(map[string]uintptr)
	}
	uncapturedCB = ""
	m, ok := g.pipes.lazyMods[mod]
	if !ok {
		m = g.makeModuleTrusted(src())
		if m != 0 && uncapturedCB != "" {
			fnShaderModRelease(m)
			m = 0
		}
		g.pipes.lazyMods[mod] = m
	}
	if m == 0 {
		g.pipes.lazy[key] = gemmPipe{}
		return gemmPipe{}, fmt.Errorf("tensai: wgpu module %s did not compile: %s", mod, uncapturedCB)
	}
	p := g.makePipelineIn(m, entry)
	if p == 0 || uncapturedCB != "" {
		if p != 0 {
			fnPipelineRelease(p)
		}
		g.pipes.lazy[key] = gemmPipe{}
		return gemmPipe{}, fmt.Errorf("tensai: wgpu pipeline %q creation failed: %s", entry, uncapturedCB)
	}
	q := gemmPipe{pipe: p, lay: fnPipelineGetLayout(p, 0)}
	g.pipes.lazy[key] = q
	return q, nil
}

// releaseLazy drops what lazyPipe compiled; the caller holds wgpuMu.
func (g *Device) releaseLazy() {
	for _, q := range g.pipes.lazy {
		if q.pipe != 0 {
			fnLayoutRelease(q.lay)
			fnPipelineRelease(q.pipe)
		}
	}
	for _, m := range g.pipes.lazyMods {
		if m != 0 {
			fnShaderModRelease(m)
		}
	}
	g.pipes.lazy, g.pipes.lazyMods = nil, nil
}

// rowWidth reports the last axis of t, the width of its rows, and how
// many rows it holds.
func rowWidth(t *Tensor) (width, rows int) {
	if len(t.shape) == 0 {
		return 0, 0
	}
	width = t.shape[len(t.shape)-1]
	if width == 0 {
		return 0, 0
	}
	return width, t.Size() / width
}

// SegmentAttention runs the multi-head attention a describes (see
// SegAttn) in one dispatch and returns its output, one row of a.Dim
// columns per query in the packed head layout -- ready for the output
// projection. Nothing is padded: the cost follows the rows of the
// segments, not the longest one. Heads of up to 128 dimensions run on
// the device; the fastest are those up to 64.
func (g *Device) SegmentAttention(a SegAttn) (*Tensor, error) {
	if a.Seg == nil || a.Q == nil || a.K == nil || a.V == nil {
		return nil, errors.New("tensai: SegmentAttention needs Seg, Q, K and V")
	}
	ops := []*Tensor{a.Seg.t, a.Q, a.K, a.V}
	if a.QRows != nil {
		ops = append(ops, a.QRows)
	}
	for _, t := range ops {
		if t.freed {
			return nil, errors.New("tensai: gpu tensor already freed")
		}
		if t.g != g {
			return nil, errors.New("tensai: gpu tensors belong to different GPUs")
		}
		if t.f16 {
			return nil, errors.New("tensai: SegmentAttention takes no half-precision tensor")
		}
	}
	qw, qRows := rowWidth(a.Q)
	kw, kRows := rowWidth(a.K)
	vw, vRows := rowWidth(a.V)
	dim := a.Dim
	if dim == 0 {
		dim = qw
	}
	heads := a.Heads
	if heads < 1 || dim < heads || dim%heads != 0 {
		return nil, fmt.Errorf("tensai: %d columns do not split into %d heads", dim, heads)
	}
	dh := dim / heads
	if dh%4 != 0 || dh > segAttnMaxDH {
		return nil, fmt.Errorf("tensai: SegmentAttention takes heads of a multiple of 4 up to %d dimensions, got %d", segAttnMaxDH, dh)
	}
	for _, x := range []struct {
		name       string
		width, col int
	}{{"q", qw, a.QCol}, {"k", kw, a.KCol}, {"v", vw, a.VCol}} {
		if x.width%4 != 0 || x.col%4 != 0 || x.col < 0 || x.col+dim > x.width {
			return nil, fmt.Errorf("tensai: %s columns [%d, %d) do not fit rows of %d as vec4s", x.name, x.col, x.col+dim, x.width)
		}
	}
	rows := a.Seg.rows
	if kRows < rows || vRows < rows {
		return nil, fmt.Errorf("tensai: k and v hold %d and %d rows, want %d", kRows, vRows, rows)
	}
	nq := rows
	if a.QRows != nil {
		nq = a.QRows.Size()
	}
	if qRows < nq {
		return nil, fmt.Errorf("tensai: q holds %d rows for %d queries", qRows, nq)
	}
	if nq < 1 {
		return nil, errors.New("tensai: SegmentAttention without queries")
	}
	blocks := (nq + segAttnWG - 1) / segAttnWG
	if blocks > 65535 || heads > 65535 {
		return nil, fmt.Errorf("tensai: %d queries of %d heads exceed one dispatch", nq, heads)
	}
	p := segAttnParams{
		nq: uint32(nq), heads: uint32(heads), rows: uint32(rows), segs: uint32(a.Seg.segs),
		qStride: uint32(qw / 4), qCol: uint32(a.QCol / 4),
		kStride: uint32(kw / 4), kCol: uint32(a.KCol / 4),
		vStride: uint32(vw / 4), vCol: uint32(a.VCol / 4),
		oStride: uint32(dim / 4),
		scale:   a.Scale,
	}
	if p.scale == 0 {
		p.scale = float32(1 / math.Sqrt(float64(dh)))
	}
	if a.QRows != nil {
		p.flags |= 1
	}
	if a.Gate != nil {
		if len(a.Gate) != heads || heads > segAttnMaxGateHeads {
			return nil, fmt.Errorf("tensai: %d gate weights for %d heads (at most %d)", len(a.Gate), heads, segAttnMaxGateHeads)
		}
		copy(p.gate[:], a.Gate)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""
	dh4 := dh / 4
	entry := "seg_attn"
	if a.Gate != nil {
		entry = "seg_attn_g"
	}
	pipe, err := g.lazyPipe(fmt.Sprintf("seg_attn/%d", dh4), entry, func() string { return segAttnWGSL(dh4) })
	if err != nil {
		return nil, err
	}
	outBytes := uint64(nq) * uint64(dim) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, err
	}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&p), segAttnParamBytes)
	defer release()
	bufOut := g.takeOutBuffer(outBytes)
	qrows := a.Seg.t // stands in for the unread binding
	if a.QRows != nil {
		qrows = a.QRows
	}
	entries := [7]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, offset: offParams, size: segAttnParamBytes},
		bind(1, a.Q), bind(2, a.K), bind(3, a.V),
		bind(4, a.Seg.t), bind(5, qrows),
		{binding: 6, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(pipe.lay, entries[:])
	runtime.KeepAlive(&entries)
	if err := g.dispatch(pipe.pipe, bindGroup, uint32(blocks), uint32(heads), 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: []int{nq, dim}}, nil
}
