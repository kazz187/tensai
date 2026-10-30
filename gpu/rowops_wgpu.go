//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/mattn/tensai"
)

// rowOpsWGSL holds the row kernels compiled on first use: pair_dot, one
// thread per pair with a plain loop over the columns (the pairs of a
// model head are a few thousand short rows), and ln_rows, the LayerNorm
// of rows of up to 1024 columns, eight threads to a row. Each entry point
// has bindings of its own, so the auto layouts stay independent.
const rowOpsWGSL = `
struct PairParams {
    n: u32, cols: u32, a_rows: u32, b_rows: u32,
    flags: u32, scale: f32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<uniform> pp: PairParams;
@group(0) @binding(1) var<storage, read> pa: array<f32>;
@group(0) @binding(2) var<storage, read> pb: array<f32>;
@group(0) @binding(3) var<storage, read> pia: array<u32>;
@group(0) @binding(4) var<storage, read> pib: array<u32>;
@group(0) @binding(5) var<storage, read_write> pout: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn pair_dot(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    if (i >= pp.n) {
        return;
    }
    var ra = i;
    if ((pp.flags & 1u) != 0u) {
        ra = pia[i];
    }
    var rb = i;
    if ((pp.flags & 2u) != 0u) {
        rb = pib[i];
    }
    if (ra >= pp.a_rows || rb >= pp.b_rows) {
        pout[i] = 0.0;
        return;
    }
    let ab = ra * pp.cols;
    let bb = rb * pp.cols;
    var s = 0.0;
    for (var c = 0u; c < pp.cols; c = c + 1u) {
        s = s + pa[ab + c] * pb[bb + c];
    }
    pout[i] = s * pp.scale;
}

struct LnParams {
    rows: u32, cols4: u32, eps: f32, pad0: u32,
}
@group(0) @binding(8) var<uniform> lp: LnParams;
@group(0) @binding(9) var<storage, read> lx: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> lg: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> lb: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> lout: array<vec4<f32>>;
var<workgroup> lpart: array<f32, 64>;

// ln_rows gives each row eight consecutive threads, which read the row's
// vec4s in turn -- eight to a 128-byte line -- and sum their parts in
// workgroup memory: the mean first, then the variance around it, as
// ln_stats does.
@compute @workgroup_size(64, 1, 1)
fn ln_rows(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let rl = lid.x / 8u;
    let lane = lid.x % 8u;
    let row = (wid.y * nwg.x + wid.x) * 8u + rl;
    let live = row < lp.rows;
    let base = row * lp.cols4;
    let n = f32(lp.cols4 * 4u);
    var s = vec4<f32>(0.0);
    if (live) {
        for (var c = lane; c < lp.cols4; c = c + 8u) {
            s = s + lx[base + c];
        }
    }
    lpart[lid.x] = s.x + s.y + s.z + s.w;
    workgroupBarrier();
    var tot = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        tot = tot + lpart[rl * 8u + k];
    }
    let mean = tot / n;
    workgroupBarrier();
    var q = vec4<f32>(0.0);
    if (live) {
        for (var c = lane; c < lp.cols4; c = c + 8u) {
            let d = lx[base + c] - vec4<f32>(mean);
            q = q + d * d;
        }
    }
    lpart[lid.x] = q.x + q.y + q.z + q.w;
    workgroupBarrier();
    tot = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        tot = tot + lpart[rl * 8u + k];
    }
    let inv = 1.0 / sqrt(tot / n + lp.eps);
    if (live) {
        for (var c = lane; c < lp.cols4; c = c + 8u) {
            lout[base + c] = (lx[base + c] - vec4<f32>(mean)) * inv * lg[c] + lb[c];
        }
    }
}
`

// pairDotParams mirrors PairParams in pairDotWGSL.
type pairDotParams struct {
	n, cols, aRows, bRows uint32
	flags                 uint32
	scale                 float32
	_                     [2]uint32
}

const pairDotParamBytes = 32

// PairDot returns the n scaled dot products scale * <t[ia[i]], o[ib[i]]>
// as a tensor of shape (n): t and o hold rows of the same width, and ia
// and ib are UploadIndices tensors naming a row of each per pair. A nil
// index list means row i; n is the length of the lists, or the rows of
// t when both are nil. An index past its tensor's rows gives 0. A pointer
// head scores every option against its segment's query this way and
// downloads only the scores.
func (t *Tensor) PairDot(o, ia, ib *Tensor, scale float32) (*Tensor, error) {
	if o == nil {
		return nil, errors.New("tensai: PairDot needs a second tensor")
	}
	ops := []*Tensor{t, o}
	for _, x := range []*Tensor{ia, ib} {
		if x != nil {
			ops = append(ops, x)
		}
	}
	for _, x := range ops {
		if x.freed {
			return nil, errors.New("tensai: gpu tensor already freed")
		}
		if x.g != t.g {
			return nil, errors.New("tensai: gpu tensors belong to different GPUs")
		}
		if x.f16 {
			return nil, errors.New("tensai: PairDot takes no half-precision tensor")
		}
	}
	cols, aRows := rowWidth(t)
	ocols, bRows := rowWidth(o)
	if cols == 0 || cols != ocols {
		return nil, fmt.Errorf("tensai: PairDot rows of %d and %d columns", cols, ocols)
	}
	n := -1
	var p pairDotParams
	if ia != nil {
		n = ia.Size()
		p.flags |= 1
	}
	if ib != nil {
		if n >= 0 && ib.Size() != n {
			return nil, fmt.Errorf("tensai: PairDot index lists of %d and %d pairs", n, ib.Size())
		}
		n = ib.Size()
		p.flags |= 2
	}
	if n < 0 {
		if aRows != bRows {
			return nil, fmt.Errorf("tensai: PairDot of %d and %d rows without index lists", aRows, bRows)
		}
		n = aRows
	}
	if n < 1 {
		return nil, errors.New("tensai: PairDot without pairs")
	}
	p.n, p.cols, p.aRows, p.bRows, p.scale = uint32(n), uint32(cols), uint32(aRows), uint32(bRows), scale

	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""
	pipe, err := g.lazyPipe("rowops", "pair_dot", func() string { return rowOpsWGSL })
	if err != nil {
		return nil, err
	}
	outBytes := uint64(n) * 4
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&p), pairDotParamBytes)
	defer release()
	bufOut := g.takeOutBuffer(outBytes)
	if ia == nil { // stand-ins for the unread bindings
		ia = t
	}
	if ib == nil {
		ib = o
	}
	entries := [6]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, offset: offParams, size: pairDotParamBytes},
		bind(1, t), bind(2, o), bind(3, ia), bind(4, ib),
		{binding: 5, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(pipe.lay, entries[:])
	runtime.KeepAlive(&entries)
	x, y := split2D((n + 63) / 64)
	if err := g.dispatch(pipe.pipe, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, err
	}
	return &Tensor{g: g, buf: bufOut, shape: []int{n}}, nil
}

// lnRowsMaxCols is the widest row LayerNorm gives ln_rows; wider rows keep
// ln_fwd's 256 threads to a row.
const lnRowsMaxCols = 1024

// layerNormRows runs LayerNorm on ln_rows when the rows suit it: widths
// that are multiples of four up to lnRowsMaxCols -- a small model's
// hidden state, where ln_fwd would leave most of its 256 threads idle and
// pay for its reductions on every row. ok is false when the rows do not
// suit it or the module did not compile; LayerNorm then runs ln_fwd.
func (t *Tensor) layerNormRows(gain, bias *Tensor, eps tensai.Float, rows, cols int) (out *Tensor, ok bool, err error) {
	if cols%4 != 0 || cols > lnRowsMaxCols || t.f16 || gain.f16 || bias.f16 || gain.freed || bias.freed ||
		gain.g != t.g || bias.g != t.g {
		return nil, false, nil
	}
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, true, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""
	pipe, err := g.lazyPipe("rowops", "ln_rows", func() string { return rowOpsWGSL })
	if err != nil {
		return nil, false, nil
	}
	outBytes := uint64(rows) * uint64(cols) * 4
	if err := g.checkSize(outBytes); err != nil {
		return nil, true, err
	}
	p := [4]uint32{uint32(rows), uint32(cols / 4), math.Float32bits(float32(eps))}
	bufParams, offParams, release := g.paramsBuffer(unsafe.Pointer(&p[0]), 16)
	defer release()
	bufOut := g.takeOutBuffer(outBytes)
	entries := [5]wgpuBindGroupEntry{
		{binding: 8, buffer: bufParams, offset: offParams, size: 16},
		bind(9, t), bind(10, gain), bind(11, bias),
		{binding: 12, buffer: bufOut, size: outBytes},
	}
	bindGroup := g.cachedBindGroup(pipe.lay, entries[:])
	runtime.KeepAlive(&entries)
	x, y := split2D((rows + 7) / 8)
	if err := g.dispatch(pipe.pipe, bindGroup, x, y, 1); err != nil {
		g.dropBuffer(bufOut)
		return nil, true, err
	}
	return &Tensor{g: g, buf: bufOut, shape: append([]int(nil), t.shape...)}, true, nil
}

// DownloadInto copies the tensor into dst, which must hold exactly Size()
// elements, instead of allocating a host tensor: a loop that downloads
// every step reuses one buffer. Like Download it flushes an open batch
// first.
func (t *Tensor) DownloadInto(dst []tensai.Float) error {
	if !t.freed && len(dst) != t.Size() {
		return fmt.Errorf("tensai: download of %d elements into %d", t.Size(), len(dst))
	}
	return t.DownloadRangeInto(0, dst)
}

// DownloadRangeInto copies elements [off, off+len(dst)) into dst -- the
// allocation-free DownloadRange.
func (t *Tensor) DownloadRangeInto(off int, dst []tensai.Float) error {
	if t.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	if t.f16 {
		return errors.New("tensai: cannot download a half-precision tensor into floats")
	}
	if off < 0 || len(dst) == 0 || off+len(dst) > t.Size() {
		return fmt.Errorf("tensai: download of %d elements at %d overflows %d", len(dst), off, t.Size())
	}
	return t.readInto(off, dst)
}

// DownloadRowsInto gathers rows of t -- rows of its last axis, the ones
// the UploadIndices tensor ids names, in that order -- into dst, which
// must hold len(ids) rows: only those rows cross the bus. Every index must
// be below the number of rows.
func (t *Tensor) DownloadRowsInto(ids *Tensor, dst []tensai.Float) error {
	if t.freed || ids == nil || ids.freed {
		return errors.New("tensai: gpu tensor already freed")
	}
	if t.g != ids.g {
		return errors.New("tensai: gpu tensors belong to different GPUs")
	}
	if t.f16 {
		return errors.New("tensai: cannot download a half-precision tensor into floats")
	}
	cols, _ := rowWidth(t)
	n := ids.Size()
	if cols == 0 || len(dst) != n*cols {
		return fmt.Errorf("tensai: %d rows of %d columns do not fill %d elements", n, cols, len(dst))
	}
	p := trainParams{count: uint32(n * cols), aCount: uint32(cols), bCount: uint32(n)}
	rows, err := t.g.embedOp(t.g.pipes.embedGather, t.g.pipes.layEmbedGather, p, n*cols, []int{n, cols}, t, ids, nil, 0)
	if err != nil {
		return err
	}
	defer rows.Free()
	return rows.readInto(0, dst)
}
