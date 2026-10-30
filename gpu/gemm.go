//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
)

// The tiled GEMM kernels of gemmWGSL multiply a (m x k) by a (k x n), both
// row-major with k, n, the row strides, and the batch offsets multiples of
// four -- every product of a model whose widths are multiples of four.
// matmulWGSL's kernels handle everything else.
//
// Each workgroup of 256 threads computes a (16*tm) x (16*tn) block of the
// output, each thread a tm x tn register tile of it. A thread's rows are
// four consecutive ones, repeated every 64 rows when tm is 8, and so are
// its columns: the four columns of a vec4 are one 16-byte workgroup-memory
// load, and the sixteen threads along a row read sixteen consecutive
// vec4s, which no two banks share.
//
// The operands go through workgroup memory a bk-deep slab at a time,
// loaded from global memory as vec4s. b is staged as it is stored -- a k,
// then the block's columns -- so each global vec4 is copied over
// unchanged. a is staged either the same way (a row of the block, then
// its k values; the inner loop reads one vec4 per row and k quad) or
// k-major (gemmTile.at: the four k values of each loaded vec4 are
// scattered to four rows of the slab, and the inner loop reads a vec4 of
// four rows per k). The k-major layout needs only two a registers per k
// instead of eight per k quad, which pays off at 8x8 register tiles. The
// next slab's global loads are issued into registers before the current
// slab is multiplied, so their latency hides behind the arithmetic.
//
// Bounds checks are paid only where needed: a workgroup whose block lies
// inside the output and a slab that lies inside k load without checks;
// only the edge blocks and the last slab test each vec4. The module is
// compiled without wgpu's own runtime checks where the library allows
// (makeModuleTrusted), which gemmFits makes safe by checking every batch's
// extent against its binding before a dispatch.
//
// The epilogue optionally adds a bias row and applies an activation as it
// stores (gemmEpi), so a dense layer is one dispatch.

// gemmTile is one tiling of the tiled kernels.
type gemmTile struct {
	tm, tn int  // rows and columns of a thread's register tile: 4 or 8
	bk     int  // k depth of one workgroup-memory slab, a multiple of 4
	at     bool // stage a k-major
}

func (c gemmTile) bm() int { return 16 * c.tm }
func (c gemmTile) bn() int { return 16 * c.tn }

func (c gemmTile) entry() string {
	t := ""
	if c.at {
		t = "t"
	}
	return fmt.Sprintf("gemm_nn_%dx%dk%d%s", c.bm(), c.bn(), c.bk, t)
}

// gemmTiles are the tilings Open compiles, biggest block first; every one
// keeps its workgroup memory within WebGPU's default 16KiB. On an NVIDIA
// RTX PRO 4000 (Vulkan) 128x128 with a k-major a runs the big products
// fastest, 128x64 the middling ones, and 64x64 with 32-deep slabs the
// small ones; the plain 128x128 and the other k-major 64x64 variants never
// won. On an Apple M5 (Metal) every tiling runs within a few percent of
// the others.
var gemmTiles = []gemmTile{
	{tm: 8, tn: 8, bk: 16, at: true},
	{tm: 8, tn: 4, bk: 16, at: true},
	{tm: 4, tn: 4, bk: 32},
	{tm: 4, tn: 4, bk: 16},
}

// gemmEpi is what the tiled kernels do to a product as they store it:
// add a bias row (one value per output column, nil for none), then apply
// an activation.
type gemmEpi struct {
	bias   *Tensor
	act    Act
	hasAct bool
}

func (e gemmEpi) empty() bool { return e.bias == nil && !e.hasAct }

// flags packs the epilogue into the params block's last word: bit 0 adds
// the bias, the bits above hold the activation plus one (0 for none).
func (e gemmEpi) flags() uint32 {
	var f uint32
	if e.bias != nil {
		f |= 1
	}
	if e.hasAct {
		f |= (uint32(e.act) + 1) << 1
	}
	return f
}

// errNoTiledGemm reports that a product with an epilogue does not fit the
// tiled kernels; the caller applies the epilogue as separate operations.
var errNoTiledGemm = errors.New("tensai: product does not fit the tiled gemm kernels")

// gemmWGSL is the module holding one entry point per gemmTiles entry.
var gemmWGSL = buildGemmWGSL(gemmTiles)

func buildGemmWGSL(tiles []gemmTile) string {
	var b strings.Builder
	b.WriteString(`
struct GemmParams {
    m: u32, k: u32, n: u32, batches: u32,
    lda: u32, ldb: u32, ldc: u32, flags: u32,
}
@group(0) @binding(0) var<uniform> gp: GemmParams;
@group(0) @binding(1) var<storage, read> ga: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gb: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> goffs: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> gout: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> gbias: array<vec4<f32>>;

// gemm_erf is matmulWGSL's erff over a vec4 (Abramowitz-Stegun 7.1.26).
fn gemm_erf(x: vec4<f32>) -> vec4<f32> {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return sign(x) * y;
}

// gemm_epilogue applies gemmEpi: the bias, then the activation, which
// are act_of's (Act plus one; 0 for none).
fn gemm_epilogue(v: vec4<f32>, col4: u32) -> vec4<f32> {
    var r = v;
    if ((gp.flags & 1u) != 0u) {
        r = r + gbias[col4];
    }
    switch (gp.flags >> 1u) {
        case 0u: {}
        case 1u: { r = max(r, vec4<f32>(0.0)); }
        // Metal's tanh returns 0 or NaN past about 40; it is +-1 in f32
        // well before 15.
        case 2u: { r = tanh(clamp(r, vec4<f32>(-15.0), vec4<f32>(15.0))); }
        case 3u: { r = 1.0 / (1.0 + exp(-r)); }
        default: { r = 0.5 * r * (1.0 + gemm_erf(r * 0.7071067811865476)); }
    }
    return r;
}
`)
	for _, c := range tiles {
		writeGemmEntry(&b, c)
	}
	return b.String()
}

// writeGemmEntry emits one tiling's entry point, fully unrolled over the
// register tile: an array the compiler cannot prove it can unroll lands in
// scratch memory instead of registers.
func writeGemmEntry(b *strings.Builder, c gemmTile) {
	bm, bn := c.bm(), c.bn()
	kq := c.bk / 4      // vec4s along k in one a-slab row
	nq := bn / 4        // vec4s along n in one b-slab row
	la := bm * kq / 256 // a-slab vec4s each thread stages
	lb := c.bk * nq / 256
	if la < 1 || lb < 1 || bm*kq%256 != 0 || c.bk*nq%256 != 0 {
		panic(fmt.Sprintf("gpu: gemm tile %+v does not stage evenly over 256 threads", c))
	}
	name := c.entry()
	sa, sb := name+"_a", name+"_b"
	w := func(format string, args ...any) { fmt.Fprintf(b, format, args...) }
	comps := []string{"x", "y", "z", "w"}

	if c.at {
		// Scalars: a store to one component of a vec4 may compile to a
		// read-modify-write of the whole vec4, racing the neighbours'.
		w("\nvar<workgroup> %s: array<f32, %d>;\n", sa, bm*c.bk)
		w("fn %s_v4(i: u32) -> vec4<f32> { return vec4<f32>(%s[i], %s[i + 1u], %s[i + 2u], %s[i + 3u]); }\n", sa, sa, sa, sa, sa)
	} else {
		w("\nvar<workgroup> %s: array<vec4<f32>, %d>;\n", sa, bm*kq)
	}
	w("var<workgroup> %s: array<vec4<f32>, %d>;\n", sb, c.bk*nq)
	w(`
@compute @workgroup_size(256, 1, 1)
fn %s(@builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = li %% 16u;
    let ty = li / 16u;
    let o = goffs[wid.z];
    let rowBlock = wid.y * %du;
    let colBlock = wid.x * %du;
    let lda4 = gp.lda / 4u;
    let ldb4 = gp.ldb / 4u;
    let edge = rowBlock + %du > gp.m || colBlock + %du > gp.n;
`, name, bm, bn, bm, bn)
	// Staging assignment: thread li moves vec4 li + s*256 of each slab --
	// row am, k quad aq of a; k row bk, column quad bq of b.
	for s := 0; s < la; s++ {
		w("    let am%d = (li + %du) / %du;\n", s, s*256, kq)
		w("    let aq%d = (li + %du) %% %du;\n", s, s*256, kq)
		w("    let aOk%d = rowBlock + am%d < gp.m;\n", s, s)
		w("    let aOff%d = o.x / 4u + (rowBlock + am%d) * lda4 + aq%d;\n", s, s, s)
		w("    var ra%d = vec4<f32>(0.0);\n", s)
	}
	for s := 0; s < lb; s++ {
		w("    let bk%d = (li + %du) / %du;\n", s, s*256, nq)
		w("    let bq%d = (li + %du) %% %du;\n", s, s*256, nq)
		w("    let bOk%d = colBlock + bq%d * 4u < gp.n;\n", s, s)
		w("    let bOff%d = o.y / 4u + bk%d * ldb4 + colBlock / 4u + bq%d;\n", s, s, s)
		w("    var rb%d = vec4<f32>(0.0);\n", s)
	}
	for i := 0; i < c.tm; i++ {
		for j := 0; j < c.tn/4; j++ {
			w("    var acc%d_%d = vec4<f32>(0.0);\n", i, j)
		}
	}
	w("    let tiles = (gp.k + %du) / %du;\n", c.bk-1, c.bk)
	// The thread's rows in a row-major a slab.
	for i := 0; i < c.tm && !c.at; i++ {
		w("    let ar%d = (%du + ty * 4u + %du) * %du;\n", i, (i/4)*64, i%4, kq)
	}

	// load fetches slab t's vec4s into the staging registers.
	load := func(t string) {
		w("    {\n        let k0 = %s * %du;\n", t, c.bk)
		w("        if (!edge && k0 + %du <= gp.k) {\n", c.bk)
		for s := 0; s < la; s++ {
			w("            ra%d = ga[aOff%d + %s * %du];\n", s, s, t, kq)
		}
		for s := 0; s < lb; s++ {
			w("            rb%d = gb[bOff%d + k0 * ldb4];\n", s, s)
		}
		w("        } else {\n")
		for s := 0; s < la; s++ {
			w("            ra%d = vec4<f32>(0.0);\n", s)
			w("            if (aOk%d && k0 + aq%d * 4u < gp.k) { ra%d = ga[aOff%d + %s * %du]; }\n", s, s, s, s, t, kq)
		}
		for s := 0; s < lb; s++ {
			w("            rb%d = vec4<f32>(0.0);\n", s)
			w("            if (bOk%d && k0 + bk%d < gp.k) { rb%d = gb[bOff%d + k0 * ldb4]; }\n", s, s, s, s)
		}
		w("        }\n    }\n")
	}

	load("0u")
	w("    for (var t = 0u; t < tiles; t = t + 1u) {\n")
	for s := 0; s < la; s++ {
		if !c.at {
			w("        %s[li + %du] = ra%d;\n", sa, s*256, s)
			continue
		}
		for j := 0; j < 4; j++ {
			w("        %s[(aq%d * 4u + %du) * %du + am%d] = ra%d.%s;\n", sa, s, j, bm, s, s, comps[j])
		}
	}
	for s := 0; s < lb; s++ {
		w("        %s[li + %du] = rb%d;\n", sb, s*256, s)
	}
	w("        workgroupBarrier();\n")
	w("        if (t + 1u < tiles) {\n            let tn = t + 1u;\n")
	load("tn")
	w("        }\n")
	w("        for (var kq = 0u; kq < %du; kq = kq + 1u) {\n", kq)
	for i := 0; i < c.tm && !c.at; i++ {
		w("            let a%d = %s[ar%d + kq];\n", i, sa, i)
	}
	for kk := 0; kk < 4; kk++ {
		for ch := 0; ch < c.tm/4 && c.at; ch++ {
			w("            let av%d_%d = %s_v4((kq * 4u + %du) * %du + %du + ty * 4u);\n", kk, ch, sa, kk, bm, ch*64)
		}
		for j := 0; j < c.tn/4; j++ {
			w("            let b%d_%d = %s[(kq * 4u + %du) * %du + %du + tx];\n", kk, j, sb, kk, nq, j*16)
		}
		for i := 0; i < c.tm; i++ {
			a := fmt.Sprintf("a%d.%s", i, comps[kk])
			if c.at {
				a = fmt.Sprintf("av%d_%d.%s", kk, i/4, comps[i%4])
			}
			for j := 0; j < c.tn/4; j++ {
				w("            acc%d_%d = fma(vec4<f32>(%s), b%d_%d, acc%d_%d);\n", i, j, a, kk, j, i, j)
			}
		}
	}
	w("        }\n        workgroupBarrier();\n    }\n")

	// Epilogue: n is a multiple of four, so a vec4 is wholly in or out.
	w("    let ldc4 = gp.ldc / 4u;\n")
	for i := 0; i < c.tm; i++ {
		w("    {\n        let row = rowBlock + %du + ty * 4u + %du;\n", (i/4)*64, i%4)
		w("        if (row < gp.m) {\n")
		for j := 0; j < c.tn/4; j++ {
			w("            { let col4 = colBlock / 4u + %du + tx;\n", j*16)
			w("              if (col4 * 4u < gp.n) { gout[o.z / 4u + row * ldc4 + col4] = gemm_epilogue(acc%d_%d, col4); } }\n", i, j)
		}
		w("        }\n    }\n")
	}
	w("}\n")
}

// gemmPipe is one compiled tiling.
type gemmPipe struct{ pipe, lay uintptr }

// Knobs for tests and benchmarks; production code leaves them alone.
var (
	gemmOff     bool // route every product to matmulWGSL's kernels
	gemmForce   = -1 // index into gemmTiles to use whenever the tiled path fits
	gemmChecked bool // compile the module with wgpu's runtime checks (read at Open)
)

// initGemm compiles the tiled GEMM module and its pipelines. Failure is
// not an error: the products then run on matmulWGSL's kernels. The caller
// holds wgpuMu.
func (g *Device) initGemm() {
	uncapturedCB = ""
	g.module4 = g.makeModuleTrusted(gemmWGSL)
	if g.module4 == 0 || uncapturedCB != "" {
		return
	}
	pipes := make([]gemmPipe, len(gemmTiles))
	for i, c := range gemmTiles {
		p := g.makePipelineIn(g.module4, c.entry())
		if p == 0 || uncapturedCB != "" {
			if p != 0 {
				fnPipelineRelease(p)
			}
			for _, q := range pipes[:i] {
				fnLayoutRelease(q.lay)
				fnPipelineRelease(q.pipe)
			}
			return
		}
		pipes[i] = gemmPipe{pipe: p, lay: fnPipelineGetLayout(p, 0)}
	}
	g.pipes.gemm = pipes
}

func (g *Device) releaseGemm() {
	for _, q := range g.pipes.gemm {
		fnLayoutRelease(q.lay)
		fnPipelineRelease(q.pipe)
	}
	g.pipes.gemm = nil
}

// gemmFits reports whether the tiled kernels should run this product, and
// with which tiling. Their loads and stores are vec4s, and the module may
// be compiled without wgpu's bounds checks, so every batch's last element
// must lie inside its binding. Skinny plain products (m or n under 32)
// stay on matmulWGSL's 16x16 kernels, whose grid fits them better; with
// an epilogue (fused) the tiled kernels take them too.
func (g *Device) gemmFits(a, b, bias *Tensor, m, k, n, batches, lda, ldb, ldc int, outBytes uint64, offs []uint32, fused bool) (int, bool) {
	if gemmOff || g.pipes.gemm == nil || m < 1 || k < 1 || n < 1 {
		return 0, false
	}
	if k%4 != 0 || n%4 != 0 || lda%4 != 0 || ldb%4 != 0 || ldc%4 != 0 ||
		lda < k || ldb < n || ldc < n || a.Size()%4 != 0 || b.Size()%4 != 0 {
		return 0, false
	}
	if (m+63)/64 > 65535 || (n+63)/64 > 65535 { // the dispatch grid's limit
		return 0, false
	}
	if bias != nil && (bias.Size() < n || bias.Size()%4 != 0) {
		return 0, false
	}
	outSize := int(outBytes / 4)
	for i := 0; i+2 < len(offs); i += 4 {
		oa, ob, oc := int(offs[i]), int(offs[i+1]), int(offs[i+2])
		if oa%4 != 0 || ob%4 != 0 || oc%4 != 0 ||
			oa+(m-1)*lda+k > a.Size() || ob+(k-1)*ldb+n > b.Size() || oc+(m-1)*ldc+n > outSize {
			return 0, false
		}
	}
	if gemmForce >= 0 {
		return gemmForce, true
	}
	if !fused && (m < 32 || n < 32) {
		return 0, false
	}
	return pickGemmTile(m, k, n, batches), true
}

// gemmMinBlocks is how many workgroups a tiling must give a product
// before pickGemmTile takes it: about one per core of a mid-range
// discrete GPU, below which the device idles.
const gemmMinBlocks = 64

// pickGemmTile picks the biggest block that still gives the device
// gemmMinBlocks workgroups, and among the 64x64 tilings the 32-deep slab
// when k fills it.
func pickGemmTile(m, k, n, batches int) int {
	for i, c := range gemmTiles[:2] {
		if ((m+c.bm()-1)/c.bm())*((n+c.bn()-1)/c.bn())*batches >= gemmMinBlocks {
			return i
		}
	}
	if k > 16 {
		return 2
	}
	return 3
}

// dispatchGemm records one tiled product; bias stands in for the bias
// binding even when the flags leave it unread. The caller holds g.mu and
// wgpuMu and has written params and offs.
func (g *Device) dispatchGemm(tile int, bufParams, bufOffs uintptr, offsBytes uint64, a, b, bias *Tensor, bufOut uintptr, outBytes uint64, m, n, batches int) error {
	c, p := gemmTiles[tile], g.pipes.gemm[tile]
	entries := []wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, size: 32},
		bind(1, a), bind(2, b),
		{binding: 3, buffer: bufOffs, size: offsBytes},
		{binding: 4, buffer: bufOut, size: outBytes},
		bind(5, bias),
	}
	bindGroup := g.cachedBindGroup(p.lay, entries)
	runtime.KeepAlive(&entries)
	return g.dispatch(p.pipe, bindGroup,
		uint32((n+c.bn()-1)/c.bn()), uint32((m+c.bm()-1)/c.bm()), uint32(batches))
}

// Linear computes act(t*w + bias), the forward pass of a dense layer: t
// (..., m, k) times w (k, n) -- broadcasting over the batch axes as MatMul
// does -- plus bias, a row of n values (shape (n) or (1, n)) added to every
// output row, then the activation if one is given. bias may be nil. When
// the tiled kernels take the product (k and n multiples of four) it is a
// single dispatch that writes the finished output; otherwise it runs as
// MatMul, Binary, and Activate.
func (t *Tensor) Linear(w, bias *Tensor, act ...Act) (*Tensor, error) {
	if len(act) > 1 {
		return nil, fmt.Errorf("tensai: Linear takes at most one activation, got %d", len(act))
	}
	epi := gemmEpi{bias: bias}
	if len(act) == 1 {
		epi.act, epi.hasAct = act[0], true
	}
	if bias != nil {
		if bias.freed {
			return nil, errors.New("tensai: gpu tensor already freed")
		}
		if bias.g != t.g {
			return nil, errors.New("tensai: gpu tensors belong to different GPUs")
		}
		n := 0
		if len(w.shape) > 0 {
			n = w.shape[len(w.shape)-1]
		}
		if bias.f16 || bias.Size() != n || (len(bias.shape) > 1 && bias.shape[len(bias.shape)-1] != n) {
			return nil, fmt.Errorf("tensai: Linear bias %v does not match %d output columns", bias.shape, n)
		}
	}
	out, err := t.matmul(w, gemmNN, epi)
	if !errors.Is(err, errNoTiledGemm) {
		return out, err
	}
	if out, err = t.MatMul(w); err != nil {
		return nil, err
	}
	if bias != nil {
		sum, err := out.Binary(OpAdd, bias)
		out.Free()
		if err != nil {
			return nil, err
		}
		out = sum
	}
	if epi.hasAct {
		y, err := out.Activate(epi.act)
		out.Free()
		if err != nil {
			return nil, err
		}
		out = y
	}
	return out, nil
}
