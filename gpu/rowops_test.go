//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/mattn/tensai/internal/kernels"
)

// TestGPULayerNormRows checks LayerNorm against the CPU kernel on the
// widths ln_rows takes (multiples of four up to 1024), with row counts
// that leave a workgroup's eight rows partly empty and past one row of
// the dispatch grid, and on widths that stay on ln_fwd.
func TestGPULayerNormRows(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(83, 0))
	for _, c := range [][2]int{
		{1, 4}, {7, 12}, {9, 128}, {1000, 128}, {65, 132}, {3, 1024}, {600000, 4}, // ln_rows
		{5, 1028}, {6, 130}, {2, 2048}, // ln_fwd
	} {
		rows, cols := c[0], c[1]
		x := randFloats(rng, rows*cols)
		for i := range x {
			x[i] = x[i]*3 + 5 // off-centre, so the mean matters
		}
		gain, bias := randFloats(rng, cols), randFloats(rng, cols)
		gx := uploadFloats(t, g, x, rows, cols)
		gg, gb := uploadFloats(t, g, gain, cols), uploadFloats(t, g, bias, cols)
		out, err := gx.LayerNorm(gg, gb, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		if s := out.Shape(); !slices.Equal(s, []int{rows, cols}) {
			t.Fatalf("%dx%d: shape %v", rows, cols, s)
		}
		got := make([]float32, rows*cols)
		if err := out.DownloadInto(got); err != nil {
			t.Fatal(err)
		}
		want, xhat := make([]float32, rows*cols), make([]float32, cols)
		for r := range rows {
			kernels.LnFwdRow(want[r*cols:(r+1)*cols], xhat, x[r*cols:(r+1)*cols], gain, bias, 1e-5)
		}
		checkFloats(t, fmt.Sprintf("%dx%d", rows, cols), got, want, 1e-4)
		for _, x := range []*Tensor{gx, gg, gb, out} {
			x.Free()
		}
	}
}

// TestGPUPairDot checks PairDot with and without index lists.
func TestGPUPairDot(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(73, 0))
	const cols = 36
	a, b := randFloats(rng, 50*cols), randFloats(rng, 70*cols)
	ga, gb := uploadFloats(t, g, a, 50, cols), uploadFloats(t, g, b, 70, cols)
	defer ga.Free()
	defer gb.Free()
	ia, ib := make([]int, 300), make([]int, 300)
	for i := range ia {
		ia[i], ib[i] = rng.IntN(50), rng.IntN(70)
	}
	ib[7] = 70 // past the rows: 0
	gia, err := g.UploadIndices(ia)
	if err != nil {
		t.Fatal(err)
	}
	defer gia.Free()
	gib, err := g.UploadIndices(ib)
	if err != nil {
		t.Fatal(err)
	}
	defer gib.Free()
	dot := func(ra, rb int) float32 {
		if ra >= 50 || rb >= 70 {
			return 0
		}
		var s float64
		for c := range cols {
			s += float64(a[ra*cols+c]) * float64(b[rb*cols+c])
		}
		return float32(s * 0.25)
	}
	for _, c := range []struct {
		name   string
		ia, ib *Tensor
		n      int
		pair   func(i int) (int, int)
	}{
		{"both", gia, gib, 300, func(i int) (int, int) { return ia[i], ib[i] }},
		{"a only", gia, nil, 300, func(i int) (int, int) { return ia[i], i }},
		{"b only", nil, gib, 300, func(i int) (int, int) { return i, ib[i] }},
	} {
		out, err := ga.PairDot(gb, c.ia, c.ib, 0.25)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got := make([]float32, c.n)
		if err := out.DownloadInto(got); err != nil {
			t.Fatal(err)
		}
		out.Free()
		want := make([]float32, c.n)
		for i := range want {
			want[i] = dot(c.pair(i))
		}
		checkFloats(t, c.name, got, want, 1e-5)
	}
	if _, err := ga.PairDot(gb, nil, nil, 1); err == nil {
		t.Error("rows of 50 and 70 without index lists: no error")
	}
	short, err := g.UploadIndices(ia[:10])
	if err != nil {
		t.Fatal(err)
	}
	defer short.Free()
	if _, err := ga.PairDot(gb, gia, short, 1); err == nil {
		t.Error("index lists of 300 and 10: no error")
	}
	gc := uploadFloats(t, g, b[:70*32], 70, 32)
	defer gc.Free()
	if _, err := ga.PairDot(gc, gia, gib, 1); err == nil {
		t.Error("rows of 36 and 32 columns: no error")
	}
}

// TestGPUDownloadInto checks the allocation-free downloads against
// Download, a view's offset included, and their size checks.
func TestGPUDownloadInto(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(79, 0))
	const rows, cols = 300, 20
	x := randFloats(rng, rows*cols)
	gx := uploadFloats(t, g, x, rows, cols)
	defer gx.Free()

	got := make([]float32, rows*cols)
	if err := gx.DownloadInto(got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, x) {
		t.Fatal("DownloadInto differs from the upload")
	}
	full, err := gx.Download()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(full.Data, x) {
		t.Fatal("Download differs from the upload")
	}
	part, err := gx.DownloadRange(130, 77)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(part.Data, x[130:207]) {
		t.Fatal("DownloadRange differs from the upload")
	}
	dst := make([]float32, 77)
	if err := gx.DownloadRangeInto(130, dst); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(dst, x[130:207]) {
		t.Fatal("DownloadRangeInto differs from the upload")
	}

	view, err := gx.View(64*4, 10, cols) // 256 floats in: a 1KiB-aligned window
	if err != nil {
		t.Fatal(err)
	}
	vd := make([]float32, 10*cols)
	if err := view.DownloadInto(vd); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(vd, x[256:256+10*cols]) {
		t.Fatal("DownloadInto of a view misses its offset")
	}

	ids := []int{299, 0, 17, 17, 150}
	gids, err := g.UploadIndices(ids)
	if err != nil {
		t.Fatal(err)
	}
	defer gids.Free()
	sel := make([]float32, len(ids)*cols)
	if err := gx.DownloadRowsInto(gids, sel); err != nil {
		t.Fatal(err)
	}
	for i, r := range ids {
		if !slices.Equal(sel[i*cols:(i+1)*cols], x[r*cols:(r+1)*cols]) {
			t.Fatalf("DownloadRowsInto row %d (%d) differs", i, r)
		}
	}

	if err := gx.DownloadInto(make([]float32, rows*cols-1)); err == nil {
		t.Error("DownloadInto into a short buffer: no error")
	}
	if err := gx.DownloadInto(make([]float32, rows*cols+1)); err == nil {
		t.Error("DownloadInto into a long buffer: no error")
	}
	if err := gx.DownloadRangeInto(rows*cols-5, make([]float32, 6)); err == nil {
		t.Error("DownloadRangeInto past the end: no error")
	}
	if err := gx.DownloadRowsInto(gids, make([]float32, len(ids)*cols-1)); err == nil {
		t.Error("DownloadRowsInto into a short buffer: no error")
	}
	gx2 := uploadFloats(t, g, x[:cols], 1, cols)
	gx2.Free()
	if err := gx2.DownloadInto(make([]float32, cols)); err == nil {
		t.Error("DownloadInto of a freed tensor: no error")
	}
}
