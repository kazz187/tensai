//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai"
)

// benchLayout packs b positions of 40 to 89 rows (56 on average, the
// token count of a format-v2 game position) with their effect rows gated
// and about eight output rows each.
func benchLayout(rng *rand.Rand, b int) segLayout {
	var l segLayout
	l.off = append(l.off, 0)
	for range b {
		n := 40 + rng.IntN(50)
		base := l.rows
		l.rows += n
		l.off = append(l.off, l.rows)
		for j := range n {
			var f RowFlag
			if j >= 20 && j < 30 {
				f = RowGated | RowSeesAll
			}
			l.flags = append(l.flags, f)
		}
		l.outs = append(l.outs, base)
		for j := base + n - 7; j < base+n; j++ {
			l.outs = append(l.outs, j)
		}
	}
	return l
}

// syncGPU waits for the device by reading one element back.
func syncGPU(b *testing.B, t *Tensor) {
	var one [1]tensai.Float
	if err := t.DownloadRangeInto(0, one[:]); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkGPUSegmentAttention times one gated attention of a width-128,
// four-head layer over b positions: all rows (an inner layer) and the
// output rows only (the last layer).
func BenchmarkGPUSegmentAttention(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	const heads, d = 4, 128
	for _, positions := range []int{64, 256, 1024} {
		for _, subset := range []bool{false, true} {
			b.Run(fmt.Sprintf("positions=%d/subset=%v", positions, subset), func(b *testing.B) {
				rng := rand.New(rand.NewPCG(1, 0))
				l := benchLayout(rng, positions)
				seg, err := g.NewSegments(l.rows, l.off, l.flags)
				if err != nil {
					b.Fatal(err)
				}
				defer seg.Free()
				qkv, err := g.Upload(randTensor(rng, l.rows, 3*d))
				if err != nil {
					b.Fatal(err)
				}
				defer qkv.Free()
				a := SegAttn{Seg: seg, Heads: heads, Dim: d, Q: qkv, K: qkv, V: qkv, KCol: d, VCol: 2 * d, Gate: []float32{0.1, -0.2, 0.3, 0}}
				if subset {
					if a.QRows, err = g.UploadIndices(l.outs); err != nil {
						b.Fatal(err)
					}
					defer a.QRows.Free()
					if a.Q, err = g.Upload(randTensor(rng, len(l.outs), d)); err != nil {
						b.Fatal(err)
					}
					defer a.Q.Free()
				}
				// Fifty calls per batch and one wait: the time of the
				// kernel, not of the round trip.
				const calls = 50
				b.ResetTimer()
				for range b.N {
					if err := g.BeginBatch(); err != nil {
						b.Fatal(err)
					}
					var last *Tensor
					for range calls {
						out, err := g.SegmentAttention(a)
						if err != nil {
							b.Fatal(err)
						}
						if last != nil {
							last.Free()
						}
						last = out
					}
					if err := g.Flush(); err != nil {
						b.Fatal(err)
					}
					syncGPU(b, last)
					last.Free()
				}
				us := float64(b.Elapsed().Microseconds()) / float64(b.N*calls)
				b.ReportMetric(us, "us/call")
				b.ReportMetric(us/float64(positions), "us/position")
			})
		}
	}
}

// BenchmarkGPUSegmentLayer times one pre-LN encoder layer of a small
// transformer (width 128, four heads, FFN 512) over b positions,
// recorded as one batch: LayerNorm, the fused q|k|v product, the gated
// attention, the output product and residual, LayerNorm, the FFN and its
// residual.
func BenchmarkGPUSegmentLayer(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	const heads, d, ffn = 4, 128, 512
	rng := rand.New(rand.NewPCG(2, 0))
	up := func(shape ...int) *Tensor {
		t, err := g.Upload(randTensor(rng, shape...))
		if err != nil {
			b.Fatal(err)
		}
		return t
	}
	ln1g, ln1b, ln2g, ln2b := up(d), up(d), up(d), up(d)
	wqkv, wo, w1, b1, w2, b2 := up(d, 3*d), up(d, d), up(d, ffn), up(ffn), up(ffn, d), up(d)
	for _, t := range []*Tensor{ln1g, ln1b, ln2g, ln2b, wqkv, wo, w1, b1, w2, b2} {
		defer t.Free()
	}
	gate := []float32{0.1, -0.2, 0.3, 0}
	for _, positions := range []int{64, 256, 1024} {
		b.Run(fmt.Sprintf("positions=%d", positions), func(b *testing.B) {
			l := benchLayout(rng, positions)
			seg, err := g.NewSegments(l.rows, l.off, l.flags)
			if err != nil {
				b.Fatal(err)
			}
			defer seg.Free()
			h := up(l.rows, d)
			defer h.Free()
			step := func() {
				must := func(t *Tensor, err error) *Tensor {
					if err != nil {
						b.Fatal(err)
					}
					return t
				}
				y := must(h.LayerNorm(ln1g, ln1b, 1e-5))
				qkv := must(y.Linear(wqkv, nil))
				y.Free()
				att := must(g.SegmentAttention(SegAttn{Seg: seg, Heads: heads, Dim: d, Q: qkv, K: qkv, V: qkv, KCol: d, VCol: 2 * d, Gate: gate}))
				qkv.Free()
				o := must(att.Linear(wo, nil))
				att.Free()
				if err := h.Add(o); err != nil {
					b.Fatal(err)
				}
				o.Free()
				y = must(h.LayerNorm(ln2g, ln2b, 1e-5))
				f := must(y.Linear(w1, b1, ActReLU))
				y.Free()
				o = must(f.Linear(w2, b2))
				f.Free()
				if err := h.Add(o); err != nil {
					b.Fatal(err)
				}
				o.Free()
			}
			b.ResetTimer()
			for range b.N {
				if err := g.BeginBatch(); err != nil {
					b.Fatal(err)
				}
				step()
				if err := g.Flush(); err != nil {
					b.Fatal(err)
				}
				syncGPU(b, h)
			}
			b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N)/float64(positions), "us/position")
			b.ReportMetric(float64(l.rows)/float64(positions), "rows/position")
		})
	}
}
