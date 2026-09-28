//go:build goexperiment.simd && arm64 && go1.27

package tensai

import (
	"runtime"
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// The dense float matmuls on arm64. A product with four or more rows runs
// in 4x16 register tiles: sixteen accumulators hold a block of the output
// across the whole contraction, so each loaded row of b feeds four rows
// and the output is written once, where the row-at-a-time kernel read it
// back and stored it for every element of a. Whatever the tiles leave --
// a column tail short of 16, rows short of 4 -- goes through that row
// kernel, 4-lane NEON vectors unrolled four deep with part loads for the
// tail. The multiply-add is fused, as the arm64 compiler fuses the
// portable bodies' `out += a*b`.

// dotRows computes rows lo..hi of out = a * b.
func dotRows(out, a, b *Matrix, lo, hi int) {
	cols := b.Cols
	if a.Cols == 0 {
		// An empty contraction sums nothing. The tiles' loop over it would
		// not run and leave whatever out held, so the rows are cleared
		// here, as the row kernel clears a row with no nonzero element.
		clear(out.Data[lo*cols : hi*cols])
		return
	}
	if hi-lo >= 4 && cols >= 16 {
		r4 := lo + (hi-lo)&^3
		n16 := cols &^ 15
		dotRowsTiled(out, a, b, lo, r4, n16)
		if n16 < cols {
			dotRowsAxpy(out, a, b, lo, r4, n16)
		}
		lo = r4
	}
	dotRowsAxpy(out, a, b, lo, hi, 0)
}

// dotRowsAxpy computes rows lo..hi, columns c0..b.Cols of out = a * b one
// row at a time: every nonzero element of a's row scales the matching row
// of b into the output row. Both rows are cut to the same length and
// capacity, so the compiler shares one bounds check between the load of b
// and the load and store of out at each step; cut by their ends, as they
// were, the two rows were checked apart, and a single row of 512 ran 1.5
// times as long.
func dotRowsAxpy(out, a, b *Matrix, lo, hi, c0 int) {
	cols := b.Cols
	width := cols - c0
	wide := width &^ 15 // widest multiple of 16
	vecs := width &^ 3  // widest multiple of 4
	for r := lo; r < hi; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		outRow := out.Data[r*cols+c0:][:width:width]
		initialized := false
		for k, av := range aRow {
			if av == 0 {
				continue
			}
			bRow := b.Data[k*cols+c0:][:width:width]
			vv := archsimd.BroadcastFloat32x4(av)
			var c int
			if !initialized {
				for ; c < wide; c += 16 {
					simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).Mul(vv), outRow[c:])
					simd.StoreF32x4(simd.LoadF32x4(bRow[c+4:]).Mul(vv), outRow[c+4:])
					simd.StoreF32x4(simd.LoadF32x4(bRow[c+8:]).Mul(vv), outRow[c+8:])
					simd.StoreF32x4(simd.LoadF32x4(bRow[c+12:]).Mul(vv), outRow[c+12:])
				}
				for ; c < vecs; c += 4 {
					simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).Mul(vv), outRow[c:])
				}
				if c < width {
					simd.StoreF32x4Part(simd.LoadF32x4Part(bRow[c:]).Mul(vv), outRow[c:])
				}
				initialized = true
				continue
			}
			for ; c < wide; c += 16 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+4:]).MulAdd(vv, simd.LoadF32x4(outRow[c+4:])), outRow[c+4:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+8:]).MulAdd(vv, simd.LoadF32x4(outRow[c+8:])), outRow[c+8:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+12:]).MulAdd(vv, simd.LoadF32x4(outRow[c+12:])), outRow[c+12:])
			}
			for ; c < vecs; c += 4 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
			}
			if c < width {
				simd.StoreF32x4Part(simd.LoadF32x4Part(bRow[c:]).MulAdd(vv, simd.LoadF32x4Part(outRow[c:])), outRow[c:])
			}
		}
		if !initialized {
			clear(outRow)
		}
	}
}

// dotTileK is how many rows of b one pass of the tiled kernels walks: a
// 16-wide strip of that many rows is 32KB, which stays in L1 while every
// block of four output rows reuses it -- Apple's cores hold 128KB of it and
// Neoverse's 64KB. Twice the AVX2 kernel's depth measured 2% faster on an
// M5; four times, no better again.
const dotTileK = 512

// dotRowsTiled computes rows lo..hi (a multiple of 4 apart), columns
// 0..n16 (a multiple of 16) of out = a * b in 4x16 register tiles. Each
// tile accumulates in k order with fused multiply-adds starting from zero,
// so a finite result comes out bit for bit as dotRowsAxpy leaves it: a
// zero element of a adds an exact zero where the row kernel skips it.
// Only the sign of a zero result can differ, and an Inf or NaN in b
// meeting a zero in a, which the row kernel never multiplies.
func dotRowsTiled(out, a, b *Matrix, lo, hi, n16 int) {
	for k0 := 0; k0 < a.Cols; k0 += dotTileK {
		k1 := min(k0+dotTileK, a.Cols)
		for c := 0; c < n16; c += 16 {
			for r := lo; r < hi; r += 4 {
				dotTile4x16(out, a, b, r, c, k0, k1)
			}
		}
	}
}

// dotTile4x16 adds a[r:r+4, k0:k1] * b[k0:k1, c:c+16] into the 4x16 tile of
// out at (r, c), or writes it there when k0 is 0. Sixteen accumulators,
// four loads of b and a broadcast per row use 21 of the 32 vector
// registers.
func dotTile4x16(out, a, b *Matrix, r, c, k0, k1 int) {
	n, kk := b.Cols, a.Cols
	o0 := out.Data[r*n+c : r*n+c+16]
	o1 := out.Data[(r+1)*n+c : (r+1)*n+c+16]
	o2 := out.Data[(r+2)*n+c : (r+2)*n+c+16]
	o3 := out.Data[(r+3)*n+c : (r+3)*n+c+16]
	var c00, c01, c02, c03, c10, c11, c12, c13 archsimd.Float32x4
	var c20, c21, c22, c23, c30, c31, c32, c33 archsimd.Float32x4
	if k0 > 0 {
		c00, c01, c02, c03 = simd.LoadF32x4(o0), simd.LoadF32x4(o0[4:]), simd.LoadF32x4(o0[8:]), simd.LoadF32x4(o0[12:])
		c10, c11, c12, c13 = simd.LoadF32x4(o1), simd.LoadF32x4(o1[4:]), simd.LoadF32x4(o1[8:]), simd.LoadF32x4(o1[12:])
		c20, c21, c22, c23 = simd.LoadF32x4(o2), simd.LoadF32x4(o2[4:]), simd.LoadF32x4(o2[8:]), simd.LoadF32x4(o2[12:])
		c30, c31, c32, c33 = simd.LoadF32x4(o3), simd.LoadF32x4(o3[4:]), simd.LoadF32x4(o3[8:]), simd.LoadF32x4(o3[12:])
	}
	a0 := a.Data[r*kk+k0 : r*kk+k1]
	a1 := a.Data[(r+1)*kk+k0 : (r+1)*kk+k1]
	a2 := a.Data[(r+2)*kk+k0 : (r+2)*kk+k1]
	a3 := a.Data[(r+3)*kk+k0 : (r+3)*kk+k1]
	a1, a2, a3 = a1[:len(a0)], a2[:len(a0)], a3[:len(a0)]
	bd, bi := b.Data, k0*n+c
	for k := range a0 {
		bRow := bd[bi : bi+16]
		bi += n
		b0, b1, b2, b3 := simd.LoadF32x4(bRow), simd.LoadF32x4(bRow[4:]), simd.LoadF32x4(bRow[8:]), simd.LoadF32x4(bRow[12:])
		v := archsimd.BroadcastFloat32x4(a0[k])
		c00, c01, c02, c03 = b0.MulAdd(v, c00), b1.MulAdd(v, c01), b2.MulAdd(v, c02), b3.MulAdd(v, c03)
		v = archsimd.BroadcastFloat32x4(a1[k])
		c10, c11, c12, c13 = b0.MulAdd(v, c10), b1.MulAdd(v, c11), b2.MulAdd(v, c12), b3.MulAdd(v, c13)
		v = archsimd.BroadcastFloat32x4(a2[k])
		c20, c21, c22, c23 = b0.MulAdd(v, c20), b1.MulAdd(v, c21), b2.MulAdd(v, c22), b3.MulAdd(v, c23)
		v = archsimd.BroadcastFloat32x4(a3[k])
		c30, c31, c32, c33 = b0.MulAdd(v, c30), b1.MulAdd(v, c31), b2.MulAdd(v, c32), b3.MulAdd(v, c33)
	}
	storeTile4x16(o0, o1, o2, o3, [16]archsimd.Float32x4{
		c00, c01, c02, c03, c10, c11, c12, c13,
		c20, c21, c22, c23, c30, c31, c32, c33,
	})
}

// storeTile4x16 writes a 4x16 tile of accumulators back to its four rows.
func storeTile4x16(o0, o1, o2, o3 []float32, t [16]archsimd.Float32x4) {
	simd.StoreF32x4(t[0], o0)
	simd.StoreF32x4(t[1], o0[4:])
	simd.StoreF32x4(t[2], o0[8:])
	simd.StoreF32x4(t[3], o0[12:])
	simd.StoreF32x4(t[4], o1)
	simd.StoreF32x4(t[5], o1[4:])
	simd.StoreF32x4(t[6], o1[8:])
	simd.StoreF32x4(t[7], o1[12:])
	simd.StoreF32x4(t[8], o2)
	simd.StoreF32x4(t[9], o2[4:])
	simd.StoreF32x4(t[10], o2[8:])
	simd.StoreF32x4(t[11], o2[12:])
	simd.StoreF32x4(t[12], o3)
	simd.StoreF32x4(t[13], o3[4:])
	simd.StoreF32x4(t[14], o3[8:])
	simd.StoreF32x4(t[15], o3[12:])
}

func dotWorkerCount(rows, inner, cols int) int {
	workers := 1
	if rows*inner*cols >= 1<<20 {
		workers = runtime.NumCPU()
		if workers > 8 {
			workers = 8
		}
		if workers > rows {
			workers = rows
		}
	}
	return workers
}

// dotTATall computes out = a^T * b when b has at most eight columns: the
// output rows lo..hi are accumulated four at a time in vector registers,
// two vectors per row, so the inputs are streamed instead of the output
// being read back and written for every element.
func dotTATall(out, a, b *Matrix, lo, hi int) {
	k, n, rows := a.Cols, b.Cols, a.Rows
	for j0 := 0; j0 < n; j0 += 8 {
		width := min(8, n-j0)
		dotTATallCols(out, a, b, lo, hi, k, n, rows, j0, width)
	}
}

// dotTATallCols is dotTATall over one eight-wide slice of b's columns,
// two vectors per output row; part loads and stores cover a slice that
// is not a whole eight, and the second vector is empty when it is four
// or fewer.
func dotTATallCols(out, a, b *Matrix, lo, hi, k, n, rows, j0, width int) {
	w0 := min(width, 4)
	for i0 := lo; i0 < hi; i0 += 4 {
		var a0, a1, a2, a3, c0, c1, c2, c3 archsimd.Float32x4
		switch hi - i0 {
		case 1:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
			}
		case 2:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
				v = archsimd.BroadcastFloat32x4(aRow[1])
				a1, c1 = b0.MulAdd(v, a1), b1.MulAdd(v, c1)
			}
		case 3:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
				v = archsimd.BroadcastFloat32x4(aRow[1])
				a1, c1 = b0.MulAdd(v, a1), b1.MulAdd(v, c1)
				v = archsimd.BroadcastFloat32x4(aRow[2])
				a2, c2 = b0.MulAdd(v, a2), b1.MulAdd(v, c2)
			}
		default:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
				v = archsimd.BroadcastFloat32x4(aRow[1])
				a1, c1 = b0.MulAdd(v, a1), b1.MulAdd(v, c1)
				v = archsimd.BroadcastFloat32x4(aRow[2])
				a2, c2 = b0.MulAdd(v, a2), b1.MulAdd(v, c2)
				v = archsimd.BroadcastFloat32x4(aRow[3])
				a3, c3 = b0.MulAdd(v, a3), b1.MulAdd(v, c3)
			}
		}
		lo4 := [4]archsimd.Float32x4{a0, a1, a2, a3}
		hi4 := [4]archsimd.Float32x4{c0, c1, c2, c3}
		for i := i0; i < hi && i < i0+4; i++ {
			row := out.Data[i*n+j0 : i*n+j0+width]
			simd.StoreF32x4Part(lo4[i-i0], row[:w0])
			simd.StoreF32x4Part(hi4[i-i0], row[w0:])
		}
	}
}

// dotTARows adds out rows lo..hi of out += a^T * b, which is the weight
// gradient of a product: a is (R, I), b is (R, J), out is (I, J). Four or
// more output rows run in 4x16 register tiles over the whole contraction,
// the way dotRows tiles the forward product; the rest goes through the row
// kernel, which scales b's row into each output row one element of a at a
// time. A contraction of one row -- a single sample's gradient -- gives
// each output element one multiply-add, which leaves the tiles nothing to
// hold in registers, so it goes to the row kernel: streaming the output in
// order, that measured about twice as fast there.
func dotTARows(out, a, b *Matrix, lo, hi int) {
	cols := b.Cols
	if hi-lo >= 4 && cols >= 16 && a.Rows >= 2 {
		i4 := lo + (hi-lo)&^3
		n16 := cols &^ 15
		dotTARowsTiled(out, a, b, lo, i4, n16)
		if n16 < cols {
			dotTARowsAxpy(out, a, b, lo, i4, n16)
		}
		lo = i4
	}
	dotTARowsAxpy(out, a, b, lo, hi, 0)
}

// dotTARowsAxpy adds out rows lo..hi, columns c0..b.Cols of a^T * b: a's
// element is broadcast against b's row and accumulated into out's row,
// both cut to one length and capacity as dotRowsAxpy cuts them.
func dotTARowsAxpy(out, a, b *Matrix, lo, hi, c0 int) {
	cols := b.Cols
	width := cols - c0
	wide := width &^ 15
	vecs := width &^ 3
	for r := 0; r < a.Rows; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		bRow := b.Data[r*cols+c0:][:width:width]
		for i := lo; i < hi; i++ {
			av := aRow[i]
			if av == 0 {
				continue
			}
			outRow := out.Data[i*cols+c0:][:width:width]
			vv := archsimd.BroadcastFloat32x4(av)
			var c int
			for ; c < wide; c += 16 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+4:]).MulAdd(vv, simd.LoadF32x4(outRow[c+4:])), outRow[c+4:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+8:]).MulAdd(vv, simd.LoadF32x4(outRow[c+8:])), outRow[c+8:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+12:]).MulAdd(vv, simd.LoadF32x4(outRow[c+12:])), outRow[c+12:])
			}
			for ; c < vecs; c += 4 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
			}
			if c < width {
				simd.StoreF32x4Part(simd.LoadF32x4Part(bRow[c:]).MulAdd(vv, simd.LoadF32x4Part(outRow[c:])), outRow[c:])
			}
		}
	}
}

// dotTARowsTiled adds out rows lo..hi (a multiple of 4 apart), columns
// 0..n16 (a multiple of 16) of a^T * b in 4x16 register tiles. A tile
// starts from what out holds and accumulates in the order of a's rows with
// fused multiply-adds, so it leaves a finite result bit for bit as
// dotTARowsAxpy does, with the same two exceptions dotRowsTiled has.
func dotTARowsTiled(out, a, b *Matrix, lo, hi, n16 int) {
	for r0 := 0; r0 < a.Rows; r0 += dotTileK {
		r1 := min(r0+dotTileK, a.Rows)
		for c := 0; c < n16; c += 16 {
			for i := lo; i < hi; i += 4 {
				dotTATile4x16(out, a, b, i, c, r0, r1)
			}
		}
	}
}

// dotTATile4x16 adds a[r0:r1, i:i+4]^T * b[r0:r1, c:c+16] into the 4x16
// tile of out at (i, c). The four elements of a's row it needs sit side by
// side, so a step reads one short run of a and one of b.
func dotTATile4x16(out, a, b *Matrix, i, c, r0, r1 int) {
	n, ka := b.Cols, a.Cols
	o0 := out.Data[i*n+c : i*n+c+16]
	o1 := out.Data[(i+1)*n+c : (i+1)*n+c+16]
	o2 := out.Data[(i+2)*n+c : (i+2)*n+c+16]
	o3 := out.Data[(i+3)*n+c : (i+3)*n+c+16]
	c00, c01, c02, c03 := simd.LoadF32x4(o0), simd.LoadF32x4(o0[4:]), simd.LoadF32x4(o0[8:]), simd.LoadF32x4(o0[12:])
	c10, c11, c12, c13 := simd.LoadF32x4(o1), simd.LoadF32x4(o1[4:]), simd.LoadF32x4(o1[8:]), simd.LoadF32x4(o1[12:])
	c20, c21, c22, c23 := simd.LoadF32x4(o2), simd.LoadF32x4(o2[4:]), simd.LoadF32x4(o2[8:]), simd.LoadF32x4(o2[12:])
	c30, c31, c32, c33 := simd.LoadF32x4(o3), simd.LoadF32x4(o3[4:]), simd.LoadF32x4(o3[8:]), simd.LoadF32x4(o3[12:])
	ad, ai := a.Data, r0*ka+i
	bd, bi := b.Data, r0*n+c
	for r := r0; r < r1; r++ {
		aq := ad[ai : ai+4]
		ai += ka
		bRow := bd[bi : bi+16]
		bi += n
		b0, b1, b2, b3 := simd.LoadF32x4(bRow), simd.LoadF32x4(bRow[4:]), simd.LoadF32x4(bRow[8:]), simd.LoadF32x4(bRow[12:])
		v := archsimd.BroadcastFloat32x4(aq[0])
		c00, c01, c02, c03 = b0.MulAdd(v, c00), b1.MulAdd(v, c01), b2.MulAdd(v, c02), b3.MulAdd(v, c03)
		v = archsimd.BroadcastFloat32x4(aq[1])
		c10, c11, c12, c13 = b0.MulAdd(v, c10), b1.MulAdd(v, c11), b2.MulAdd(v, c12), b3.MulAdd(v, c13)
		v = archsimd.BroadcastFloat32x4(aq[2])
		c20, c21, c22, c23 = b0.MulAdd(v, c20), b1.MulAdd(v, c21), b2.MulAdd(v, c22), b3.MulAdd(v, c23)
		v = archsimd.BroadcastFloat32x4(aq[3])
		c30, c31, c32, c33 = b0.MulAdd(v, c30), b1.MulAdd(v, c31), b2.MulAdd(v, c32), b3.MulAdd(v, c33)
	}
	storeTile4x16(o0, o1, o2, o3, [16]archsimd.Float32x4{
		c00, c01, c02, c03, c10, c11, c12, c13,
		c20, c21, c22, c23, c30, c31, c32, c33,
	})
}
