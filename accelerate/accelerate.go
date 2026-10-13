// Package accelerate runs dense matrix products through Apple's Accelerate
// framework, whose BLAS uses the CPU's matrix unit (AMX on M1-M3, SME on M4
// and later). It loads the framework at run time through purego, so a build
// needs no cgo and CGO_ENABLED=0 still works. Where Accelerate is missing
// (any other OS, a failed load, or a failed self-test) every function falls
// back to tensai's own serial kernels, so a caller can use it unconditionally.
//
// The results are tensai's bit for bit: when it loads, the package checks
// Accelerate against tensai on the products of a small transformer and falls
// back to tensai if any bit differs. Products of one row, or of fewer than 16
// columns, always run on tensai, because Accelerate takes other paths there.
//
// Each call blocks the calling thread's asynchronous signals for its duration
// (a signal handled during Accelerate's matrix-unit kernel corrupts the
// result) and runs single-threaded, since the callers it is for already run
// one product per goroutine across many goroutines.
package accelerate

import "github.com/mattn/tensai"

// Available reports whether products run through Accelerate (false means the
// tensai fallback).
func Available() bool { return available() }

// Err returns why Accelerate is not available (nil when it is).
func Err() error { return loadErr() }

// DotInto computes out = a * b, overwriting out.
func DotInto(out, a, b *tensai.Matrix) error {
	if a.Cols != b.Rows || out.Rows != a.Rows || out.Cols != b.Cols {
		return tensai.DotIntoSerial(out, a, b) // reports the shape mismatch the same way
	}
	if !available() || a.Rows < 2 || b.Cols < 16 {
		return tensai.DotIntoSerial(out, a, b)
	}
	if a.Cols == 0 {
		clear(out.Data)
		return nil
	}
	gemm(false, a.Rows, b.Cols, a.Cols, a.Data, b.Data, out.Data)
	return nil
}

// DotTBInto computes out = a * b^T, overwriting out.
func DotTBInto(out, a, b *tensai.Matrix) error {
	if a.Cols != b.Cols || out.Rows != a.Rows || out.Cols != b.Rows {
		return tensai.DotTBIntoSerial(out, a, b)
	}
	if !availableTB() || a.Rows < 2 || b.Rows < 16 {
		return tensai.DotTBIntoSerial(out, a, b)
	}
	if a.Cols == 0 {
		clear(out.Data)
		return nil
	}
	gemm(true, a.Rows, b.Rows, a.Cols, a.Data, b.Data, out.Data)
	return nil
}
