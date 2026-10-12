// Package accelerate runs dense matrix products through Apple's Accelerate
// framework, whose BLAS uses the CPU's matrix unit (AMX on M1-M3, SME on M4
// and later). It loads the framework at run time through purego, so a build
// needs no cgo and CGO_ENABLED=0 still works. Where Accelerate is missing
// (any other OS, or a failed load) every function falls back to tensai's own
// serial kernels, so a caller can use it unconditionally.
//
// The products are single-threaded: the package asks Accelerate for its
// single-threaded mode when it loads (BLASSetThreading, macOS 15 and later),
// because the callers it is for already run one product per goroutine across
// many goroutines. The results are not bit for bit tensai's: Accelerate sums
// in its own order.
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
	if !available() || a.Rows == 0 || b.Cols == 0 {
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
	if !available() || a.Rows == 0 || b.Rows == 0 {
		return tensai.DotTBIntoSerial(out, a, b)
	}
	if a.Cols == 0 {
		clear(out.Data)
		return nil
	}
	gemm(true, a.Rows, b.Rows, a.Cols, a.Data, b.Data, out.Data)
	return nil
}
