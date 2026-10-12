//go:build !darwin

package accelerate

import "errors"

var errUnsupported = errors.New("accelerate: Accelerate is only on Apple platforms")

func available() bool { return false }

func loadErr() error { return errUnsupported }

func gemm(bool, int, int, int, []float32, []float32, []float32) {}
