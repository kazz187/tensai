//go:build darwin

package accelerate

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

const frameworkPath = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

// CBLAS constants (cblas_new.h) and BLAS_THREADING_SINGLE_THREADED (thread_api.h).
const (
	cblasRowMajor        = 101
	cblasNoTrans         = 111
	cblasTrans           = 112
	blasSingleThreaded   = 1
	sgemmSymbol          = "cblas_sgemm$NEWLAPACK$ILP64"
	setThreadingSymbol   = "BLASSetThreading"
	minAccelerateVersion = "macOS 13.3"
)

var (
	once    sync.Once
	errLoad error
	// sgemm is the ILP64 interface: every integer argument is 64 bits, so the
	// ones passed on the stack (the arm64 ABI on Apple packs stack arguments
	// by their own size) are all 8-byte slots.
	sgemm func(order, transA, transB int32, m, n, k int64, alpha float32, a *float32, lda int64,
		b *float32, ldb int64, beta float32, c *float32, ldc int64)
)

func load() {
	lib, err := purego.Dlopen(frameworkPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		errLoad = fmt.Errorf("accelerate: open %s: %w", frameworkPath, err)
		return
	}
	sym, err := purego.Dlsym(lib, sgemmSymbol)
	if err != nil {
		errLoad = fmt.Errorf("accelerate: %s (needs %s or later): %w", sgemmSymbol, minAccelerateVersion, err)
		return
	}
	purego.RegisterFunc(&sgemm, sym)
	if s, err := purego.Dlsym(lib, setThreadingSymbol); err == nil {
		var setThreading func(uint32) int32
		purego.RegisterFunc(&setThreading, s)
		setThreading(blasSingleThreaded)
	}
}

func available() bool {
	once.Do(load)
	return errLoad == nil
}

func loadErr() error {
	once.Do(load)
	return errLoad
}

// gemm runs C = A * op(B) in row-major order: A is m x k, C is m x n, and B is
// k x n, or n x k when transB.
func gemm(transB bool, m, n, k int, a, b, c []float32) {
	tb, ldb := int32(cblasNoTrans), int64(n)
	if transB {
		tb, ldb = cblasTrans, int64(k)
	}
	sgemm(cblasRowMajor, cblasNoTrans, tb, int64(m), int64(n), int64(k), 1, &a[0], int64(k), &b[0], ldb, 0, &c[0], int64(n))
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	runtime.KeepAlive(c)
}
