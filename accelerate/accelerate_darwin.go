//go:build darwin

package accelerate

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/mattn/tensai"
)

const (
	frameworkPath = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
	libSystemPath = "/usr/lib/libSystem.B.dylib"
	// sgemmSymbol is the Fortran-style ILP64 GEMM (macOS 13.3 and later). Every argument is a pointer, so
	// purego.SyscallN can call it directly (it cannot pass cblas_sgemm's float alpha and beta by value): about 80 ns
	// a call instead of about 1 µs through purego.RegisterFunc, and no allocation.
	sgemmSymbol        = "sgemm$NEWLAPACK$ILP64"
	setThreadingSymbol = "BLASSetThreading" // macOS 15 and later; the setting is per thread
	blasSingleThreaded = 1
	sigBlock           = 1 // pthread_sigmask how
	sigSetmask         = 3
)

// asyncSignals is the set blocked around every Accelerate call: every signal that can arrive asynchronously (Go's
// SIGURG preemption, SIGPROF from a CPU profile, and the rest), but not the faults a thread raises itself.
// A signal handled while Accelerate's matrix-unit (SME) kernel runs corrupts the product: a plain C program that
// signals threads running sgemm gets wrong results in about 2.6% of the calls, even with an integer-only handler.
// A blocked signal stays pending and is delivered when the mask is restored after the call.
var asyncSignals = func() uint32 {
	set := ^uint32(0)
	for _, s := range []uint{4, 5, 6, 7, 8, 9, 10, 11, 12, 17} { // ILL TRAP ABRT EMT FPE KILL BUS SEGV SYS STOP
		set &^= 1 << (s - 1)
	}
	return set
}()

var (
	once                            sync.Once
	errLoad                         error
	errTB                           error // a*b^T differs from tensai's (DotTBInto then runs on tensai)
	fnSgemm, fnSetThreading, fnMask uintptr
	argPool                         = sync.Pool{New: func() any { return new(gemmArgs) }}
)

// gemmArgs holds the pointer arguments of one call. It is pooled, so a call allocates nothing.
type gemmArgs struct {
	transA, transB byte
	m, n, k        int64
	lda, ldb, ldc  int64
	alpha, beta    float32
	mask, old      uint32
}

func load() {
	lib, err := purego.Dlopen(frameworkPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		errLoad = fmt.Errorf("accelerate: open %s: %w", frameworkPath, err)
		return
	}
	if fnSgemm, err = purego.Dlsym(lib, sgemmSymbol); err != nil {
		errLoad = fmt.Errorf("accelerate: %s (needs macOS 13.3 or later): %w", sgemmSymbol, err)
		return
	}
	if s, err := purego.Dlsym(lib, setThreadingSymbol); err == nil {
		fnSetThreading = s
	}
	sys, err := purego.Dlopen(libSystemPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		errLoad = fmt.Errorf("accelerate: open %s: %w", libSystemPath, err)
		return
	}
	if fnMask, err = purego.Dlsym(sys, "pthread_sigmask"); err != nil {
		errLoad = fmt.Errorf("accelerate: pthread_sigmask: %w", err)
		return
	}
	errLoad, errTB = selfTest()
}

func available() bool {
	once.Do(load)
	return errLoad == nil
}

func loadErr() error {
	once.Do(load)
	return errLoad
}

func availableTB() bool {
	once.Do(load)
	return errLoad == nil && errTB == nil
}

// gemm runs C = A * op(B) in row-major order (A is m x k, C is m x n, B is k x n, or n x k when transB) as the
// column-major product C^T = op(B)^T A^T. The calling goroutine stays on its OS thread with asyncSignals blocked for
// the whole call, and asks for single-threaded BLAS on that thread first (the callers run one product per goroutine
// across many goroutines).
func gemm(transB bool, m, n, k int, a, b, c []float32) {
	p := argPool.Get().(*gemmArgs)
	p.transA, p.lda = 'N', int64(n)
	if transB {
		p.transA, p.lda = 'T', int64(k)
	}
	p.transB, p.ldb, p.ldc = 'N', int64(k), int64(n)
	p.m, p.n, p.k = int64(n), int64(m), int64(k)
	p.alpha, p.beta = 1, 0
	p.mask = asyncSignals
	runtime.LockOSThread()
	purego.SyscallN(fnMask, sigBlock, uintptr(unsafe.Pointer(&p.mask)), uintptr(unsafe.Pointer(&p.old)))
	if fnSetThreading != 0 {
		purego.SyscallN(fnSetThreading, blasSingleThreaded)
	}
	purego.SyscallN(fnSgemm,
		uintptr(unsafe.Pointer(&p.transA)), uintptr(unsafe.Pointer(&p.transB)),
		uintptr(unsafe.Pointer(&p.m)), uintptr(unsafe.Pointer(&p.n)), uintptr(unsafe.Pointer(&p.k)),
		uintptr(unsafe.Pointer(&p.alpha)), uintptr(unsafe.Pointer(&b[0])), uintptr(unsafe.Pointer(&p.lda)),
		uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&p.ldb)),
		uintptr(unsafe.Pointer(&p.beta)), uintptr(unsafe.Pointer(&c[0])), uintptr(unsafe.Pointer(&p.ldc)),
		1, 1) // the hidden lengths of the two one-character Fortran arguments
	purego.SyscallN(fnMask, sigSetmask, uintptr(unsafe.Pointer(&p.old)), 0)
	runtime.UnlockOSThread()
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	runtime.KeepAlive(c)
	argPool.Put(p)
}

// selfTest checks that Accelerate gives tensai's bits on the products callers send it: a*b at the shapes of a small
// transformer (K x N of 387x128, 128x384, 128x128, 128x512, 512x128 and 32x128 over a range of row counts, half of
// the rows sparse like one-hot features), and separately a*b^T. For a*b both sum each element over k in order with
// one fused multiply-add per step, so they agree on this machine; if a future OS or chip sums otherwise, Available
// reports false and every product falls back to tensai. tensai's SIMD a*b^T sums in vector lanes and adds them at the
// end, so it differs from Accelerate in GOEXPERIMENT=simd builds; then only DotTBInto falls back.
func selfTest() (dot, tb error) {
	rng := rand.New(rand.NewPCG(11, 13)) //nolint:gosec // test data
	mat := func(rows, cols int, sparse bool) *tensai.Matrix {
		m := tensai.NewMatrix(rows, cols)
		for i := range m.Data {
			if !sparse || rng.Float64() < 0.05 {
				m.Data[i] = tensai.Float(rng.NormFloat64())
			}
		}
		return m
	}
	same := func(x, y []tensai.Float) bool {
		for i := range x {
			if math.Float32bits(float32(x[i])) != math.Float32bits(float32(y[i])) {
				return false
			}
		}
		return true
	}
	for _, kn := range [][2]int{{387, 128}, {128, 384}, {128, 128}, {128, 512}, {512, 128}, {32, 128}} {
		k, n := kn[0], kn[1]
		w := mat(k, n, false)
		for _, m := range []int{2, 3, 5, 8, 15, 17, 58, 146, 464} {
			a := mat(m, k, m%2 == 1)
			got, want := tensai.NewMatrix(m, n), tensai.NewMatrix(m, n)
			gemm(false, m, n, k, a.Data, w.Data, got.Data)
			if err := tensai.DotIntoSerial(want, a, w); err != nil {
				return err, err
			}
			if !same(got.Data, want.Data) {
				err := fmt.Errorf("accelerate: results differ from tensai at %dx%dx%d; using tensai", m, k, n)
				return err, err
			}
		}
	}
	a, bt := mat(58, 128, false), mat(384, 128, false)
	got, want := tensai.NewMatrix(58, 384), tensai.NewMatrix(58, 384)
	gemm(true, 58, 384, 128, a.Data, bt.Data, got.Data)
	if err := tensai.DotTBIntoSerial(want, a, bt); err != nil {
		return nil, err
	}
	if !same(got.Data, want.Data) {
		return nil, fmt.Errorf("accelerate: a*b^T results differ from tensai's; DotTBInto uses tensai")
	}
	return nil, nil
}
