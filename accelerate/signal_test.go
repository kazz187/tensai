package accelerate

import (
	"io"
	"math"
	"math/rand/v2"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mattn/tensai"
)

// TestDotIntoUnderSignals runs products from many goroutines while a CPU profile (SIGPROF) and constant garbage
// collection (SIGURG preemption) signal the threads, and checks every result against tensai bit for bit. Without the
// signal mask around each Accelerate call, a few results per ten thousand come out corrupted.
func TestDotIntoUnderSignals(t *testing.T) {
	if !Available() {
		t.Skipf("Accelerate unavailable: %v", Err())
	}
	if err := pprof.StartCPUProfile(io.Discard); err == nil {
		defer pprof.StopCPUProfile()
	}
	stop := make(chan struct{})
	var gcWG sync.WaitGroup
	gcWG.Add(1)
	go func() { // garbage and collections: preemption signals all the time
		defer gcWG.Done()
		var keep [][]byte
		for {
			select {
			case <-stop:
				return
			default:
			}
			keep = append(keep, make([]byte, 1<<16))
			if len(keep) > 64 {
				keep = keep[:0]
				runtime.GC()
			}
		}
	}()
	const k, n = 512, 128
	rng := rand.New(rand.NewPCG(21, 22)) //nolint:gosec // test data
	w := randMatrix(rng, k, n)
	var bad, calls atomic.Int64
	deadline := time.Now().Add(3 * time.Second)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		a := randMatrix(rng, 58, k)
		want := tensai.NewMatrix(58, n)
		if err := tensai.DotIntoSerial(want, a, w); err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := tensai.NewMatrix(58, n)
			for time.Now().Before(deadline) {
				if err := DotInto(got, a, w); err != nil {
					t.Error(err)
					return
				}
				calls.Add(1)
				for i := range got.Data {
					if math.Float32bits(float32(got.Data[i])) != math.Float32bits(float32(want.Data[i])) {
						bad.Add(1)
						break
					}
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	gcWG.Wait()
	if bad.Load() != 0 {
		t.Fatalf("%d of %d products differ from tensai under signals", bad.Load(), calls.Load())
	}
	t.Logf("%d products, all bit-identical to tensai", calls.Load())
}
