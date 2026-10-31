//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/mattn/tensai"
)

// chainJob is one goroutine's independent product chain: y = relu(x*w1
// + b) * w2^T, all on its own tensors.
type chainJob struct {
	x, w1, b, w2 *Tensor
	n            int // elements of y
}

func newChainJob(t *testing.T, g *Device, seed uint64) chainJob {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 11))
	m, k, h, n := 48+int(seed%5)*16, 96, 128, 80
	up := func(shape ...int) *Tensor {
		x, err := g.Upload(randTensor(rng, shape...))
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	return chainJob{x: up(m, k), w1: up(k, h), b: up(h), w2: up(n, h), n: m * n}
}

func (j chainJob) free() {
	for _, x := range []*Tensor{j.x, j.w1, j.b, j.w2} {
		x.Free()
	}
}

// run computes the chain and returns y on the device.
func (j chainJob) run() (*Tensor, error) {
	h, err := j.x.Linear(j.w1, j.b, ActReLU)
	if err != nil {
		return nil, err
	}
	defer h.Free()
	return h.MatMulT(j.w2)
}

func sameBits(a, b []tensai.Float) error {
	if len(a) != len(b) {
		return fmt.Errorf("length %d, want %d", len(a), len(b))
	}
	for i := range a {
		if math.Float32bits(float32(a[i])) != math.Float32bits(float32(b[i])) {
			return fmt.Errorf("element %d is %v, want %v", i, a[i], b[i])
		}
	}
	return nil
}

// Goroutines running independent chains and downloads at the same time
// get exactly the bytes each chain gives when run alone: one-shot
// operations with Download, and batches whose downloads are begun inside
// the batch and waited outside it while other goroutines' batches run.
func TestGPUConcurrentDownloads(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	const workers, iters = 8, 12
	jobs := make([]chainJob, workers)
	want := make([][]tensai.Float, workers)
	for i := range jobs {
		jobs[i] = newChainJob(t, g, uint64(i))
		defer jobs[i].free()
		y, err := jobs[i].run()
		if err != nil {
			t.Fatal(err)
		}
		want[i] = make([]tensai.Float, jobs[i].n)
		err = y.DownloadInto(want[i])
		y.Free()
		if err != nil {
			t.Fatal(err)
		}
	}

	var batchMu sync.Mutex // one batch at a time per device, as BeginBatch requires
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for i := range jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			job := jobs[i]
			got := make([]tensai.Float, job.n)
			for it := range iters {
				var err error
				if i%2 == 0 {
					err = func() error {
						y, err := job.run()
						if err != nil {
							return err
						}
						defer y.Free()
						return y.DownloadInto(got)
					}()
				} else {
					err = func() error {
						batchMu.Lock()
						if err := g.BeginBatch(); err != nil {
							batchMu.Unlock()
							return err
						}
						y, err := job.run()
						if err != nil {
							g.Flush()
							batchMu.Unlock()
							return err
						}
						r, err := y.BeginDownloadRangeInto(0, got)
						y.Free() // the copy is already submitted
						batchMu.Unlock()
						if err != nil {
							return err
						}
						return r.Wait()
					}()
				}
				if err == nil {
					err = sameBits(got, want[i])
				}
				if err != nil {
					errs <- fmt.Errorf("worker %d iteration %d: %w", i, it, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// A begun download holds no device lock: other operations run before it
// is waited, and its result is still the tensor's value at the time it
// began, even though the tensor is overwritten in between.
func TestGPUBeginDownload(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()
	rng := rand.New(rand.NewPCG(3, 4))
	a, b := randTensor(rng, 37, 21), randTensor(rng, 37, 21)
	ga, err := g.Upload(a)
	if err != nil {
		t.Fatal(err)
	}
	defer ga.Free()
	gb, err := g.Upload(b)
	if err != nil {
		t.Fatal(err)
	}
	defer gb.Free()

	got := make([]tensai.Float, 5*21)
	r, err := ga.BeginDownloadRangeInto(3*21, got)
	if err != nil {
		t.Fatal(err)
	}
	if !g.mu.TryLock() {
		t.Fatal("the device lock is held while a download is in flight")
	}
	g.mu.Unlock()
	if !wgpuMu.TryLock() {
		t.Fatal("wgpuMu is held while a download is in flight")
	}
	wgpuMu.Unlock()
	if err := ga.Add(gb); err != nil { // overwrites a after the copy was queued
		t.Fatal(err)
	}
	sum := make([]tensai.Float, 37*21)
	if err := ga.DownloadInto(sum); err != nil {
		t.Fatal(err)
	}
	if err := r.Wait(); err != nil {
		t.Fatal(err)
	}
	if err := r.Wait(); err != nil { // a second Wait returns the first result
		t.Fatal(err)
	}
	if err := sameBits(got, a.Data[3*21:8*21]); err != nil {
		t.Errorf("begun download: %v", err)
	}
	for i := range sum {
		if want := a.Data[i] + b.Data[i]; sum[i] != want {
			t.Fatalf("sum element %d is %v, want %v", i, sum[i], want)
		}
	}
	if _, err := ga.BeginDownloadRangeInto(37*21-2, make([]tensai.Float, 3)); err == nil {
		t.Error("a download past the end began")
	}
}

// Close lands the downloads still in flight, and one that was never
// waited reports the closed device instead of touching it.
func TestGPUCloseWithPendingDownload(t *testing.T) {
	g := openTestGPU(t)
	x, err := g.Upload(randTensor(rand.New(rand.NewPCG(1, 1)), 64, 64))
	if err != nil {
		t.Fatal(err)
	}
	r, err := x.BeginDownloadRangeInto(0, make([]tensai.Float, 64*64))
	if err != nil {
		t.Fatal(err)
	}
	g.Close()
	if err := r.Wait(); err == nil {
		t.Error("a download waited after Close succeeded")
	}
	g.Close() // a second Close is a no-op
}
