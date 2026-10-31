//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

// Downloads that wait without the device lock. A download records a copy
// into a MapRead staging buffer, submits it, and asks wgpu to map the
// buffer; all of that is quick and runs under Device.mu and wgpuMu like
// any other operation. The wait for the mapping is the only part that
// takes real time, and it holds neither lock, so while one goroutine
// waits for its results others keep encoding and submitting work on the
// same device -- several models sharing one GPU no longer queue behind
// each other's readbacks.
//
// Waiting means polling the device, and a blocking wgpuDevicePoll is the
// wrong tool once goroutines share it: wgpu-core holds the device's fence
// lock (shared) for the whole blocking wait, and every queue submit takes
// that lock exclusively, so one goroutine's wait would stall everyone
// else's submit -- with Device.mu and wgpuMu held -- until the GPU had
// finished the waited-for work. Each device instead runs one poller
// goroutine while any mapping is in flight, which polls without blocking
// and sleeps pollPause in between. (Metal's blocking wait is no finer
// anyway: wgpu-hal sleeps a millisecond at a time until the command
// buffer completes. On Linux a process with nothing else to run may see
// the Go runtime round the sleep up to a millisecond.)
//
// A map callback fires inside whichever wgpu call next processes the
// device's completed work -- the poller's poll, or another goroutine's
// submit or poll -- so each mapping's result travels through a registry
// keyed by the callback's userdata, never through package globals, and the
// callback takes only the registry's own locks (its caller may hold
// Device.mu and wgpuMu).

import (
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/mattn/tensai"
)

// pollPause is how long the poller sleeps between polls.
const pollPause = 20 * time.Microsecond

// maxReadbacks caps the idle staging buffers a device keeps: one per
// download in flight at once, which a single goroutine keeps at one.
const maxReadbacks = 64

// mapWait is one buffer mapping in flight.
type mapWait struct {
	g    *Device
	done chan struct{} // closed by the map callback
	ok   bool
	msg  string
}

// maps routes map callbacks, by their userdata, to their mapWait.
var maps struct {
	sync.Mutex
	next uintptr
	m    map[uintptr]*mapWait
}

// mapDone is the body of the map callback the binding files register:
// it hands the result to the mapping registered as id.
func mapDone(id uintptr, ok bool, msg string) {
	maps.Lock()
	w := maps.m[id]
	delete(maps.m, id)
	maps.Unlock()
	if w == nil {
		return
	}
	w.ok, w.msg = ok, msg
	ws := &w.g.waits
	ws.mu.Lock()
	for i, p := range ws.pending {
		if p == w {
			ws.pending = append(ws.pending[:i], ws.pending[i+1:]...)
			break
		}
	}
	ws.mu.Unlock()
	close(w.done)
}

// gpuWaits is a device's mappings in flight and the goroutine polling for
// them.
type gpuWaits struct {
	mu      sync.Mutex
	pending []*mapWait
	polling bool
	poller  sync.WaitGroup
}

// watchMap registers a mapping that is about to be requested and starts
// the poller if none runs; the mapping must be requested with the
// returned id as its callback userdata. The caller holds Device.mu and
// wgpuMu.
func (g *Device) watchMap() (*mapWait, uintptr) {
	w := &mapWait{g: g, done: make(chan struct{})}
	maps.Lock()
	if maps.m == nil {
		maps.m = make(map[uintptr]*mapWait)
	}
	maps.next++
	id := maps.next
	maps.m[id] = w
	maps.Unlock()
	ws := &g.waits
	ws.mu.Lock()
	ws.pending = append(ws.pending, w)
	if !ws.polling {
		ws.polling = true
		ws.poller.Add(1)
		go g.pollMaps()
	}
	ws.mu.Unlock()
	return w, id
}

// pollMaps polls g until no mapping is in flight. It holds no lock while
// it polls.
func (g *Device) pollMaps() {
	ws := &g.waits
	defer ws.poller.Done()
	for ws.inFlight() {
		g.pollDevice()
		if !ws.inFlight() {
			return
		}
		time.Sleep(pollPause)
	}
}

// inFlight reports whether any mapping is in flight; when none is, it
// marks the poller stopped, so the next watchMap starts another.
func (ws *gpuWaits) inFlight() bool {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	if len(ws.pending) > 0 {
		return true
	}
	ws.polling = false
	return false
}

// beginClose marks g closed, so no new operation starts, and waits until
// every mapping in flight has landed and the poller has stopped; Close then
// releases the device. It reports false when g was already closed.
func (g *Device) beginClose() bool {
	g.mu.Lock()
	wgpuMu.Lock()
	was := g.closed
	g.closed = true
	wgpuMu.Unlock()
	g.mu.Unlock()
	if was {
		return false
	}
	g.waits.poller.Wait()
	return true
}

// PendingDownload is a download whose copy is already on the GPU queue;
// Wait finishes it.
type PendingDownload struct {
	g       *Device
	w       *mapWait
	staging uintptr
	size    uint64 // the staging buffer's capacity
	dst     []tensai.Float
	waited  bool
	err     error
}

// BeginDownloadRangeInto starts copying elements [off, off+len(dst)) of t
// into dst and returns without waiting for the GPU: it flushes an open
// batch, submits the copy, and asks for the staging buffer to be mapped.
// Wait finishes the download and fills dst, which must not be touched
// until then. In between the device is free: other goroutines encode and
// submit work on it while the download is in flight. t may be freed, or
// written by later work, as soon as this returns -- the copy is queued
// ahead of anything submitted after it. DownloadRangeInto is this followed
// by Wait. Every PendingDownload must be waited, from one goroutine.
func (t *Tensor) BeginDownloadRangeInto(off int, dst []tensai.Float) (*PendingDownload, error) {
	if t.freed {
		return nil, errors.New("tensai: gpu tensor already freed")
	}
	if t.f16 {
		return nil, errors.New("tensai: cannot download a half-precision tensor into floats")
	}
	if off < 0 || len(dst) == 0 || off+len(dst) > t.Size() {
		return nil, fmt.Errorf("tensai: download of %d elements at %d overflows %d", len(dst), off, t.Size())
	}
	return t.startRead(off, dst)
}

// startRead is BeginDownloadRangeInto past the argument checks.
func (t *Tensor) startRead(off int, dst []tensai.Float) (*PendingDownload, error) {
	g := t.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	if err := g.flushLocked(); err != nil {
		return nil, err
	}
	uncapturedCB = ""
	bytes := uint64(len(dst)) * 4
	staging, size := g.takeReadback(bytes)
	if staging == 0 {
		return nil, errors.New("tensai: gpu readback buffer allocation failed")
	}
	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	fnEncoderCopyBuffer(encoder, t.buf, t.off+uint64(off)*4, staging, 0, bytes)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)
	if uncapturedCB != "" {
		fnBufferRelease(staging)
		return nil, fmt.Errorf("tensai: gpu readback failed: %s", uncapturedCB)
	}
	w, id := g.watchMap()
	g.mapForRead(staging, bytes, id)
	if uncapturedCB != "" {
		// The request failed validation. Its callback may already have
		// reported, in which case this is a no-op; either way Wait sees
		// the failure instead of waiting for a mapping that never comes.
		mapDone(id, false, uncapturedCB)
	}
	r := &PendingDownload{g: g, w: w, staging: staging, size: size, dst: dst}
	if g.downloads == nil {
		g.downloads = make(map[*PendingDownload]struct{})
	}
	g.downloads[r] = struct{}{}
	return r, nil
}

// Wait blocks until the download's copy has landed, without holding the
// device lock while it does, then fills dst. Later calls return the first
// call's result.
func (r *PendingDownload) Wait() error {
	if r.waited {
		return r.err
	}
	r.waited = true
	<-r.w.done
	g := r.g
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if _, ok := g.downloads[r]; !ok { // Close released the staging buffer
		r.err = errors.New("tensai: gpu is closed")
		return r.err
	}
	delete(g.downloads, r)
	if !r.w.ok {
		fnBufferRelease(r.staging)
		r.err = fmt.Errorf("tensai: gpu readback failed: %s", r.w.msg)
		return r.err
	}
	src := fnBufferGetConstMapped(r.staging, 0, uintptr(len(r.dst))*4)
	if src == nil {
		fnBufferRelease(r.staging)
		r.err = errors.New("tensai: gpu readback: mapped range unavailable")
		return r.err
	}
	copy(r.dst, unsafe.Slice((*tensai.Float)(src), len(r.dst)))
	fnBufferUnmap(r.staging)
	g.putReadback(r.staging, r.size)
	return nil
}

// readInto copies len(dst) elements of t starting at element off into
// dst, flushing an open batch first: the work behind Download,
// DownloadRange and their Into forms, whose callers check the range.
func (t *Tensor) readInto(off int, dst []tensai.Float) error {
	r, err := t.startRead(off, dst)
	if err != nil {
		return err
	}
	return r.Wait()
}

// gpuReadbackBuffer is an idle MapRead staging buffer. Downloads in flight
// each own one; the idle ones wait in Device.readbacks for the next.
// Staging buffers are never bound (WebGPU forbids MapRead with storage or
// uniform usage), so no cached bind group can name one, and they are
// released without invalidating the bind-group cache.
type gpuReadbackBuffer struct {
	buf  uintptr
	size uint64
}

// takeReadback returns an unmapped staging buffer at least bytes large:
// the smallest idle one that fits, or a new one of exactly bytes. A new
// one replaces the largest idle buffer that was too small, so a single
// goroutine whose downloads grow keeps one buffer that grows with them.
// The caller holds Device.mu and wgpuMu.
func (g *Device) takeReadback(bytes uint64) (uintptr, uint64) {
	fit, small := -1, -1
	for i, r := range g.readbacks {
		switch {
		case r.size >= bytes:
			if fit < 0 || r.size < g.readbacks[fit].size {
				fit = i
			}
		case small < 0 || r.size > g.readbacks[small].size:
			small = i
		}
	}
	take := fit
	if fit < 0 {
		take = small
	}
	var r gpuReadbackBuffer
	if take >= 0 {
		r = g.readbacks[take]
		last := len(g.readbacks) - 1
		g.readbacks[take] = g.readbacks[last]
		g.readbacks = g.readbacks[:last]
	}
	if fit >= 0 {
		return r.buf, r.size
	}
	if r.buf != 0 {
		fnBufferRelease(r.buf)
	}
	return g.newBuffer(wgpuBufferUsageMapRead|wgpuBufferUsageCopyDst, bytes), bytes
}

// putReadback keeps a successfully unmapped staging buffer for a later
// download. The caller holds Device.mu and wgpuMu.
func (g *Device) putReadback(buf uintptr, size uint64) {
	if g.closed || len(g.readbacks) >= maxReadbacks {
		fnBufferRelease(buf)
		return
	}
	g.readbacks = append(g.readbacks, gpuReadbackBuffer{buf: buf, size: size})
}

// releaseReadback drops the idle staging buffers and those of downloads
// never waited, during Device.Close. The caller holds wgpuMu.
func (g *Device) releaseReadback() {
	for _, r := range g.readbacks {
		fnBufferRelease(r.buf)
	}
	g.readbacks = nil
	for r := range g.downloads {
		fnBufferRelease(r.staging)
	}
	g.downloads = nil
}
