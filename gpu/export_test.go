//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

// BuffersMade reports how many buffers the device has made because its
// pool had none of the size asked for. A training step that reuses what
// the previous one freed makes none.
func BuffersMade(g *Device) uint64 {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	return g.pool.made
}
