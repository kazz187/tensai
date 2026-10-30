package gpu

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// AdapterType says what kind of device an adapter is.
type AdapterType uint8

const (
	AdapterUnknown    AdapterType = iota
	AdapterDiscrete               // a discrete GPU
	AdapterIntegrated             // a GPU sharing memory with the CPU
	AdapterCPU                    // a software rasterizer: llvmpipe, lavapipe, WARP, SwiftShader
)

func (t AdapterType) String() string {
	switch t {
	case AdapterDiscrete:
		return "discrete"
	case AdapterIntegrated:
		return "integrated"
	case AdapterCPU:
		return "cpu"
	}
	return "unknown"
}

// AdapterInfo describes the adapter Open picked, as the driver reports it.
// Fields the driver leaves empty stay empty.
type AdapterInfo struct {
	Name         string // the device name, e.g. "NVIDIA RTX PRO 4000 Blackwell"
	Description  string // the driver's description, often its name and version
	Vendor       string
	Architecture string
	Backend      string // the graphics API: "Vulkan", "Metal", "D3D12", ...
	Type         AdapterType
	VendorID     uint32 // PCI vendor ID: 0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel
	DeviceID     uint32
}

// Software reports whether the adapter runs on the CPU. wgpu falls back to
// one silently when the real GPU's driver is not found -- on Linux, a
// Vulkan loader that cannot see the vendor's ICD hands out Mesa's
// llvmpipe -- and a product there runs orders of magnitude slower than on
// the GPU.
func (a AdapterInfo) Software() bool {
	if a.Type == AdapterCPU {
		return true
	}
	name := strings.ToLower(a.Name + " " + a.Description)
	for _, s := range []string{"llvmpipe", "lavapipe", "swiftshader"} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

func (a AdapterInfo) String() string {
	s := a.Name
	if s == "" {
		s = a.Description
	}
	return fmt.Sprintf("%s (%s, %s, vendor 0x%04x)", s, a.Backend, a.Type, a.VendorID)
}

// backendName names a WGPUBackendType; the values agree between the v22
// and v29 C APIs.
func backendName(t uint32) string {
	switch t {
	case 1:
		return "Null"
	case 2:
		return "WebGPU"
	case 3:
		return "D3D11"
	case 4:
		return "D3D12"
	case 5:
		return "Metal"
	case 6:
		return "Vulkan"
	case 7:
		return "OpenGL"
	case 8:
		return "OpenGLES"
	}
	return "unknown"
}

// requireHardware fails when TENSAI_WGPU_REQUIRE_HW is set to a true value
// and the adapter is a software one, so a caller that only wants the GPU
// gets an error from Open instead of quietly running on the CPU. Unset or
// false leaves Open as it was.
func requireHardware(a AdapterInfo) error {
	v := os.Getenv("TENSAI_WGPU_REQUIRE_HW")
	if v == "" {
		return nil
	}
	on, err := strconv.ParseBool(v)
	if err != nil {
		return fmt.Errorf("tensai: TENSAI_WGPU_REQUIRE_HW=%q is not a boolean", v)
	}
	if on && a.Software() {
		return fmt.Errorf("tensai: wgpu picked the software adapter %s and TENSAI_WGPU_REQUIRE_HW is set (on Linux, check that the Vulkan loader sees the GPU driver's ICD, e.g. VK_ICD_FILENAMES)", a)
	}
	return nil
}
