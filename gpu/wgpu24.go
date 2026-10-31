//go:build wgpu24 && (linux || darwin || windows)

package gpu

// The WebGPU compute backend against the reworked wgpu-native C API
// (v24 and later; these bindings target the v29.0.1.1 release). Build with
// -tags wgpu24 instead of -tags wgpu; the exported surface (Open, Device,
// Power) is identical, so everything else — including _example/wgpu —
// works unchanged.
//
// What justifies the second binding generation is
// WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter, which un-hides
// non-conformant Vulkan drivers — concretely Mesa's dozen
// (Vulkan-on-D3D12), the only route to the real Device inside WSL2.
//
// The new API passes WGPUStringView and callback-info structs by value.
// Every such struct here is reached through a pointer field except the
// three callback-info arguments, whose calling convention differs between
// SysV/AAPCS and Windows x64; wgpu24_callinfo.go and its _windows sibling
// hold that difference, and nothing else in this file has to care.
//
// Use a wgpu-native v29-series shared library with this build; the v22
// library will fail symbol registration cleanly.

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Device is a WebGPU device with a compiled matmul pipeline. Open one with
// Open, use it from any goroutine (calls are serialized internally, except
// that a download waits for the GPU outside that: see readback_wgpu.go),
// and Close it when done.
type Device struct {
	mu         sync.Mutex
	instance   uintptr
	adapter    uintptr
	device     uintptr
	queue      uintptr
	module     uintptr
	module2    uintptr // optional integer-dot module; 0 if unsupported
	module3    uintptr // optional f16 module; 0 if unsupported
	module4    uintptr // optional tiled-GEMM module; 0 if it did not compile
	hasIntDot  bool
	hasF16     bool
	pipes      gpuPipelines
	readbacks  []gpuReadbackBuffer           // idle staging buffers, see takeReadback
	downloads  map[*PendingDownload]struct{} // begun and not yet waited
	waits      gpuWaits                      // mappings in flight, see readback_wgpu.go
	pool       gpuBufferPool
	bgCache    map[bgKey]uintptr
	batchEnc   uintptr // open command encoder while batching, see BeginBatch
	batchPass  uintptr // open compute pass inside the batch encoder
	uarBuf     uintptr // uniform staging arena, see stageUniform
	uarStage   []byte
	uarOff     uint64
	offsCache  map[string]uintptr // resident attention-offset buffers
	name       string
	info       AdapterInfo
	maxStorage uint64 // usable bytes per storage buffer, 0 = unknown
	closed     bool   // no new operation may start
	released   bool   // closeLocked has released everything
}

// Power tells Open which adapter to prefer on machines with more than
// one Device. It is a hint: when only one adapter exists it is returned
// regardless.
type Power uint32

const (
	Default         Power = 0 // let the driver pick
	LowPower        Power = 1 // prefer the integrated Device
	HighPerformance Power = 2 // prefer the discrete Device
)

// v29 API constants. Flags widened to 64 bits, statuses now start at 1.
const (
	wgpuSTypeShaderSourceWGSL = 0x00000002
	wgpuSTypeInstanceExtras   = 0x00030004

	wgpuBufferUsageMapRead uint64 = 0x01
	wgpuBufferUsageCopySrc uint64 = 0x04
	wgpuBufferUsageCopyDst uint64 = 0x08
	wgpuBufferUsageUniform uint64 = 0x40
	wgpuBufferUsageStorage uint64 = 0x80

	wgpuMapModeRead uint64 = 0x01

	wgpuStatusSuccess = 1 // request-adapter/device and map-async alike

	wgpuCallbackModeAllowProcessEvents = 2
	wgpuCallbackModeAllowSpontaneous   = 3

	wgpuInstanceFlagAllowNoncompliant uint64 = 1 << 3

	wgpuStrlen = ^uintptr(0) // WGPU_STRLEN: treat data as null-terminated
)

// Hand-mirrored v29 structs; field order, widths, and padding must match
// the 64-bit C layouts exactly.

type wgpuStringView struct {
	data   *byte
	length uintptr
}

// sv builds a WGPUStringView over a Go string. The returned backing slice
// must be kept alive across the C call.
func sv(s string) (wgpuStringView, []byte) {
	if s == "" {
		return wgpuStringView{}, nil
	}
	b := []byte(s)
	return wgpuStringView{data: &b[0], length: uintptr(len(b))}, b
}

type wgpuChainedStruct struct {
	next  uintptr
	sType uint32
	_     uint32
}

type wgpuInstanceExtras struct {
	chain        wgpuChainedStruct
	backends     uint64 // 0 = all
	flags        uint64
	dx12Compiler uint32
	gles3Minor   uint32
	glFence      uint32
	_            uint32
	dxcPath      wgpuStringView
	dxcMaxSM     uint32
	dx12Swap     uint32
	budgetCreate uintptr
	budgetLoss   uintptr
	dhType       uint32 // WGPUNativeDisplayHandle, union flattened
	_            uint32
	dh1, dh2     uintptr
}

type wgpuInstanceDescriptor struct {
	nextInChain          *wgpuInstanceExtras
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
}

type wgpuRequestAdapterOptions struct {
	nextInChain       uintptr
	featureLevel      uint32 // 0 = undefined, defaults to Core
	powerPreference   uint32
	forceFallback     uint32
	backendType       uint32
	compatibleSurface uintptr
}

// wgpuCallbackInfo is the shared {mode, callback, userdata1, userdata2}
// shape of WGPURequestAdapterCallbackInfo, WGPURequestDeviceCallbackInfo,
// and WGPUBufferMapCallbackInfo. It is passed to the C API by value.
type wgpuCallbackInfo struct {
	nextInChain uintptr
	mode        uint32
	_           uint32
	callback    uintptr
	userdata1   uintptr
	userdata2   uintptr
}

type wgpuUncapturedErrorCallbackInfo struct {
	nextInChain uintptr
	callback    uintptr
	userdata1   uintptr
	userdata2   uintptr
}

// wgpuLimits mirrors the v29 WGPULimits: a chain pointer, 14 leading u32s,
// and three u64 sizes split by u32 runs (the tail gained maxImmediateSize
// and lost maxInterStageShaderComponents relative to v22, keeping 12
// u32s). Only the named sizes are read; the rest ride along so the
// adapter's limits can be requested back verbatim.
type wgpuLimits struct {
	nextInChain                 uintptr
	u32a                        [14]uint32
	maxUniformBufferBindingSize uint64
	maxStorageBufferBindingSize uint64
	u32b                        [3]uint32
	_                           uint32
	maxBufferSize               uint64
	u32c                        [12]uint32
}

type wgpuDeviceDescriptor struct {
	nextInChain          uintptr
	label                wgpuStringView
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
	queueNext            uintptr // WGPUQueueDescriptor, inlined
	queueLabel           wgpuStringView
	lost                 wgpuCallbackInfo // WGPUDeviceLostCallbackInfo, same shape
	uncaptured           wgpuUncapturedErrorCallbackInfo
}

type wgpuShaderSourceWGSL struct {
	chain wgpuChainedStruct
	code  wgpuStringView
}

type wgpuShaderModuleDescriptor struct {
	nextInChain *wgpuShaderSourceWGSL
	label       wgpuStringView
}

type wgpuComputeState struct {
	nextInChain   uintptr
	module        uintptr
	entryPoint    wgpuStringView
	constantCount uintptr
	constants     uintptr
}

type wgpuComputePipelineDescriptor struct {
	nextInChain uintptr
	label       wgpuStringView
	layout      uintptr // 0 = auto layout
	compute     wgpuComputeState
}

type wgpuBufferDescriptor struct {
	nextInChain      uintptr
	label            wgpuStringView
	usage            uint64
	size             uint64
	mappedAtCreation uint32
	_                uint32
}

type wgpuBindGroupEntry struct {
	nextInChain uintptr
	binding     uint32
	_           uint32
	buffer      uintptr
	offset      uint64
	size        uint64
	sampler     uintptr
	textureView uintptr
}

type wgpuBindGroupDescriptor struct {
	nextInChain uintptr
	label       wgpuStringView
	layout      uintptr
	entryCount  uintptr
	entries     *wgpuBindGroupEntry
}

type wgpuAdapterInfo struct {
	nextInChain  uintptr
	vendor       wgpuStringView
	architecture wgpuStringView
	device       wgpuStringView
	description  wgpuStringView
	backendType  uint32
	adapterType  uint32
	vendorID     uint32
	deviceID     uint32
	sgMin, sgMax uint32
}

// Function pointers resolved from the shared library. RequestAdapter,
// RequestDevice, and MapAsync are not here: they take a callback-info
// struct by value, so they live in the per-OS wgpu24_callinfo files behind
// the requestAdapter/requestDevice/bufferMapAsync wrappers.
var (
	fnCreateInstance          func(*wgpuInstanceDescriptor) uintptr
	fnAdapterGetInfo          func(uintptr, *wgpuAdapterInfo) uint32
	fnAdapterHasFeature       func(uintptr, uint32) uint32
	fnAdapterGetLimits        func(uintptr, *wgpuLimits) uint32
	fnDeviceGetQueue          func(uintptr) uintptr
	fnDeviceCreateShaderMod   func(uintptr, *wgpuShaderModuleDescriptor) uintptr
	fnDeviceCreatePipeline    func(uintptr, *wgpuComputePipelineDescriptor) uintptr
	fnPipelineGetLayout       func(uintptr, uint32) uintptr
	fnDeviceCreateBuffer      func(uintptr, *wgpuBufferDescriptor) uintptr
	fnDeviceCreateBindGroup   func(uintptr, *wgpuBindGroupDescriptor) uintptr
	fnDeviceCreateCmdEncoder  func(uintptr, unsafe.Pointer) uintptr
	fnEncoderBeginComputePass func(uintptr, unsafe.Pointer) uintptr
	fnPassSetPipeline         func(uintptr, uintptr)
	fnPassSetBindGroup        func(uintptr, uint32, uintptr, uintptr, unsafe.Pointer)
	fnPassDispatch            func(uintptr, uint32, uint32, uint32)
	fnPassEnd                 func(uintptr)
	fnEncoderCopyBuffer       func(uintptr, uintptr, uint64, uintptr, uint64, uint64)
	fnEncoderFinish           func(uintptr, unsafe.Pointer) uintptr
	fnQueueSubmit             func(uintptr, uintptr, unsafe.Pointer)
	fnQueueWriteBuffer        func(uintptr, uintptr, uint64, unsafe.Pointer, uintptr)
	fnBufferGetConstMapped    func(uintptr, uintptr, uintptr) unsafe.Pointer
	fnBufferUnmap             func(uintptr)
	fnDevicePoll              func(uintptr, uint32, unsafe.Pointer) uint32
	fnBufferRelease           func(uintptr)
	fnBindGroupRelease        func(uintptr)
	fnCmdBufferRelease        func(uintptr)
	fnCmdEncoderRelease       func(uintptr)
	fnPassRelease             func(uintptr)
	fnShaderModRelease        func(uintptr)
	fnLayoutRelease           func(uintptr)
	fnPipelineRelease         func(uintptr)
	fnQueueRelease            func(uintptr)
	fnDeviceRelease           func(uintptr)
	fnAdapterRelease          func(uintptr)
	fnInstanceRelease         func(uintptr)

	// fnDeviceCreateShaderModTrusted is nil when the library lacks it.
	fnDeviceCreateShaderModTrusted func(uintptr, *wgpuShaderModuleDescriptor, uint64) uintptr
)

// registerOptional is purego.RegisterLibFunc for a symbol the library may
// not export, which RegisterLibFunc answers with a panic.
func registerOptional(fptr any, lib uintptr, name string) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	purego.RegisterLibFunc(fptr, lib, name)
	return true
}

// Callback plumbing, identical in spirit to the v22 file: NewCallback slots
// are permanent, created once, routed through globals under wgpuMu. The
// adapter, device and uncaptured-error callbacks fire synchronously inside
// the call that causes them, and every such call is made under wgpuMu. Map
// callbacks are the exception: they fire in whichever call next processes
// the device's completed work, including the readback poller's polls, which
// run without wgpuMu, so they report through the registry in
// readback_wgpu.go instead. (wgpuDevicePoll's own errors abort the process
// in wgpu-native rather than reaching the uncaptured-error callback.)
var (
	wgpuMu       sync.Mutex
	loadOnce     sync.Once
	loadErr      error
	cbAdapter    uintptr
	cbDevice     uintptr
	cbFailMsg    string
	uncapturedCB string

	adapterCB, deviceCB, mapCB, errorCB uintptr
)

// svString copies a WGPUStringView arriving from C as its two register
// halves (data pointer, length).
func svString(data, length uintptr) string {
	if data == 0 || length == 0 {
		return ""
	}
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&data))
	if length == wgpuStrlen {
		var out []byte
		for i := 0; ; i++ {
			c := *(*byte)(unsafe.Add(ptr, i))
			if c == 0 {
				return string(out)
			}
			out = append(out, c)
		}
	}
	return string(unsafe.Slice((*byte)(ptr), length))
}

func viewString(v wgpuStringView) string {
	return svString(uintptr(unsafe.Pointer(v.data)), v.length)
}

func libCandidates() []string {
	if p := os.Getenv("TENSAI_WGPU_LIB"); p != "" {
		return []string{p}
	}
	switch runtime.GOOS {
	case "darwin":
		return []string{"libwgpu_native.dylib"}
	case "windows":
		// The release zip ships wgpu_native.dll; the name with the lib
		// prefix turns up in MSYS2/MinGW-style installs. Without these
		// a Windows build asked for a .so and reported the library
		// missing however plainly it sat beside the binary.
		return []string{"wgpu_native.dll", "libwgpu_native.dll"}
	}
	return []string{"libwgpu_native.so"}
}

func loadWGPU() error {
	loadOnce.Do(func() {
		var lib uintptr
		var err error
		var loaded string
		for _, name := range libCandidates() {
			lib, err = dlopenWGPU(name)
			if err == nil {
				loaded = name
				break
			}
		}
		if lib == 0 {
			loadErr = fmt.Errorf("tensai: cannot load wgpu-native (set TENSAI_WGPU_LIB): %w", err)
			return
		}
		// Same symbol names, different ABI: calling a v22 library through
		// these bindings crashes, so gate on the packed version number.
		// Name the file that was loaded: with several versions installed
		// the mismatch is usually TENSAI_WGPU_LIB pointing at the wrong one.
		var fnGetVersion func() uint32
		purego.RegisterLibFunc(&fnGetVersion, lib, "wgpuGetVersion")
		if major := fnGetVersion() >> 24; major < 24 {
			loadErr = fmt.Errorf("tensai: %s is wgpu-native v%d, too old for -tags wgpu24 (needs v24+): point TENSAI_WGPU_LIB at a v24+ library, or build with -tags wgpu instead", loaded, major)
			return
		}
		for _, f := range []struct {
			ptr  any
			name string
		}{
			{&fnCreateInstance, "wgpuCreateInstance"},
			{&fnAdapterGetInfo, "wgpuAdapterGetInfo"},
			{&fnAdapterHasFeature, "wgpuAdapterHasFeature"},
			{&fnAdapterGetLimits, "wgpuAdapterGetLimits"},
			{&fnDeviceGetQueue, "wgpuDeviceGetQueue"},
			{&fnDeviceCreateShaderMod, "wgpuDeviceCreateShaderModule"},
			{&fnDeviceCreatePipeline, "wgpuDeviceCreateComputePipeline"},
			{&fnPipelineGetLayout, "wgpuComputePipelineGetBindGroupLayout"},
			{&fnDeviceCreateBuffer, "wgpuDeviceCreateBuffer"},
			{&fnDeviceCreateBindGroup, "wgpuDeviceCreateBindGroup"},
			{&fnDeviceCreateCmdEncoder, "wgpuDeviceCreateCommandEncoder"},
			{&fnEncoderBeginComputePass, "wgpuCommandEncoderBeginComputePass"},
			{&fnPassSetPipeline, "wgpuComputePassEncoderSetPipeline"},
			{&fnPassSetBindGroup, "wgpuComputePassEncoderSetBindGroup"},
			{&fnPassDispatch, "wgpuComputePassEncoderDispatchWorkgroups"},
			{&fnPassEnd, "wgpuComputePassEncoderEnd"},
			{&fnEncoderCopyBuffer, "wgpuCommandEncoderCopyBufferToBuffer"},
			{&fnEncoderFinish, "wgpuCommandEncoderFinish"},
			{&fnQueueSubmit, "wgpuQueueSubmit"},
			{&fnQueueWriteBuffer, "wgpuQueueWriteBuffer"},
			{&fnBufferGetConstMapped, "wgpuBufferGetConstMappedRange"},
			{&fnBufferUnmap, "wgpuBufferUnmap"},
			{&fnDevicePoll, "wgpuDevicePoll"},
			{&fnBufferRelease, "wgpuBufferRelease"},
			{&fnBindGroupRelease, "wgpuBindGroupRelease"},
			{&fnCmdBufferRelease, "wgpuCommandBufferRelease"},
			{&fnCmdEncoderRelease, "wgpuCommandEncoderRelease"},
			{&fnPassRelease, "wgpuComputePassEncoderRelease"},
			{&fnShaderModRelease, "wgpuShaderModuleRelease"},
			{&fnLayoutRelease, "wgpuBindGroupLayoutRelease"},
			{&fnPipelineRelease, "wgpuComputePipelineRelease"},
			{&fnQueueRelease, "wgpuQueueRelease"},
			{&fnDeviceRelease, "wgpuDeviceRelease"},
			{&fnAdapterRelease, "wgpuAdapterRelease"},
			{&fnInstanceRelease, "wgpuInstanceRelease"},
		} {
			purego.RegisterLibFunc(f.ptr, lib, f.name)
		}
		registerCallInfoFns(lib)
		// Optional: only some v24+ releases export it.
		registerOptional(&fnDeviceCreateShaderModTrusted, lib, "wgpuDeviceCreateShaderModuleTrusted")

		// New-API callbacks receive WGPUStringView messages by value; on
		// SysV/AAPCS a 16-byte aggregate arrives as two register words, so
		// the Go side declares (data, length) pairs.
		adapterCB = purego.NewCallback(func(status uint32, adapter, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			if status == wgpuStatusSuccess {
				cbAdapter = adapter
			} else {
				cbFailMsg = svString(msgData, msgLen)
			}
			return 0
		})
		deviceCB = purego.NewCallback(func(status uint32, device, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			if status == wgpuStatusSuccess {
				cbDevice = device
			} else {
				cbFailMsg = svString(msgData, msgLen)
			}
			return 0
		})
		mapCB = purego.NewCallback(func(status uint32, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			ok := status == wgpuStatusSuccess
			var msg string
			if !ok {
				msg = svString(msgData, msgLen)
			}
			mapDone(ud1, ok, msg)
			return 0
		})
		errorCB = purego.NewCallback(func(device uintptr, errType uint32, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			if uncapturedCB == "" {
				uncapturedCB = svString(msgData, msgLen)
			}
			return 0
		})
	})
	return loadErr
}

// Open loads wgpu-native, picks an adapter, and compiles the matmul
// pipeline. On machines with several GPUs an optional Power steers the
// choice: LowPower prefers the integrated Device, HighPerformance the
// discrete one. Non-conformant Vulkan drivers (Mesa's dozen inside WSL2)
// are allowed. The returned Device is safe for concurrent use.
func Open(power ...Power) (*Device, error) {
	return open(matmulWGSL, power...)
}

// open is Open with the main module's WGSL as an argument, so a test can
// hand it one that does not compile and walk the error path.
func open(src string, power ...Power) (*Device, error) {
	if err := loadWGPU(); err != nil {
		return nil, err
	}
	poolMax, err := poolLimit()
	if err != nil {
		return nil, err
	}
	wgpuMu.Lock()
	defer wgpuMu.Unlock()

	g := &Device{pool: gpuBufferPool{max: poolMax}}
	extras := wgpuInstanceExtras{
		chain: wgpuChainedStruct{sType: wgpuSTypeInstanceExtras},
		flags: wgpuInstanceFlagAllowNoncompliant,
	}
	iDesc := wgpuInstanceDescriptor{nextInChain: &extras}
	g.instance = fnCreateInstance(&iDesc)
	runtime.KeepAlive(&extras)
	runtime.KeepAlive(&iDesc)
	if g.instance == 0 {
		return nil, errors.New("tensai: wgpuCreateInstance failed")
	}
	opts := wgpuRequestAdapterOptions{}
	if len(power) > 0 {
		opts.powerPreference = uint32(power[0])
	}
	// wgpu-native still fires these callbacks synchronously.
	cbAdapter, cbDevice, cbFailMsg = 0, 0, ""
	requestAdapter(g.instance, &opts, wgpuCallbackInfo{
		mode: wgpuCallbackModeAllowSpontaneous, callback: adapterCB,
	})
	runtime.KeepAlive(&opts)
	if cbAdapter == 0 {
		g.closeLocked()
		return nil, fmt.Errorf("tensai: wgpu adapter request failed: %s", cbFailMsg)
	}
	g.adapter = cbAdapter

	// The info strings are tiny and Open runs once per process, so they
	// are copied and never handed back to wgpuAdapterInfoFreeMembers.
	var info wgpuAdapterInfo
	fnAdapterGetInfo(g.adapter, &info)
	runtime.KeepAlive(&info)
	g.info = AdapterInfo{
		Name:         viewString(info.device),
		Description:  viewString(info.description),
		Vendor:       viewString(info.vendor),
		Architecture: viewString(info.architecture),
		Backend:      backendName(info.backendType),
		VendorID:     info.vendorID,
		DeviceID:     info.deviceID,
	}
	g.name = g.info.Name
	if g.name == "" {
		g.name = g.info.Description
	}
	switch info.adapterType {
	case 1:
		g.info.Type = AdapterDiscrete
		g.name += " (discrete)"
	case 2:
		g.info.Type = AdapterIntegrated
		g.name += " (integrated)"
	case 3:
		g.info.Type = AdapterCPU
		g.name += " (cpu)"
	}
	if err := requireHardware(g.info); err != nil {
		g.closeLocked()
		return nil, err
	}

	// Request the adapter's own limits back so big storage buffers are not
	// capped at the conservative defaults (128MiB bindings).
	var lim wgpuLimits
	dDesc := wgpuDeviceDescriptor{
		uncaptured: wgpuUncapturedErrorCallbackInfo{callback: errorCB},
	}
	if fnAdapterGetLimits(g.adapter, &lim) == 1 {
		lim.nextInChain = 0
		dDesc.requiredLimits = uintptr(unsafe.Pointer(&lim))
		g.maxStorage = min(lim.maxStorageBufferBindingSize, lim.maxBufferSize)
	}
	// Shader-f16 halves the resident KV cache; ask for it when the
	// adapter advertises it, and drop it if the device still refuses.
	const featShaderF16 = 0x0B // WGPUFeatureName_ShaderF16
	feats := []uint32{featShaderF16}
	if fnAdapterHasFeature(g.adapter, featShaderF16) == 1 {
		dDesc.requiredFeatureCount = 1
		dDesc.requiredFeatures = uintptr(unsafe.Pointer(&feats[0]))
		g.hasF16 = true
	}
	requestDevice(g.adapter, &dDesc, wgpuCallbackInfo{
		mode: wgpuCallbackModeAllowSpontaneous, callback: deviceCB,
	})
	if cbDevice == 0 && g.hasF16 {
		g.hasF16 = false
		dDesc.requiredFeatureCount = 0
		dDesc.requiredFeatures = 0
		requestDevice(g.adapter, &dDesc, wgpuCallbackInfo{
			mode: wgpuCallbackModeAllowSpontaneous, callback: deviceCB,
		})
	}
	runtime.KeepAlive(&dDesc)
	runtime.KeepAlive(&lim)
	runtime.KeepAlive(feats)
	if cbDevice == 0 {
		g.closeLocked()
		return nil, fmt.Errorf("tensai: wgpu device request failed: %s", cbFailMsg)
	}
	g.device = cbDevice
	g.queue = fnDeviceGetQueue(g.device)

	uncapturedCB = ""
	code, codeBuf := sv(src)
	wgsl := wgpuShaderSourceWGSL{
		chain: wgpuChainedStruct{sType: wgpuSTypeShaderSourceWGSL},
		code:  code,
	}
	smDesc := wgpuShaderModuleDescriptor{nextInChain: &wgsl}
	g.module = fnDeviceCreateShaderMod(g.device, &smDesc)
	runtime.KeepAlive(&wgsl)
	runtime.KeepAlive(&smDesc)
	runtime.KeepAlive(codeBuf)
	if g.module == 0 || uncapturedCB != "" {
		g.closeLocked()
		return nil, fmt.Errorf("tensai: wgpu shader compilation failed: %s", uncapturedCB)
	}
	if err := g.initPipelines(); err != nil {
		g.closeLocked()
		return nil, err
	}
	return g, nil
}

// makeModuleFrom compiles an extra WGSL module, returning 0 on failure
// (the caller checks uncapturedCB for the reason).
func (g *Device) makeModuleFrom(src string) uintptr {
	code, codeBuf := sv(src)
	wgsl := wgpuShaderSourceWGSL{
		chain: wgpuChainedStruct{sType: wgpuSTypeShaderSourceWGSL},
		code:  code,
	}
	smDesc := wgpuShaderModuleDescriptor{nextInChain: &wgsl}
	m := fnDeviceCreateShaderMod(g.device, &smDesc)
	runtime.KeepAlive(&wgsl)
	runtime.KeepAlive(&smDesc)
	runtime.KeepAlive(codeBuf)
	return m
}

// makeModuleTrusted compiles a module the way makeModuleFrom does, but
// without the checks wgpu otherwise adds to every shader: a clamp or a
// test on every storage-buffer index, and a counter on every loop. The
// module's own index arithmetic must then keep every access in bounds.
// Libraries without wgpuDeviceCreateShaderModuleTrusted, and a set
// gemmChecked, get the checked module.
func (g *Device) makeModuleTrusted(src string) uintptr {
	if fnDeviceCreateShaderModTrusted == nil || gemmChecked {
		return g.makeModuleFrom(src)
	}
	code, codeBuf := sv(src)
	wgsl := wgpuShaderSourceWGSL{
		chain: wgpuChainedStruct{sType: wgpuSTypeShaderSourceWGSL},
		code:  code,
	}
	smDesc := wgpuShaderModuleDescriptor{nextInChain: &wgsl}
	m := fnDeviceCreateShaderModTrusted(g.device, &smDesc, 0) // WGPUShaderRuntimeChecks_None
	runtime.KeepAlive(&wgsl)
	runtime.KeepAlive(&smDesc)
	runtime.KeepAlive(codeBuf)
	return m
}

// makePipelineIn builds a compute pipeline for one entry point of an
// explicit module.
func (g *Device) makePipelineIn(module uintptr, entry string) uintptr {
	e, eBuf := sv(entry)
	pDesc := wgpuComputePipelineDescriptor{
		compute: wgpuComputeState{module: module, entryPoint: e},
	}
	p := fnDeviceCreatePipeline(g.device, &pDesc)
	runtime.KeepAlive(&pDesc)
	runtime.KeepAlive(eBuf)
	return p
}

// makePipeline builds a compute pipeline for one entry point of g.module.
func (g *Device) makePipeline(entry string) uintptr {
	e, eBuf := sv(entry)
	pDesc := wgpuComputePipelineDescriptor{
		compute: wgpuComputeState{module: g.module, entryPoint: e},
	}
	p := fnDeviceCreatePipeline(g.device, &pDesc)
	runtime.KeepAlive(&pDesc)
	runtime.KeepAlive(eBuf)
	return p
}

// Backend names the wgpu-native C API these bindings target, for builds
// that report which GPU path they were compiled with.
func Backend() string { return "wgpu24" }

// Name reports the adapter wgpu selected, e.g. "Microsoft Direct3D12 (AMD
// Radeon(TM) Graphics) (integrated)" — handy for checking which Device a power
// preference landed on.
func (g *Device) Name() string { return g.name }

// Adapter describes the adapter Open picked: its backend, vendor, and
// whether it is a GPU at all (see AdapterInfo.Software).
func (g *Device) Adapter() AdapterInfo { return g.info }

// Close releases the device and every object Open created, after the
// downloads in flight have landed. The Device must not be used afterwards.
func (g *Device) Close() {
	if !g.beginClose() {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	g.closeLocked()
}

// closeLocked is the release half of Close for a caller that already
// holds wgpuMu -- Close, or Open backing out of a device it could not
// finish. wgpuMu is not reentrant, so calling Close there would wait on
// the lock Open itself holds.
func (g *Device) closeLocked() {
	if g.released {
		return
	}
	g.closed, g.released = true, true
	g.releaseReadback()
	g.releasePool()
	g.releasePipelines()
	for _, r := range []struct {
		fn func(uintptr)
		h  uintptr
	}{
		{fnShaderModRelease, g.module4},
		{fnShaderModRelease, g.module3},
		{fnShaderModRelease, g.module2},
		{fnShaderModRelease, g.module},
		{fnQueueRelease, g.queue},
		{fnDeviceRelease, g.device},
		{fnAdapterRelease, g.adapter},
		{fnInstanceRelease, g.instance},
	} {
		if r.h != 0 {
			r.fn(r.h)
		}
	}
}

// The primitives below back the shared Tensor layer in wgputensor.go;
// their signatures match the v22 bindings so that layer compiles against
// either API generation.

func (g *Device) newBuffer(usage, size uint64) uintptr {
	desc := wgpuBufferDescriptor{usage: usage, size: size}
	buf := fnDeviceCreateBuffer(g.device, &desc)
	runtime.KeepAlive(&desc)
	return buf
}

func (g *Device) makeBindGroup(layout uintptr, entries []wgpuBindGroupEntry) uintptr {
	desc := wgpuBindGroupDescriptor{layout: layout, entryCount: uintptr(len(entries)), entries: &entries[0]}
	bg := fnDeviceCreateBindGroup(g.device, &desc)
	runtime.KeepAlive(&desc)
	return bg
}

// mapForRead asks for the first bytes of buf (usage MapRead) to be mapped;
// the callback reports to mapDone with id.
func (g *Device) mapForRead(buf uintptr, bytes uint64, id uintptr) {
	bufferMapAsync(buf, wgpuMapModeRead, 0, uintptr(bytes), wgpuCallbackInfo{
		mode: wgpuCallbackModeAllowProcessEvents, callback: mapCB, userdata1: id,
	})
}

// pollDevice processes the device's completed work without blocking,
// firing the callbacks it makes ready.
func (g *Device) pollDevice() { fnDevicePoll(g.device, 0, nil) }
