package vietasr

// Native library loading. The Go module ships source only; the platform
// libvietasr (~67 MB, ONNX model embedded) is dlopen'd via purego — no cgo —
// and downloaded from the matching GitHub Release on first use.

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	repo       = "dangvansam/viet-asr"
	modulePath = "github.com/dangvansam/viet-asr/bindings/go"
)

var (
	loadOnce sync.Once
	loadErr  error

	vietasrPipelinePreset      func(name string) uintptr
	vietasrPipelineNew         func() uintptr
	vietasrPipelineAddModule   func(p uintptr, name, cfg string) int32
	vietasrPipelineSetBackend  func(p uintptr, backend int32) int32
	vietasrPipelineSetModelDir func(p uintptr, dir string) int32
	vietasrPipelineBuild       func(p uintptr) int32
	vietasrPipelineFree        func(p uintptr)
	vietasrListModulesFn       func() uintptr
	vietasrListPresetsFn       func() uintptr
	vietasrSessionNew          func(p uintptr, sampleRate float32) uintptr
	vietasrSessionFree         func(s uintptr)
	vietasrSessionReset        func(s uintptr)
	vietasrAcceptS16           func(s uintptr, pcm unsafe.Pointer, n int32) int32
	vietasrAcceptF32           func(s uintptr, pcm unsafe.Pointer, n int32) int32
	vietasrPartialResult       func(s uintptr) uintptr
	vietasrResultFn            func(s uintptr) uintptr
	vietasrFinalResult         func(s uintptr) uintptr
	vietasrTranscribeFile      func(p uintptr, path string) uintptr
	vietasrTranscribeBuffer    func(p uintptr, pcm unsafe.Pointer, n int32, sampleRate float32) uintptr
	vietasrSetLogLevelFn       func(level int32)
	vietasrVersionFn           func() uintptr
	vietasrLastErrorFn         func() uintptr
)

// Init loads the native library (downloading it on first use). It is called
// automatically by the API, but may be invoked explicitly to surface load
// errors early.
func Init() error { return ensureLoaded() }

func ensureLoaded() error {
	loadOnce.Do(func() { loadErr = loadLibrary() })
	return loadErr
}

func loadLibrary() error {
	dir, err := resolveNativeDir()
	if err != nil {
		return err
	}
	preloadONNX(dir)
	handle, err := purego.Dlopen(
		filepath.Join(dir, libName("vietasr")),
		purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("vietasr: failed to load native library: %w", err)
	}
	purego.RegisterLibFunc(&vietasrPipelinePreset, handle, "vietasr_pipeline_preset")
	purego.RegisterLibFunc(&vietasrPipelineNew, handle, "vietasr_pipeline_new")
	purego.RegisterLibFunc(&vietasrPipelineAddModule, handle, "vietasr_pipeline_add_module")
	purego.RegisterLibFunc(&vietasrPipelineSetBackend, handle, "vietasr_pipeline_set_backend")
	purego.RegisterLibFunc(&vietasrPipelineSetModelDir, handle, "vietasr_pipeline_set_model_dir")
	purego.RegisterLibFunc(&vietasrPipelineBuild, handle, "vietasr_pipeline_build")
	purego.RegisterLibFunc(&vietasrPipelineFree, handle, "vietasr_pipeline_free")
	purego.RegisterLibFunc(&vietasrListModulesFn, handle, "vietasr_list_modules")
	purego.RegisterLibFunc(&vietasrListPresetsFn, handle, "vietasr_list_presets")
	purego.RegisterLibFunc(&vietasrSessionNew, handle, "vietasr_session_new")
	purego.RegisterLibFunc(&vietasrSessionFree, handle, "vietasr_session_free")
	purego.RegisterLibFunc(&vietasrSessionReset, handle, "vietasr_session_reset")
	purego.RegisterLibFunc(&vietasrAcceptS16, handle, "vietasr_accept_waveform_s16")
	purego.RegisterLibFunc(&vietasrAcceptF32, handle, "vietasr_accept_waveform_f32")
	purego.RegisterLibFunc(&vietasrPartialResult, handle, "vietasr_partial_result")
	purego.RegisterLibFunc(&vietasrResultFn, handle, "vietasr_result")
	purego.RegisterLibFunc(&vietasrFinalResult, handle, "vietasr_final_result")
	purego.RegisterLibFunc(&vietasrTranscribeFile, handle, "vietasr_transcribe_file")
	purego.RegisterLibFunc(&vietasrTranscribeBuffer, handle, "vietasr_transcribe_buffer")
	purego.RegisterLibFunc(&vietasrSetLogLevelFn, handle, "vietasr_set_log_level")
	purego.RegisterLibFunc(&vietasrVersionFn, handle, "vietasr_version")
	purego.RegisterLibFunc(&vietasrLastErrorFn, handle, "vietasr_last_error")
	return nil
}

// preloadONNX loads libonnxruntime with RTLD_GLOBAL so libvietasr's dependency
// resolves regardless of rpath.
func preloadONNX(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		name := e.Name()
		if strings.Contains(name, "onnxruntime") && isLibFile(name) {
			_, _ = purego.Dlopen(filepath.Join(dir, name),
				purego.RTLD_NOW|purego.RTLD_GLOBAL)
		}
	}
}

func resolveNativeDir() (string, error) {
	if d := os.Getenv("VIETASR_NATIVE_DIR"); d != "" {
		return d, nil
	}
	version := bindingVersion()
	dir := filepath.Join(cacheRoot(), "viet-asr", version)
	if fileExists(filepath.Join(dir, libName("vietasr"))) {
		return dir, nil
	}
	if err := downloadNative(dir, version); err != nil {
		return "", err
	}
	return dir, nil
}

func downloadNative(dir, version string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	asset := "viet-asr-native-" + platformKey() + ".tar.gz"
	urls := []string{
		fmt.Sprintf("https://github.com/%s/releases/download/v%s/%s", repo, version, asset),
		fmt.Sprintf("https://github.com/%s/releases/latest/download/%s", repo, asset),
	}
	var lastErr error
	for _, url := range urls {
		if err := fetchAndExtract(url, dir); err != nil {
			lastErr = err
			continue
		}
		if fileExists(filepath.Join(dir, libName("vietasr"))) {
			return nil
		}
	}
	return fmt.Errorf("vietasr: could not download the native library for %s "+
		"(set VIETASR_NATIVE_DIR to override): %w", platformKey(), lastErr)
}

func fetchAndExtract(url, dir string) error {
	resp, err := http.Get(url) //nolint:gosec // release asset URL
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		out := filepath.Join(dir, filepath.Base(hdr.Name))
		switch hdr.Typeflag {
		case tar.TypeReg:
			f, err := os.OpenFile(out, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil { //nolint:gosec // trusted asset
				f.Close()
				return err
			}
			f.Close()
		case tar.TypeSymlink:
			_ = os.Remove(out)
			_ = os.Symlink(hdr.Linkname, out)
		}
	}
	return nil
}

// bindingVersion reports this module's released version (e.g. "0.2.0") so the
// native download targets the matching GitHub Release; "latest" when unknown
// (local checkout / replace directive).
func bindingVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "latest"
	}
	if info.Main.Path == modulePath && isReleaseVersion(info.Main.Version) {
		return strings.TrimPrefix(info.Main.Version, "v")
	}
	for _, d := range info.Deps {
		if d.Path == modulePath && isReleaseVersion(d.Version) {
			return strings.TrimPrefix(d.Version, "v")
		}
	}
	return "latest"
}

func isReleaseVersion(v string) bool {
	return v != "" && v != "(devel)" && strings.HasPrefix(v, "v")
}

func platformKey() string {
	switch runtime.GOOS {
	case "windows":
		return "win-x64"
	case "darwin":
		return "darwin-universal2"
	default:
		if runtime.GOARCH == "arm64" {
			return "linux-arm64"
		}
		return "linux-x64"
	}
}

func libName(stem string) string {
	switch runtime.GOOS {
	case "windows":
		return stem + ".dll"
	case "darwin":
		return "lib" + stem + ".dylib"
	default:
		return "lib" + stem + ".so"
	}
}

func isLibFile(name string) bool {
	return strings.HasSuffix(name, ".dll") ||
		strings.HasSuffix(name, ".dylib") ||
		strings.Contains(name, ".so")
}

func cacheRoot() string {
	if x := os.Getenv("XDG_CACHE_HOME"); x != "" {
		return x
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".cache")
	}
	return os.TempDir()
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// goString copies a NUL-terminated C string returned across the FFI boundary.
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Pointer(p + uintptr(n))) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(p)), n))
}
