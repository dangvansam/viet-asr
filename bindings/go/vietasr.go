// Package vietasr is the Go binding for the VietASR Vietnamese speech AI SDK.
//
// It is pure Go: the native libvietasr is loaded via purego (no cgo) and
// downloaded from the matching GitHub Release on first use. See native.go.
package vietasr

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type Backend int

const (
	BackendAuto   Backend = 0
	BackendOnnx   Backend = 1
	BackendCoreML Backend = 2
)

type LogLevel int

const (
	LogTrace LogLevel = 0
	LogDebug LogLevel = 1
	LogInfo  LogLevel = 2
	LogWarn  LogLevel = 3
	LogError LogLevel = 4
	LogOff   LogLevel = 5
)

type Pipeline struct {
	handle uintptr
}

type Session struct {
	handle uintptr
}

type Result struct {
	Text     string                   `json:"text"`
	Partial  string                   `json:"partial"`
	IsFinal  bool                     `json:"is_final"`
	Segments []map[string]interface{} `json:"segments,omitempty"`
	Speakers []map[string]interface{} `json:"speakers,omitempty"`
	Extra    map[string]interface{}   `json:"-"`
	Raw      string                   `json:"-"`
}

func (r *Result) Field(key string) interface{} {
	if r.Extra == nil {
		return nil
	}
	return r.Extra[key]
}

func parseResult(raw string) Result {
	r := Result{Raw: raw}
	if raw == "" {
		return r
	}
	var generic map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &generic); err != nil {
		return r
	}
	if v, ok := generic["text"].(string); ok {
		r.Text = v
	}
	if v, ok := generic["partial"].(string); ok {
		r.Partial = v
	}
	if v, ok := generic["is_final"].(bool); ok {
		r.IsFinal = v
	}
	if v, ok := generic["segments"].([]interface{}); ok {
		for _, seg := range v {
			if m, ok := seg.(map[string]interface{}); ok {
				r.Segments = append(r.Segments, m)
			}
		}
	}
	if v, ok := generic["speakers"].([]interface{}); ok {
		for _, sp := range v {
			if m, ok := sp.(map[string]interface{}); ok {
				r.Speakers = append(r.Speakers, m)
			}
		}
	}
	r.Extra = generic
	return r
}

func lastError() string {
	if vietasrLastErrorFn == nil {
		return ""
	}
	return goString(vietasrLastErrorFn())
}

func errorOr(msg string) error {
	if e := lastError(); e != "" {
		return errors.New(e)
	}
	return errors.New(msg)
}

// PipelinePreset constructs a pipeline from a named preset (e.g. "transcribe").
func PipelinePreset(name string) (*Pipeline, error) {
	if err := ensureLoaded(); err != nil {
		return nil, err
	}
	handle := vietasrPipelinePreset(name)
	if handle == 0 {
		return nil, errorOr(fmt.Sprintf("unknown preset: %s", name))
	}
	p := &Pipeline{handle: handle}
	runtime.SetFinalizer(p, func(p *Pipeline) { p.Close() })
	return p, nil
}

// NewPipeline constructs an empty pipeline for module-by-module composition.
// If the native library cannot be loaded, the returned pipeline's methods
// report the load error.
func NewPipeline() *Pipeline {
	if err := ensureLoaded(); err != nil {
		return &Pipeline{}
	}
	p := &Pipeline{handle: vietasrPipelineNew()}
	runtime.SetFinalizer(p, func(p *Pipeline) { p.Close() })
	return p
}

// Add appends a module to the pipeline. config may be nil for default config.
func (p *Pipeline) Add(moduleName string, config map[string]interface{}) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	cfgJSON := "{}"
	if config != nil {
		bytes, _ := json.Marshal(config)
		cfgJSON = string(bytes)
	}
	status := vietasrPipelineAddModule(p.handle, moduleName, cfgJSON)
	if status != 0 {
		return errorOr(fmt.Sprintf("add module failed: %s (%d)", moduleName, int(status)))
	}
	return nil
}

func (p *Pipeline) SetBackend(b Backend) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	if vietasrPipelineSetBackend(p.handle, int32(b)) != 0 {
		return errorOr("set backend failed")
	}
	return nil
}

func (p *Pipeline) SetModelDir(dir string) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	if vietasrPipelineSetModelDir(p.handle, dir) != 0 {
		return errorOr("set model dir failed")
	}
	return nil
}

func (p *Pipeline) Build() error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	if vietasrPipelineBuild(p.handle) != 0 {
		return errorOr("build failed")
	}
	return nil
}

// TranscribeFile reads a WAV file and returns the transcript.
func (p *Pipeline) TranscribeFile(wavPath string) (Result, error) {
	if err := ensureLoaded(); err != nil {
		return Result{}, err
	}
	raw := vietasrTranscribeFile(p.handle, wavPath)
	if raw == 0 {
		return Result{}, errorOr("transcribe_file failed")
	}
	return parseResult(goString(raw)), nil
}

// TranscribeBuffer transcribes raw 16-bit PCM samples at sampleRate.
func (p *Pipeline) TranscribeBuffer(pcm []int16, sampleRate float32) (Result, error) {
	if err := ensureLoaded(); err != nil {
		return Result{}, err
	}
	if len(pcm) == 0 {
		return Result{}, errors.New("empty pcm buffer")
	}
	raw := vietasrTranscribeBuffer(
		p.handle, unsafe.Pointer(&pcm[0]), int32(len(pcm)), sampleRate)
	runtime.KeepAlive(pcm)
	if raw == 0 {
		return Result{}, errorOr("transcribe_buffer failed")
	}
	return parseResult(goString(raw)), nil
}

// Stream returns a Session for incremental audio feeding.
func (p *Pipeline) Stream(sampleRate float32) (*Session, error) {
	if err := ensureLoaded(); err != nil {
		return nil, err
	}
	handle := vietasrSessionNew(p.handle, sampleRate)
	if handle == 0 {
		return nil, errorOr("session creation failed")
	}
	s := &Session{handle: handle}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

func (p *Pipeline) Close() {
	if p.handle != 0 {
		vietasrPipelineFree(p.handle)
		p.handle = 0
	}
}

// Accept feeds a chunk of int16 PCM. Returns true if endpoint was reached.
func (s *Session) Accept(pcm []int16) bool {
	if len(pcm) == 0 {
		return false
	}
	r := vietasrAcceptS16(s.handle, unsafe.Pointer(&pcm[0]), int32(len(pcm)))
	runtime.KeepAlive(pcm)
	return r == 1
}

// AcceptFloat feeds a chunk of float32 PCM in [-1, 1] range.
func (s *Session) AcceptFloat(pcm []float32) bool {
	if len(pcm) == 0 {
		return false
	}
	r := vietasrAcceptF32(s.handle, unsafe.Pointer(&pcm[0]), int32(len(pcm)))
	runtime.KeepAlive(pcm)
	return r == 1
}

func (s *Session) Partial() Result {
	return parseResult(goString(vietasrPartialResult(s.handle)))
}

func (s *Session) Result() Result {
	return parseResult(goString(vietasrResultFn(s.handle)))
}

func (s *Session) Final() Result {
	return parseResult(goString(vietasrFinalResult(s.handle)))
}

func (s *Session) Reset() {
	vietasrSessionReset(s.handle)
}

func (s *Session) Close() {
	if s.handle != 0 {
		vietasrSessionFree(s.handle)
		s.handle = 0
	}
}

// ListModules returns the names of all registered modules.
func ListModules() []string {
	if ensureLoaded() != nil {
		return nil
	}
	var modules []string
	json.Unmarshal([]byte(goString(vietasrListModulesFn())), &modules)
	return modules
}

// ListPresets returns the names of all registered presets.
func ListPresets() []string {
	if ensureLoaded() != nil {
		return nil
	}
	var presets []string
	json.Unmarshal([]byte(goString(vietasrListPresetsFn())), &presets)
	return presets
}

// Version returns the native library version, or "" if it cannot be loaded.
func Version() string {
	if ensureLoaded() != nil {
		return ""
	}
	return goString(vietasrVersionFn())
}

func SetLogLevel(level LogLevel) {
	if ensureLoaded() != nil {
		return
	}
	vietasrSetLogLevelFn(int32(level))
}
