package vietasr

/*
#cgo CFLAGS: -I${SRCDIR}/../../core/include
#cgo LDFLAGS: -L${SRCDIR}/_native -lvietasr -Wl,-rpath,${SRCDIR}/_native

#include <stdlib.h>
#include "vietasr.h"
*/
import "C"

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
	handle *C.VietasrPipeline
}

type Session struct {
	handle *C.VietasrSession
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
	return C.GoString(C.vietasr_last_error())
}

func errorOr(msg string) error {
	if e := lastError(); e != "" {
		return errors.New(e)
	}
	return errors.New(msg)
}

// PipelinePreset constructs a pipeline from a named preset (e.g. "transcribe").
func PipelinePreset(name string) (*Pipeline, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	handle := C.vietasr_pipeline_preset(cname)
	if handle == nil {
		return nil, errorOr(fmt.Sprintf("unknown preset: %s", name))
	}
	p := &Pipeline{handle: handle}
	runtime.SetFinalizer(p, func(p *Pipeline) { p.Close() })
	return p, nil
}

// NewPipeline constructs an empty pipeline for module-by-module composition.
func NewPipeline() *Pipeline {
	handle := C.vietasr_pipeline_new()
	p := &Pipeline{handle: handle}
	runtime.SetFinalizer(p, func(p *Pipeline) { p.Close() })
	return p
}

// Add appends a module to the pipeline. config may be nil for default config.
func (p *Pipeline) Add(moduleName string, config map[string]interface{}) error {
	cname := C.CString(moduleName)
	defer C.free(unsafe.Pointer(cname))
	cfgJSON := "{}"
	if config != nil {
		bytes, _ := json.Marshal(config)
		cfgJSON = string(bytes)
	}
	ccfg := C.CString(cfgJSON)
	defer C.free(unsafe.Pointer(ccfg))
	status := C.vietasr_pipeline_add_module(p.handle, cname, ccfg)
	if status != 0 {
		return errorOr(fmt.Sprintf("add module failed: %s (%d)", moduleName, int(status)))
	}
	return nil
}

func (p *Pipeline) SetBackend(b Backend) error {
	status := C.vietasr_pipeline_set_backend(p.handle, C.VietasrBackend(b))
	if status != 0 {
		return errorOr("set backend failed")
	}
	return nil
}

func (p *Pipeline) SetModelDir(dir string) error {
	cdir := C.CString(dir)
	defer C.free(unsafe.Pointer(cdir))
	status := C.vietasr_pipeline_set_model_dir(p.handle, cdir)
	if status != 0 {
		return errorOr("set model dir failed")
	}
	return nil
}

func (p *Pipeline) Build() error {
	status := C.vietasr_pipeline_build(p.handle)
	if status != 0 {
		return errorOr("build failed")
	}
	return nil
}

// TranscribeFile reads a WAV file and returns the transcript.
func (p *Pipeline) TranscribeFile(wavPath string) (Result, error) {
	cpath := C.CString(wavPath)
	defer C.free(unsafe.Pointer(cpath))
	raw := C.vietasr_transcribe_file(p.handle, cpath)
	if raw == nil {
		return Result{}, errorOr("transcribe_file failed")
	}
	return parseResult(C.GoString(raw)), nil
}

// TranscribeBuffer transcribes raw 16-bit PCM samples at sampleRate.
func (p *Pipeline) TranscribeBuffer(pcm []int16, sampleRate float32) (Result, error) {
	if len(pcm) == 0 {
		return Result{}, errors.New("empty pcm buffer")
	}
	raw := C.vietasr_transcribe_buffer(
		p.handle,
		(*C.short)(unsafe.Pointer(&pcm[0])),
		C.int(len(pcm)),
		C.float(sampleRate),
	)
	if raw == nil {
		return Result{}, errorOr("transcribe_buffer failed")
	}
	return parseResult(C.GoString(raw)), nil
}

// Stream returns a Session for incremental audio feeding.
func (p *Pipeline) Stream(sampleRate float32) (*Session, error) {
	handle := C.vietasr_session_new(p.handle, C.float(sampleRate))
	if handle == nil {
		return nil, errorOr("session creation failed")
	}
	s := &Session{handle: handle}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

func (p *Pipeline) Close() {
	if p.handle != nil {
		C.vietasr_pipeline_free(p.handle)
		p.handle = nil
	}
}

// Accept feeds a chunk of int16 PCM. Returns true if endpoint was reached.
func (s *Session) Accept(pcm []int16) bool {
	if len(pcm) == 0 {
		return false
	}
	return C.vietasr_accept_waveform_s16(
		s.handle,
		(*C.short)(unsafe.Pointer(&pcm[0])),
		C.int(len(pcm)),
	) == 1
}

// AcceptFloat feeds a chunk of float32 PCM in [-1, 1] range.
func (s *Session) AcceptFloat(pcm []float32) bool {
	if len(pcm) == 0 {
		return false
	}
	return C.vietasr_accept_waveform_f32(
		s.handle,
		(*C.float)(unsafe.Pointer(&pcm[0])),
		C.int(len(pcm)),
	) == 1
}

func (s *Session) Partial() Result {
	return parseResult(C.GoString(C.vietasr_partial_result(s.handle)))
}

func (s *Session) Result() Result {
	return parseResult(C.GoString(C.vietasr_result(s.handle)))
}

func (s *Session) Final() Result {
	return parseResult(C.GoString(C.vietasr_final_result(s.handle)))
}

func (s *Session) Reset() {
	C.vietasr_session_reset(s.handle)
}

func (s *Session) Close() {
	if s.handle != nil {
		C.vietasr_session_free(s.handle)
		s.handle = nil
	}
}

// ListModules returns the names of all registered modules.
func ListModules() []string {
	raw := C.GoString(C.vietasr_list_modules())
	var modules []string
	json.Unmarshal([]byte(raw), &modules)
	return modules
}

// ListPresets returns the names of all registered presets.
func ListPresets() []string {
	raw := C.GoString(C.vietasr_list_presets())
	var presets []string
	json.Unmarshal([]byte(raw), &presets)
	return presets
}

func Version() string {
	return C.GoString(C.vietasr_version())
}

func SetLogLevel(level LogLevel) {
	C.vietasr_set_log_level(C.VietasrLogLevel(level))
}
