use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_short, c_void};

#[repr(C)]
struct CPipeline {
    _private: [u8; 0],
}

#[repr(C)]
struct CSession {
    _private: [u8; 0],
}

extern "C" {
    fn vietasr_pipeline_preset(name: *const c_char) -> *mut CPipeline;
    fn vietasr_pipeline_new() -> *mut CPipeline;
    fn vietasr_pipeline_add_module(
        pipeline: *mut CPipeline,
        module_name: *const c_char,
        json_config: *const c_char,
    ) -> c_int;
    fn vietasr_pipeline_set_backend(pipeline: *mut CPipeline, backend: c_int) -> c_int;
    fn vietasr_pipeline_set_model_dir(pipeline: *mut CPipeline, dir: *const c_char) -> c_int;
    fn vietasr_pipeline_build(pipeline: *mut CPipeline) -> c_int;
    fn vietasr_pipeline_free(pipeline: *mut CPipeline);

    fn vietasr_list_modules() -> *const c_char;
    fn vietasr_list_presets() -> *const c_char;

    fn vietasr_session_new(pipeline: *mut CPipeline, sample_rate: c_float) -> *mut CSession;
    fn vietasr_session_free(session: *mut CSession);
    fn vietasr_session_reset(session: *mut CSession);

    fn vietasr_accept_waveform_s16(
        session: *mut CSession,
        pcm: *const c_short,
        len: c_int,
    ) -> c_int;
    fn vietasr_accept_waveform_f32(
        session: *mut CSession,
        pcm: *const c_float,
        len: c_int,
    ) -> c_int;

    fn vietasr_partial_result(session: *mut CSession) -> *const c_char;
    fn vietasr_result(session: *mut CSession) -> *const c_char;
    fn vietasr_final_result(session: *mut CSession) -> *const c_char;

    fn vietasr_transcribe_file(
        pipeline: *mut CPipeline,
        wav_path: *const c_char,
    ) -> *const c_char;
    fn vietasr_transcribe_buffer(
        pipeline: *mut CPipeline,
        pcm: *const c_short,
        len: c_int,
        sample_rate: c_float,
    ) -> *const c_char;

    fn vietasr_set_log_level(level: c_int);
    fn vietasr_version() -> *const c_char;
    fn vietasr_last_error() -> *const c_char;
}

const _: fn() = || {
    let _ = std::ptr::null::<c_void>();
};

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    Auto = 0,
    Onnx = 1,
    CoreMl = 2,
}

#[derive(Debug)]
pub struct VietasrError(pub String);

impl std::fmt::Display for VietasrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vietasr: {}", self.0)
    }
}

impl std::error::Error for VietasrError {}

pub type Result<T> = std::result::Result<T, VietasrError>;

fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

fn last_error(fallback: &str) -> VietasrError {
    let e = cstr_to_string(unsafe { vietasr_last_error() });
    VietasrError(if e.is_empty() { fallback.to_string() } else { e })
}

/// One transcription result — a thin typed view over the JSON envelope.
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub raw_json: String,
}

impl TranscriptResult {
    fn from_json(raw: String) -> Self {
        TranscriptResult { raw_json: raw }
    }

    pub fn text(&self) -> String {
        self.extract_string("text")
    }

    pub fn partial(&self) -> String {
        self.extract_string("partial")
    }

    pub fn is_final(&self) -> bool {
        self.raw_json.contains("\"is_final\":true")
    }

    fn extract_string(&self, key: &str) -> String {
        let needle = format!("\"{}\":\"", key);
        let Some(start) = self.raw_json.find(&needle) else {
            return String::new();
        };
        let value_start = start + needle.len();
        let rest = &self.raw_json[value_start..];
        let mut out = String::new();
        let mut escape = false;
        for c in rest.chars() {
            if escape {
                match c {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    other => out.push(other),
                }
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                break;
            } else {
                out.push(c);
            }
        }
        out
    }
}

/// A speech pipeline. Owns the engine + module templates.
pub struct Pipeline {
    handle: *mut CPipeline,
}

unsafe impl Send for Pipeline {}

impl Pipeline {
    /// Construct a pipeline from a named preset (e.g. "transcribe").
    pub fn preset(name: &str) -> Result<Pipeline> {
        let cname = CString::new(name).map_err(|_| VietasrError("nul in name".into()))?;
        let handle = unsafe { vietasr_pipeline_preset(cname.as_ptr()) };
        if handle.is_null() {
            return Err(last_error(&format!("unknown preset: {name}")));
        }
        Ok(Pipeline { handle })
    }

    /// Construct an empty pipeline for module-by-module composition.
    pub fn new() -> Pipeline {
        Pipeline {
            handle: unsafe { vietasr_pipeline_new() },
        }
    }

    pub fn add(&mut self, module_name: &str) -> Result<&mut Self> {
        self.add_with_config(module_name, "{}")
    }

    pub fn add_with_config(&mut self, module_name: &str, json_config: &str) -> Result<&mut Self> {
        let cname = CString::new(module_name).map_err(|_| VietasrError("nul in name".into()))?;
        let ccfg = CString::new(json_config).map_err(|_| VietasrError("nul in config".into()))?;
        let status =
            unsafe { vietasr_pipeline_add_module(self.handle, cname.as_ptr(), ccfg.as_ptr()) };
        if status != 0 {
            return Err(last_error(&format!("add({module_name}) failed")));
        }
        Ok(self)
    }

    pub fn set_backend(&mut self, backend: Backend) -> Result<&mut Self> {
        let status = unsafe { vietasr_pipeline_set_backend(self.handle, backend as c_int) };
        if status != 0 {
            return Err(last_error("set_backend failed"));
        }
        Ok(self)
    }

    pub fn set_model_dir(&mut self, dir: &str) -> Result<&mut Self> {
        let cdir = CString::new(dir).map_err(|_| VietasrError("nul in dir".into()))?;
        let status = unsafe { vietasr_pipeline_set_model_dir(self.handle, cdir.as_ptr()) };
        if status != 0 {
            return Err(last_error("set_model_dir failed"));
        }
        Ok(self)
    }

    pub fn build(&mut self) -> Result<&mut Self> {
        let status = unsafe { vietasr_pipeline_build(self.handle) };
        if status != 0 {
            return Err(last_error("build failed"));
        }
        Ok(self)
    }

    pub fn transcribe_file(&self, wav_path: &str) -> Result<TranscriptResult> {
        let cpath = CString::new(wav_path).map_err(|_| VietasrError("nul in path".into()))?;
        let raw = unsafe { vietasr_transcribe_file(self.handle, cpath.as_ptr()) };
        if raw.is_null() {
            return Err(last_error("transcribe_file failed"));
        }
        Ok(TranscriptResult::from_json(cstr_to_string(raw)))
    }

    pub fn transcribe_buffer(&self, pcm: &[i16], sample_rate: f32) -> Result<TranscriptResult> {
        let raw = unsafe {
            vietasr_transcribe_buffer(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if raw.is_null() {
            return Err(last_error("transcribe_buffer failed"));
        }
        Ok(TranscriptResult::from_json(cstr_to_string(raw)))
    }

    pub fn stream(&self, sample_rate: f32) -> Result<Session> {
        let handle = unsafe { vietasr_session_new(self.handle, sample_rate) };
        if handle.is_null() {
            return Err(last_error("session creation failed"));
        }
        Ok(Session { handle })
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { vietasr_pipeline_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// A streaming session. Owns per-stream state (caches, beam, segments).
pub struct Session {
    handle: *mut CSession,
}

unsafe impl Send for Session {}

impl Session {
    pub fn accept_i16(&mut self, pcm: &[i16]) -> bool {
        unsafe {
            vietasr_accept_waveform_s16(self.handle, pcm.as_ptr(), pcm.len() as c_int) == 1
        }
    }

    pub fn accept_f32(&mut self, pcm: &[f32]) -> bool {
        unsafe {
            vietasr_accept_waveform_f32(self.handle, pcm.as_ptr(), pcm.len() as c_int) == 1
        }
    }

    pub fn partial(&self) -> TranscriptResult {
        TranscriptResult::from_json(cstr_to_string(unsafe {
            vietasr_partial_result(self.handle)
        }))
    }

    pub fn result(&self) -> TranscriptResult {
        TranscriptResult::from_json(cstr_to_string(unsafe { vietasr_result(self.handle) }))
    }

    pub fn finalize(&mut self) -> TranscriptResult {
        TranscriptResult::from_json(cstr_to_string(unsafe {
            vietasr_final_result(self.handle)
        }))
    }

    pub fn reset(&mut self) {
        unsafe { vietasr_session_reset(self.handle) };
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { vietasr_session_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

pub fn list_modules() -> Vec<String> {
    parse_json_string_array(&cstr_to_string(unsafe { vietasr_list_modules() }))
}

pub fn list_presets() -> Vec<String> {
    parse_json_string_array(&cstr_to_string(unsafe { vietasr_list_presets() }))
}

pub fn version() -> String {
    cstr_to_string(unsafe { vietasr_version() })
}

pub fn set_log_level(level: i32) {
    unsafe { vietasr_set_log_level(level as c_int) };
}

fn parse_json_string_array(raw: &str) -> Vec<String> {
    let trimmed = raw.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect()
}
