#include <jni.h>
#include <string>
#include <vector>

#include "vietasr.h"

namespace {

jstring make_jstring(JNIEnv* env, const char* utf8) {
    if (utf8 == nullptr) return nullptr;
    return env->NewStringUTF(utf8);
}

VietasrPipeline* to_pipeline(jlong handle) {
    return reinterpret_cast<VietasrPipeline*>(handle);
}

VietasrSession* to_session(jlong handle) {
    return reinterpret_cast<VietasrSession*>(handle);
}

}

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_vietasr_Pipeline_nativePipelinePreset(JNIEnv* env, jclass, jstring jname) {
    const char* name = env->GetStringUTFChars(jname, nullptr);
    auto* pipeline = vietasr_pipeline_preset(name);
    env->ReleaseStringUTFChars(jname, name);
    return reinterpret_cast<jlong>(pipeline);
}

JNIEXPORT jlong JNICALL
Java_io_vietasr_Pipeline_nativePipelineNew(JNIEnv*, jclass) {
    return reinterpret_cast<jlong>(vietasr_pipeline_new());
}

JNIEXPORT jint JNICALL
Java_io_vietasr_Pipeline_nativePipelineAddModule(JNIEnv* env, jclass,
                                                 jlong handle,
                                                 jstring jname,
                                                 jstring jconfig) {
    const char* name = env->GetStringUTFChars(jname, nullptr);
    const char* config = env->GetStringUTFChars(jconfig, nullptr);
    auto status = vietasr_pipeline_add_module(to_pipeline(handle), name, config);
    env->ReleaseStringUTFChars(jname, name);
    env->ReleaseStringUTFChars(jconfig, config);
    return static_cast<jint>(status);
}

JNIEXPORT jint JNICALL
Java_io_vietasr_Pipeline_nativePipelineSetBackend(JNIEnv*, jclass,
                                                  jlong handle, jint backend) {
    return static_cast<jint>(vietasr_pipeline_set_backend(
        to_pipeline(handle), static_cast<VietasrBackend>(backend)));
}

JNIEXPORT jint JNICALL
Java_io_vietasr_Pipeline_nativePipelineSetModelDir(JNIEnv* env, jclass,
                                                   jlong handle, jstring jdir) {
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    auto status = vietasr_pipeline_set_model_dir(to_pipeline(handle), dir);
    env->ReleaseStringUTFChars(jdir, dir);
    return static_cast<jint>(status);
}

JNIEXPORT jint JNICALL
Java_io_vietasr_Pipeline_nativePipelineBuild(JNIEnv*, jclass, jlong handle) {
    return static_cast<jint>(vietasr_pipeline_build(to_pipeline(handle)));
}

JNIEXPORT void JNICALL
Java_io_vietasr_Pipeline_nativePipelineFree(JNIEnv*, jclass, jlong handle) {
    vietasr_pipeline_free(to_pipeline(handle));
}

JNIEXPORT jlong JNICALL
Java_io_vietasr_Pipeline_nativeSessionNew(JNIEnv*, jclass,
                                          jlong handle, jfloat sample_rate) {
    return reinterpret_cast<jlong>(
        vietasr_session_new(to_pipeline(handle), sample_rate));
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Pipeline_nativeTranscribeFile(JNIEnv* env, jclass,
                                              jlong handle, jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    const char* raw = vietasr_transcribe_file(to_pipeline(handle), path);
    env->ReleaseStringUTFChars(jpath, path);
    return make_jstring(env, raw);
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Pipeline_nativeTranscribeBuffer(JNIEnv* env, jclass,
                                                jlong handle,
                                                jshortArray jpcm,
                                                jfloat sample_rate) {
    jsize len = env->GetArrayLength(jpcm);
    std::vector<jshort> pcm(static_cast<std::size_t>(len));
    env->GetShortArrayRegion(jpcm, 0, len, pcm.data());
    const char* raw = vietasr_transcribe_buffer(
        to_pipeline(handle),
        reinterpret_cast<const short*>(pcm.data()),
        static_cast<int>(len),
        sample_rate);
    return make_jstring(env, raw);
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Pipeline_nativeListModules(JNIEnv* env, jclass) {
    return make_jstring(env, vietasr_list_modules());
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Pipeline_nativeListPresets(JNIEnv* env, jclass) {
    return make_jstring(env, vietasr_list_presets());
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Pipeline_nativeLastError(JNIEnv* env, jclass) {
    return make_jstring(env, vietasr_last_error());
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Pipeline_nativeVersion(JNIEnv* env, jclass) {
    return make_jstring(env, vietasr_version());
}

JNIEXPORT void JNICALL
Java_io_vietasr_Pipeline_nativeSetLogLevel(JNIEnv*, jclass, jint level) {
    vietasr_set_log_level(static_cast<VietasrLogLevel>(level));
}

JNIEXPORT void JNICALL
Java_io_vietasr_Session_nativeReset(JNIEnv*, jclass, jlong handle) {
    vietasr_session_reset(to_session(handle));
}

JNIEXPORT jint JNICALL
Java_io_vietasr_Session_nativeAcceptS16(JNIEnv* env, jclass,
                                        jlong handle, jshortArray jpcm) {
    jsize len = env->GetArrayLength(jpcm);
    std::vector<jshort> pcm(static_cast<std::size_t>(len));
    env->GetShortArrayRegion(jpcm, 0, len, pcm.data());
    return static_cast<jint>(vietasr_accept_waveform_s16(
        to_session(handle),
        reinterpret_cast<const short*>(pcm.data()),
        static_cast<int>(len)));
}

JNIEXPORT jint JNICALL
Java_io_vietasr_Session_nativeAcceptF32(JNIEnv* env, jclass,
                                        jlong handle, jfloatArray jpcm) {
    jsize len = env->GetArrayLength(jpcm);
    std::vector<jfloat> pcm(static_cast<std::size_t>(len));
    env->GetFloatArrayRegion(jpcm, 0, len, pcm.data());
    return static_cast<jint>(vietasr_accept_waveform_f32(
        to_session(handle), pcm.data(), static_cast<int>(len)));
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Session_nativePartial(JNIEnv* env, jclass, jlong handle) {
    return make_jstring(env, vietasr_partial_result(to_session(handle)));
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Session_nativeResult(JNIEnv* env, jclass, jlong handle) {
    return make_jstring(env, vietasr_result(to_session(handle)));
}

JNIEXPORT jstring JNICALL
Java_io_vietasr_Session_nativeFinal(JNIEnv* env, jclass, jlong handle) {
    return make_jstring(env, vietasr_final_result(to_session(handle)));
}

JNIEXPORT void JNICALL
Java_io_vietasr_Session_nativeFree(JNIEnv*, jclass, jlong handle) {
    vietasr_session_free(to_session(handle));
}

}
