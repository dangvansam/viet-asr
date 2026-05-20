const native = require("./native");
const { Result } = require("./result");

class Session {
    constructor(handle) {
        this.handle = handle;
    }

    reset() {
        native.fn.vietasr_session_reset(this.handle);
    }

    accept(pcm) {
        if (!this.handle) {
            throw new Error("Session is closed");
        }
        if (pcm instanceof Int16Array) {
            return native.fn.vietasr_accept_waveform_s16(
                this.handle, pcm, pcm.length) === 1;
        }
        if (pcm instanceof Float32Array) {
            return native.fn.vietasr_accept_waveform_f32(
                this.handle, pcm, pcm.length) === 1;
        }
        if (Buffer.isBuffer(pcm)) {
            const view = new Int16Array(pcm.buffer, pcm.byteOffset,
                pcm.byteLength / 2);
            return native.fn.vietasr_accept_waveform_s16(
                this.handle, view, view.length) === 1;
        }
        throw new TypeError(
            "accept() expects Int16Array, Float32Array, or Buffer (16-bit PCM)");
    }

    partial() {
        return Result.fromJson(native.fn.vietasr_partial_result(this.handle));
    }

    result() {
        return Result.fromJson(native.fn.vietasr_result(this.handle));
    }

    final() {
        return Result.fromJson(native.fn.vietasr_final_result(this.handle));
    }

    close() {
        if (this.handle) {
            native.fn.vietasr_session_free(this.handle);
            this.handle = null;
        }
    }
}

module.exports = { Session };
