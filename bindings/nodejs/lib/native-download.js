"use strict";

// Resolves the libvietasr native library. The npm package ships small; the
// platform-specific native (~67 MB, ONNX model embedded) is downloaded from the
// matching GitHub Release into a per-user cache on install / first use.

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");

const REPO = "dangvansam/viet-asr";

function platformKey() {
    if (process.platform === "win32") return "win-x64";
    if (process.platform === "darwin") return "darwin-universal2";
    if (process.platform === "linux") {
        if (process.arch === "x64") return "linux-x64";
        if (process.arch === "arm64") return "linux-arm64";
    }
    throw new Error(
        `viet-asr: unsupported platform ${process.platform}/${process.arch}`
    );
}

function libFileName() {
    if (process.platform === "win32") return "vietasr.dll";
    if (process.platform === "darwin") return "libvietasr.dylib";
    return "libvietasr.so";
}

function cacheDir(version) {
    const base =
        process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
    return path.join(base, "viet-asr", version);
}

// Downloads + extracts the native bundle into the per-user cache. Synchronous
// (shells out to curl + tar, present on every supported OS) so it works both
// from the postinstall hook and as a lazy fallback inside require().
function ensureNative(version) {
    const dir = cacheDir(version);
    const lib = path.join(dir, libFileName());
    if (fs.existsSync(lib)) return dir;

    const asset = `viet-asr-native-${platformKey()}.tar.gz`;
    fs.mkdirSync(dir, { recursive: true });
    const tarball = path.join(dir, asset);
    const urls = [
        `https://github.com/${REPO}/releases/download/v${version}/${asset}`,
        `https://github.com/${REPO}/releases/latest/download/${asset}`,
    ];

    let lastErr;
    for (const url of urls) {
        try {
            execFileSync("curl", ["-fSL", "--retry", "3", "-o", tarball, url], {
                stdio: "inherit",
            });
            execFileSync("tar", ["-xzf", tarball, "-C", dir], {
                stdio: "inherit",
            });
            fs.rmSync(tarball, { force: true });
            if (fs.existsSync(lib)) return dir;
        } catch (err) {
            lastErr = err;
        }
    }
    fs.rmSync(tarball, { force: true });
    throw new Error(
        `viet-asr: could not download the native library for ${platformKey()}. ` +
            `Set VIETASR_NATIVE_DIR to a directory containing ${libFileName()}. ` +
            `Cause: ${lastErr && lastErr.message}`
    );
}

module.exports = { platformKey, libFileName, cacheDir, ensureNative };
