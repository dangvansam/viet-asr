"use strict";

// postinstall hook — pre-fetch the native library so the first require() is
// instant. Failure here is non-fatal: lib/native.js retries lazily at runtime
// (covers offline installs and `npm install --ignore-scripts`).

const { ensureNative } = require("../lib/native-download");
const { version } = require("../package.json");

try {
    ensureNative(version);
    console.log(`viet-asr: native library ready (v${version})`);
} catch (err) {
    console.warn(`viet-asr: native prefetch skipped — ${err.message}`);
}
