const path = require("path");
const fs = require("fs");

const vietasr = require("..");

function assertEquals(actual, expected, msg) {
    if (actual !== expected) {
        console.error(`FAIL: ${msg}\n  expected: ${expected}\n  actual:   ${actual}`);
        process.exit(1);
    }
    console.log(`  OK  ${msg}`);
}

function main() {
    const fixture = process.argv[2] || path.join(__dirname, "fixtures", "audio.wav");
    if (!fs.existsSync(fixture)) {
        console.error(`fixture missing: ${fixture}`);
        process.exit(2);
    }

    console.log(`vietasr ${vietasr.version()}`);
    console.log(`presets: ${vietasr.listPresets().join(", ")}`);
    console.log(`modules: ${vietasr.listModules().join(", ")}`);
    console.log();

    const pipe = vietasr.Pipeline.preset("transcribe");
    const first = pipe.transcribe(fixture).text;
    const second = pipe.transcribe(fixture).text;
    assertEquals(second, first, "two batch calls on same pipeline are idempotent");
    pipe.close();

    const pipe2 = vietasr.Pipeline.preset("transcribe");
    const batchText = pipe2.transcribe(fixture).text;
    pipe2.close();

    if (!batchText.startsWith("sao lại không liên quan")) {
        console.error(`FAIL: unexpected transcript prefix: ${batchText.slice(0, 80)}`);
        process.exit(3);
    }
    console.log(`  OK  transcript starts with expected Vietnamese: "sao lại không liên quan ..."`);

    console.log();
    console.log("all smoke checks passed");
}

main();
