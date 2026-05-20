#!/usr/bin/env node
const vietasr = require("..");

function main() {
    const argv = process.argv.slice(2);
    let preset = "transcribe";
    let pretty = false;
    let wavPath = null;
    const modules = [];

    for (let i = 0; i < argv.length; ++i) {
        const arg = argv[i];
        if (arg === "--preset" && i + 1 < argv.length) {
            preset = argv[++i];
        } else if (arg === "--module" && i + 1 < argv.length) {
            modules.push(argv[++i]);
        } else if (arg === "--pretty") {
            pretty = true;
        } else if (arg === "-h" || arg === "--help") {
            console.log(`Usage: vietasr [--preset NAME] [--module NAME]... [--pretty] <wav-file>`);
            console.log(`\nAvailable presets: ${JSON.stringify(vietasr.listPresets())}`);
            console.log(`Available modules: ${JSON.stringify(vietasr.listModules())}`);
            process.exit(0);
        } else if (!arg.startsWith("-")) {
            wavPath = arg;
        }
    }

    if (!wavPath) {
        console.error("missing WAV path; pass --help for usage");
        process.exit(1);
    }

    let pipe;
    if (modules.length) {
        pipe = vietasr.Pipeline.new();
        for (const m of modules) pipe.add(m);
        pipe.build();
    } else {
        pipe = vietasr.Pipeline.preset(preset);
    }

    try {
        const result = pipe.transcribe(wavPath);
        if (pretty) {
            console.log(JSON.stringify(result.payload, null, 2));
        } else {
            console.log(result.toJson());
        }
    } finally {
        pipe.close();
    }
}

main();
