const path = require("path");
const vietasr = require("..");

function main() {
    const wavPath = process.argv[2];
    if (!wavPath) {
        console.error(`usage: node ${path.basename(__filename)} <wav-file>`);
        console.error("\nAvailable presets:", vietasr.listPresets());
        console.error("Available modules:", vietasr.listModules());
        process.exit(1);
    }

    const pipe = vietasr.Pipeline.preset("transcribe");
    const result = pipe.transcribe(wavPath);
    console.log(result.text);
    pipe.close();
}

main();
