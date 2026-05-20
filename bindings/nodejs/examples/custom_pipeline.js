const path = require("path");
const vietasr = require("..");

function main() {
    const wavPath = process.argv[2];
    if (!wavPath) {
        console.error(`usage: node ${path.basename(__filename)} <wav-file>`);
        process.exit(1);
    }

    const pipe = vietasr.Pipeline.new()
        .add("vad")
        .add("vietasr")
        .add("gender")
        .add("emotion")
        .add("dialect")
        .add("noise")
        .build();

    const result = pipe.transcribe(wavPath);
    console.log("text:    ", result.text.slice(0, 80) + "...");
    console.log("gender:  ", JSON.stringify(result.field("gender")));
    console.log("emotion: ", JSON.stringify(result.field("emotion")));
    console.log("dialect: ", JSON.stringify(result.field("dialect")));
    console.log("noise:   ", JSON.stringify(result.field("noise")));
    pipe.close();
}

main();
