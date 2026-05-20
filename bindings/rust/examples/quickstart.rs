use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: quickstart <wav-file>");
        eprintln!("presets: {:?}", vietasr::list_presets());
        eprintln!("modules: {:?}", vietasr::list_modules());
        exit(1);
    }

    let pipeline = match vietasr::Pipeline::preset("transcribe") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("preset: {e}");
            exit(2);
        }
    };

    match pipeline.transcribe_file(&args[1]) {
        Ok(result) => println!("{}", result.text()),
        Err(e) => {
            eprintln!("transcribe: {e}");
            exit(3);
        }
    }
}
