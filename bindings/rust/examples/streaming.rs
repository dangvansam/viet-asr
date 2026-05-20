use std::fs::File;
use std::io::Read;
use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: streaming <wav-file>");
        exit(1);
    }

    let (pcm, sample_rate) = match read_wav_mono_i16(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("read wav: {e}");
            exit(2);
        }
    };
    let duration_s = pcm.len() as f64 / sample_rate as f64;
    println!("audio: {duration_s:.2}s @ {sample_rate} Hz");

    let pipeline = vietasr::Pipeline::preset("transcribe").expect("preset");
    let mut session = pipeline.stream(sample_rate as f32).expect("stream");

    let chunk = (sample_rate / 1000 * 320) as usize;
    let started = Instant::now();
    let mut last_partial = String::new();

    let mut offset = 0;
    while offset < pcm.len() {
        let end = (offset + chunk).min(pcm.len());
        session.accept_i16(&pcm[offset..end]);
        let partial = session.partial().text();
        if !partial.is_empty() && partial != last_partial {
            let t = end as f64 / sample_rate as f64;
            println!("  [{t:6.2}s] {partial}");
            last_partial = partial;
        }
        offset = end;
    }

    let final_text = session.finalize().text();
    let elapsed = started.elapsed().as_secs_f64();
    println!();
    println!(
        "FINAL ({elapsed:.2}s wall, RTF {:.2}):\n{final_text}",
        elapsed / duration_s
    );
}

fn read_wav_mono_i16(path: &str) -> Result<(Vec<i16>, u32), String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut sample_rate: u32 = 16000;
    let mut channels: u16 = 1;
    let mut bits: u16 = 16;
    let mut offset = 12usize;

    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let body = offset + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
            bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
        } else if id == b"data" {
            if bits != 16 {
                return Err("only 16-bit PCM supported".into());
            }
            let total = size / 2;
            let mut all = Vec::with_capacity(total);
            for i in 0..total {
                all.push(i16::from_le_bytes(
                    bytes[body + i * 2..body + i * 2 + 2].try_into().unwrap(),
                ));
            }
            if channels == 1 {
                return Ok((all, sample_rate));
            }
            let ch = channels as usize;
            let mono: Vec<i16> = all
                .chunks(ch)
                .map(|frame| {
                    let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                    (sum / ch as i32) as i16
                })
                .collect();
            return Ok((mono, sample_rate));
        }
        offset = body + size + (size & 1);
    }
    Err("no data chunk".into())
}
