import io.vietasr.Pipeline;
import io.vietasr.Session;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public final class Streaming {
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("usage: Streaming <wav-file>");
            System.exit(1);
        }

        WavData wav = readWav(args[0]);
        double durationS = (double) wav.pcm.length / wav.sampleRate;
        System.out.printf("audio: %.2fs @ %d Hz%n", durationS, wav.sampleRate);

        try (Pipeline pipe = Pipeline.preset("transcribe");
             Session session = pipe.stream(wav.sampleRate)) {

            int chunk = wav.sampleRate / 1000 * 320;
            long started = System.nanoTime();
            String lastPartial = "";

            for (int offset = 0; offset < wav.pcm.length; offset += chunk) {
                int end = Math.min(offset + chunk, wav.pcm.length);
                short[] slice = new short[end - offset];
                System.arraycopy(wav.pcm, offset, slice, 0, slice.length);
                session.accept(slice);
                String partial = session.partial().text();
                if (!partial.isEmpty() && !partial.equals(lastPartial)) {
                    System.out.printf("  [%5.2fs] %s%n",
                            (double) end / wav.sampleRate, partial);
                    lastPartial = partial;
                }
            }

            String finalText = session.finalResult().text();
            double elapsed = (System.nanoTime() - started) / 1e9;
            System.out.printf("%nFINAL (%.2fs wall, RTF %.2f):%n%s%n",
                    elapsed, elapsed / durationS, finalText);
        }
    }

    private static final class WavData {
        final short[] pcm;
        final int sampleRate;

        WavData(short[] pcm, int sampleRate) {
            this.pcm = pcm;
            this.sampleRate = sampleRate;
        }
    }

    private static WavData readWav(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(path))) {
            byte[] header = new byte[12];
            in.readFully(header);
            if (!"RIFF".equals(new String(header, 0, 4))
                    || !"WAVE".equals(new String(header, 8, 4))) {
                throw new IOException("not a RIFF/WAVE file");
            }
            int sampleRate = 16000;
            int channels = 1;
            int bits = 16;
            while (true) {
                byte[] chunkHeader = new byte[8];
                in.readFully(chunkHeader);
                String id = new String(chunkHeader, 0, 4);
                int size = le32(chunkHeader, 4);
                byte[] body = new byte[size];
                in.readFully(body);
                if ("fmt ".equals(id)) {
                    channels = le16(body, 2);
                    sampleRate = le32(body, 4);
                    bits = le16(body, 14);
                } else if ("data".equals(id)) {
                    if (bits != 16) {
                        throw new IOException("only 16-bit PCM supported");
                    }
                    int total = size / 2;
                    short[] all = new short[total];
                    for (int i = 0; i < total; ++i) {
                        all[i] = (short) le16(body, i * 2);
                    }
                    if (channels == 1) {
                        return new WavData(all, sampleRate);
                    }
                    short[] mono = new short[total / channels];
                    for (int i = 0; i < mono.length; ++i) {
                        int mixed = 0;
                        for (int c = 0; c < channels; ++c) {
                            mixed += all[i * channels + c];
                        }
                        mono[i] = (short) (mixed / channels);
                    }
                    return new WavData(mono, sampleRate);
                }
            }
        }
    }

    private static int le16(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8);
    }

    private static int le32(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8)
                | ((b[off + 2] & 0xFF) << 16) | ((b[off + 3] & 0xFF) << 24);
    }
}
