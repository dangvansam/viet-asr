using System;
using System.Diagnostics;
using System.IO;
using Vietasr;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: Quickstart [--stream] <wav-file>");
            Console.Error.WriteLine("presets: " + string.Join(", ", Pipeline.ListPresets()));
            Console.Error.WriteLine("modules: " + string.Join(", ", Pipeline.ListModules()));
            return 1;
        }

        bool stream = args[0] == "--stream";
        string wavPath = stream ? args[1] : args[0];

        return stream ? RunStreaming(wavPath) : RunBatch(wavPath);
    }

    private static int RunBatch(string wavPath)
    {
        using var pipe = Pipeline.Preset("transcribe");
        Console.WriteLine(pipe.Transcribe(wavPath).Text);
        return 0;
    }

    private static int RunStreaming(string wavPath)
    {
        (short[] pcm, int sampleRate) = ReadWav(wavPath);
        double durationS = (double)pcm.Length / sampleRate;
        Console.WriteLine($"audio: {durationS:F2}s @ {sampleRate} Hz");

        using var pipe = Pipeline.Preset("transcribe");
        using var session = pipe.Stream(sampleRate);

        int chunk = sampleRate / 1000 * 320;
        var sw = Stopwatch.StartNew();
        string lastPartial = string.Empty;

        for (int offset = 0; offset < pcm.Length; offset += chunk)
        {
            int len = Math.Min(chunk, pcm.Length - offset);
            var slice = new short[len];
            Array.Copy(pcm, offset, slice, 0, len);
            session.Accept(slice);
            string partial = session.Partial().Text;
            if (partial.Length > 0 && partial != lastPartial)
            {
                Console.WriteLine($"  [{(double)(offset + len) / sampleRate,6:F2}s] {partial}");
                lastPartial = partial;
            }
        }

        string finalText = session.Final().Text;
        sw.Stop();
        Console.WriteLine();
        Console.WriteLine($"FINAL ({sw.Elapsed.TotalSeconds:F2}s wall, " +
                          $"RTF {sw.Elapsed.TotalSeconds / durationS:F2}):");
        Console.WriteLine(finalText);
        return 0;
    }

    private static (short[], int) ReadWav(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < 12
            || System.Text.Encoding.ASCII.GetString(bytes, 0, 4) != "RIFF"
            || System.Text.Encoding.ASCII.GetString(bytes, 8, 4) != "WAVE")
        {
            throw new InvalidDataException("not a RIFF/WAVE file");
        }

        int sampleRate = 16000;
        int channels = 1;
        int bits = 16;
        int offset = 12;

        while (offset + 8 <= bytes.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(bytes, offset, 4);
            int size = BitConverter.ToInt32(bytes, offset + 4);
            int body = offset + 8;
            if (id == "fmt ")
            {
                channels = BitConverter.ToInt16(bytes, body + 2);
                sampleRate = BitConverter.ToInt32(bytes, body + 4);
                bits = BitConverter.ToInt16(bytes, body + 14);
            }
            else if (id == "data")
            {
                if (bits != 16)
                {
                    throw new InvalidDataException("only 16-bit PCM supported");
                }
                int total = size / 2;
                var all = new short[total];
                for (int i = 0; i < total; ++i)
                {
                    all[i] = BitConverter.ToInt16(bytes, body + i * 2);
                }
                if (channels == 1)
                {
                    return (all, sampleRate);
                }
                var mono = new short[total / channels];
                for (int i = 0; i < mono.Length; ++i)
                {
                    int mixed = 0;
                    for (int c = 0; c < channels; ++c)
                    {
                        mixed += all[i * channels + c];
                    }
                    mono[i] = (short)(mixed / channels);
                }
                return (mono, sampleRate);
            }
            offset = body + size + (size & 1);
        }
        throw new InvalidDataException("no data chunk");
    }
}
