import io.vietasr.Pipeline;
import io.vietasr.Result;

public final class Quickstart {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("usage: Quickstart <wav-file>");
            System.err.println("presets: " + String.join(", ", Pipeline.listPresets()));
            System.err.println("modules: " + String.join(", ", Pipeline.listModules()));
            System.exit(1);
        }
        try (Pipeline pipe = Pipeline.preset("transcribe")) {
            Result result = pipe.transcribe(args[0]);
            System.out.println(result.text());
        }
    }
}
