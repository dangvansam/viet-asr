package io.vietasr.demo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import io.vietasr.Pipeline;
import io.vietasr.Result;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity {

    private static final String TAG = "VietASRDemo";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        TextView textView = new TextView(this);
        textView.setTextSize(14f);
        textView.setPadding(32, 80, 32, 32);
        textView.setText("VietASR " + Pipeline.version()
            + "\n\npresets: " + String.join(", ", Pipeline.listPresets())
            + "\n\nTap 'Transcribe' to run on the bundled sample.");

        Button button = new Button(this);
        button.setText("Transcribe");
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new Thread(() -> {
                    try {
                        File wav = copyAsset("sample.wav");
                        Pipeline pipe = Pipeline.preset("transcribe");
                        Result result = pipe.transcribe(wav.getAbsolutePath());
                        pipe.close();
                        runOnUiThread(() -> textView.setText(result.getText()));
                    } catch (Exception exc) {
                        Log.e(TAG, "transcribe failed", exc);
                        runOnUiThread(() -> textView.setText("error: " + exc.getMessage()));
                    }
                }).start();
            }
        });

        android.widget.LinearLayout layout = new android.widget.LinearLayout(this);
        layout.setOrientation(android.widget.LinearLayout.VERTICAL);
        layout.addView(textView);
        layout.addView(button);
        setContentView(layout);
    }

    private File copyAsset(String name) throws Exception {
        File out = new File(getCacheDir(), name);
        if (out.exists() && out.length() > 0) return out;
        try (InputStream in = getAssets().open(name);
             FileOutputStream fos = new FileOutputStream(out)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) fos.write(buf, 0, n);
        }
        return out;
    }
}
