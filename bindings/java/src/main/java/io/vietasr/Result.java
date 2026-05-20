package io.vietasr;

public final class Result {
    private final String rawJson;
    private final String text;
    private final String partial;
    private final boolean isFinal;

    private Result(String rawJson, String text, String partial, boolean isFinal) {
        this.rawJson = rawJson;
        this.text = text == null ? "" : text;
        this.partial = partial == null ? "" : partial;
        this.isFinal = isFinal;
    }

    static Result fromJson(String raw) {
        if (raw == null || raw.isEmpty()) {
            return new Result("{}", "", "", false);
        }
        String text = extractString(raw, "text");
        String partial = extractString(raw, "partial");
        boolean isFinal = "true".equals(extractLiteral(raw, "is_final"));
        return new Result(raw, text, partial, isFinal);
    }

    public String text() {
        return text;
    }

    public String partial() {
        return partial;
    }

    public boolean isFinal() {
        return isFinal;
    }

    public String toJson() {
        return rawJson;
    }

    @Override
    public String toString() {
        return text;
    }

    private static String extractString(String json, String key) {
        String needle = "\"" + key + "\":\"";
        int idx = json.indexOf(needle);
        if (idx < 0) return null;
        int start = idx + needle.length();
        StringBuilder sb = new StringBuilder();
        boolean escape = false;
        for (int i = start; i < json.length(); ++i) {
            char c = json.charAt(i);
            if (escape) {
                sb.append(c);
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static String extractLiteral(String json, String key) {
        String needle = "\"" + key + "\":";
        int idx = json.indexOf(needle);
        if (idx < 0) return "";
        int start = idx + needle.length();
        int end = start;
        while (end < json.length() && Character.isLetterOrDigit(json.charAt(end))) {
            end++;
        }
        return json.substring(start, end);
    }
}
