using System.Text.Json;

namespace Vietasr
{
    public sealed class Result
    {
        public string Text { get; }
        public string Partial { get; }
        public bool IsFinal { get; }
        public string Raw { get; }

        private readonly JsonElement root;

        private Result(string raw, JsonElement root)
        {
            Raw = raw;
            this.root = root;
            Text = GetString("text");
            Partial = GetString("partial");
            IsFinal = root.ValueKind == JsonValueKind.Object
                      && root.TryGetProperty("is_final", out var f)
                      && f.ValueKind == JsonValueKind.True;
        }

        internal static Result FromJson(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                using var empty = JsonDocument.Parse("{}");
                return new Result("{}", empty.RootElement.Clone());
            }
            try
            {
                using var doc = JsonDocument.Parse(raw);
                return new Result(raw, doc.RootElement.Clone());
            }
            catch (JsonException)
            {
                using var empty = JsonDocument.Parse("{}");
                return new Result(raw, empty.RootElement.Clone());
            }
        }

        public JsonElement? Field(string key)
        {
            if (root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty(key, out var value))
            {
                return value;
            }
            return null;
        }

        public string ToJson() => Raw;

        public override string ToString() => Text;

        private string GetString(string key)
        {
            if (root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty(key, out var value)
                && value.ValueKind == JsonValueKind.String)
            {
                return value.GetString() ?? string.Empty;
            }
            return string.Empty;
        }
    }
}
