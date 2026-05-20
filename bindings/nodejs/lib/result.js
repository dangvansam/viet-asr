class Result {
    constructor(payload) {
        this.payload = payload;
    }

    static fromJson(raw) {
        if (!raw) return new Result({});
        try {
            return new Result(JSON.parse(raw));
        } catch (_) {
            return new Result({});
        }
    }

    get text() {
        return this.payload.text || "";
    }

    get partial() {
        return this.payload.partial || "";
    }

    get isFinal() {
        return Boolean(this.payload.is_final);
    }

    get segments() {
        return this.payload.segments || [];
    }

    get speakers() {
        return this.payload.speakers || [];
    }

    field(key) {
        return this.payload[key];
    }

    toJson() {
        return JSON.stringify(this.payload);
    }
}

module.exports = { Result };
