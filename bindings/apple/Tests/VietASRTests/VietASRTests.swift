import XCTest
@testable import VietASR

final class VietASRTests: XCTestCase {

    func testVersionIsReported() {
        XCTAssertFalse(VietASR.version.isEmpty)
        XCTAssertNotEqual(VietASR.version, "unknown")
    }

    func testPresetsAndModulesAreListed() {
        let presets = VietASR.listPresets()
        XCTAssertTrue(presets.contains("transcribe"),
                      "expected a 'transcribe' preset, got \(presets)")
        XCTAssertFalse(VietASR.listModules().isEmpty)
    }

    func testPresetPipelineBuilds() throws {
        _ = try Pipeline.preset("transcribe")
    }

    func testUnknownPresetThrows() {
        XCTAssertThrowsError(try Pipeline.preset("no-such-preset"))
    }

    func testBatchTranscribeOfSilence() throws {
        // 1s of 16 kHz silence — exercises the full embedded-model path; an
        // empty transcript is the expected, correct output for silence.
        let pipeline = try Pipeline.preset("transcribe")
        let silence = [Int16](repeating: 0, count: 16_000)
        let result = try pipeline.transcribe(silence, sampleRate: 16_000)
        XCTAssertEqual(result.text, "")
        XCTAssertFalse(result.json.isEmpty)
    }

    func testStreamingSessionOfSilence() throws {
        let pipeline = try Pipeline.preset("transcribe")
        let session = try pipeline.stream(sampleRate: 16_000)
        let chunk = [Int16](repeating: 0, count: 1_600) // 100 ms
        for _ in 0..<10 {
            _ = session.accept(chunk)
        }
        XCTAssertEqual(session.partialResult.text, "")
        XCTAssertEqual(session.finalResult.text, "")
        session.reset()
    }

    func testCustomPipelineBuilds() throws {
        let pipeline = try Pipeline.create()
        try pipeline.add("vad").add("vietasr").build()
        let result = try pipeline.transcribe([Int16](repeating: 0, count: 16_000))
        XCTAssertEqual(result.text, "")
    }
}
