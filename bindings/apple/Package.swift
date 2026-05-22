// swift-tools-version:5.9
import PackageDescription
import Foundation

// VietASR — Apple (iOS + macOS) Swift binding.
//
// CVietASR.xcframework is large (the ~66 MB ASR model and ONNX Runtime are
// baked in) so it is never committed to git. The manifest resolves it two ways:
//
//   • Local checkout — run `bash build-xcframework.sh`; that populates
//     artifacts/CVietASR.xcframework and the manifest below links it directly.
//   • Released package — a consumer fetching a version tag has no artifacts/
//     directory, so the manifest falls back to the GitHub Release asset. The
//     `url:` / `checksum:` are stamped per release by
//     scripts/stamp-apple-package.py (run from the `release` workflow).

let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let localXCFramework = "artifacts/CVietASR.xcframework"
let hasLocalXCFramework = FileManager.default.fileExists(
    atPath: packageDir.appendingPathComponent(localXCFramework).path)

let cvietasr: Target = hasLocalXCFramework
    ? .binaryTarget(name: "CVietASR", path: localXCFramework)
    : .binaryTarget(
        name: "CVietASR",
        // stamp:url — replaced at release time
        url: "https://github.com/dangvansam/viet-asr/releases/download/v0.1.0/CVietASR.xcframework.zip",
        // stamp:checksum — replaced at release time
        checksum: "0000000000000000000000000000000000000000000000000000000000000000"
    )

let package = Package(
    name: "VietASR",
    // Minimum supported OS — one build runs on these versions and every newer
    // release. iOS 13 / macOS 11 are the floors set by ONNX Runtime 1.20 and
    // the C++ core's use of std::filesystem.
    platforms: [
        .iOS(.v13),
        .macOS(.v11),
    ],
    products: [
        .library(name: "VietASR", targets: ["VietASR"]),
    ],
    targets: [
        // C ABI: CVietASR.framework — a self-contained dynamic framework with
        // libvietasr, the ~66 MB ASR model and ONNX Runtime all baked in,
        // plus vietasr.h and a Clang module map.
        cvietasr,
        // Idiomatic Swift wrapper over the C ABI.
        .target(
            name: "VietASR",
            dependencies: ["CVietASR"],
            path: "Sources/VietASR"
        ),
        .testTarget(
            name: "VietASRTests",
            dependencies: ["VietASR"],
            path: "Tests/VietASRTests"
        ),
    ]
)
