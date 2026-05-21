// swift-tools-version:5.7
import PackageDescription

// VietASR — Vietnamese speech recognition for iOS.
//
// The native code (libvietasr + onnxruntime, with the CoreML execution
// provider) ships as a precompiled VietASR.xcframework attached to each GitHub
// release. `checksum` is recomputed and patched in by the release workflow
// (`swift package compute-checksum VietASR.xcframework.zip`) at tag time.
//
// To build from a source checkout: run bindings/ios/build-xcframework.sh, then
// swap the binaryTarget below to `path: "../ios/VietASR.xcframework"`.
let package = Package(
    name: "VietASR",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "VietASR", targets: ["VietASR"]),
    ],
    targets: [
        .binaryTarget(
            name: "CVietASR",
            url: "https://github.com/dangvansam/viet-asr/releases/download/v0.2.0/VietASR.xcframework.zip",
            checksum: "0000000000000000000000000000000000000000000000000000000000000000"
        ),
        .target(
            name: "VietASR",
            dependencies: ["CVietASR"],
            path: "Sources/VietASR"
        ),
    ]
)
