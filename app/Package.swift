// swift-tools-version: 6.0
import PackageDescription
import Foundation

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path

let package = Package(
    name: "WLEDDJ",
    platforms: [.macOS("14.2")],
    targets: [
        .target(name: "DJCore"),
        .executableTarget(
            name: "wled-capture",
            dependencies: ["DJCore"],
            exclude: ["Info.plist"],
            linkerSettings: [
                .linkedFramework("CoreAudio"),
                .linkedFramework("AVFoundation"),
                .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist",
                              "-Xlinker", "\(root)/Sources/wled-capture/Info.plist"]),
            ]
        ),
        .executableTarget(
            name: "WLEDDJ",
            dependencies: ["DJCore"],
            linkerSettings: [.linkedFramework("AppKit")]
        ),
        .testTarget(name: "DJCoreTests", dependencies: ["DJCore"]),
    ],
    swiftLanguageModes: [.v5]
)
