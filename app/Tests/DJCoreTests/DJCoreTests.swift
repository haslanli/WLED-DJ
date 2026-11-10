import Foundation
import Testing
@testable import DJCore

@Test func audioFrameLayout() {
    let data = encodeAudio(hostTime: 1.5, drops: 2, interleaved: [0.25, -0.25, 1, 0])
    #expect(data.count == 5 + 28)
    #expect(data[0] == UInt8(ascii: "A"))
    #expect(data.subdata(in: 1..<5).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian == 28)
    #expect(data.subdata(in: 5..<13).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } == 1.5.bitPattern.littleEndian)
    #expect(data.subdata(in: 13..<17).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian == 2)
    let floats = data.subdata(in: 17..<33).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    #expect(floats == [0.25, -0.25, 1, 0])
}

@Test func eventFrameRoundTrips() throws {
    let data = encodeEvent(["event": "ready"])
    #expect(data[0] == UInt8(ascii: "J"))
    let length = data.subdata(in: 1..<5).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
    #expect(Int(length) == data.count - 5)
    let object = try JSONSerialization.jsonObject(with: data.subdata(in: 5..<data.count)) as? [String: String]
    #expect(object == ["event": "ready"])
}

@Test func outputLatencyMatchesMeasuredDevices() {
    #expect(abs(totalOutputLatencyMs(device: 7680, safety: 0, buffer: 512, stream: 0, rate: 48000) - 170.67) < 0.01)
    #expect(abs(totalOutputLatencyMs(device: 120, safety: 96, buffer: 512, stream: 1390, rate: 96000) - 22.06) < 0.01)
    #expect(totalOutputLatencyMs(device: 1, safety: 1, buffer: 1, stream: 1, rate: 0) == 0)
}

@Test func projectRootPrefersTheBundlesFolder() {
    let bundle = URL(fileURLWithPath: "/p/dj/WLED DJ.app")
    let root = resolveProjectRoot(bundleURL: bundle, bakedRoot: "/q/dj", exists: { $0 == "/p/dj/pyproject.toml" || $0 == "/q/dj/pyproject.toml" })
    #expect(root?.path == "/p/dj")
}

@Test func projectRootFallsBackToTheBakedPath() {
    let bundle = URL(fileURLWithPath: "/Applications/WLED DJ.app")
    #expect(resolveProjectRoot(bundleURL: bundle, bakedRoot: "/q/dj", exists: { $0 == "/q/dj/pyproject.toml" })?.path == "/q/dj")
    #expect(resolveProjectRoot(bundleURL: bundle, bakedRoot: "/q/dj", exists: { _ in false }) == nil)
    #expect(resolveProjectRoot(bundleURL: bundle, bakedRoot: nil, exists: { _ in false }) == nil)
}

@Test func uvIsFoundInOrder() {
    #expect(findUV(home: "/Users/x", exists: { _ in true }) == "/Users/x/.local/bin/uv")
    #expect(findUV(home: "/Users/x", exists: { $0 != "/Users/x/.local/bin/uv" }) == "/opt/homebrew/bin/uv")
    #expect(findUV(home: "/Users/x", exists: { $0 == "/usr/local/bin/uv" }) == "/usr/local/bin/uv")
    #expect(findUV(home: "/Users/x", exists: { _ in false }) == nil)
}

@Test func restartPolicyBacksOffThenGivesUp() {
    var policy = RestartPolicy()
    let t0 = Date(timeIntervalSince1970: 1000)
    #expect(policy.next(now: t0) == 1)
    #expect(policy.next(now: t0.addingTimeInterval(1)) == 2)
    #expect(policy.next(now: t0.addingTimeInterval(3)) == 4)
    #expect(policy.next(now: t0.addingTimeInterval(7)) == nil)
    #expect(policy.next(now: t0.addingTimeInterval(62)) == 2)
}

@Test func boundedWriterDropsAudioButNeverEvents() {
    let gate = DispatchSemaphore(value: 0)
    let lock = NSLock()
    var written: [UInt8] = []
    let writer = BoundedWriter(limit: 4) { data in
        if data.first == UInt8(ascii: "A") && written.isEmpty { gate.wait() }   // the engine stops reading
        lock.lock(); written.append(data.first!); lock.unlock()
    }
    var accepted = 0
    for _ in 0..<10 { if writer.writeAudio(Data([UInt8(ascii: "A")])) { accepted += 1 } }
    writer.writeEvent(Data([UInt8(ascii: "J")]))
    #expect(accepted == 4)
    #expect(writer.dropped == 6)
    gate.signal()
    writer.flush()
    lock.lock(); defer { lock.unlock() }
    #expect(written == Array(repeating: UInt8(ascii: "A"), count: 4) + [UInt8(ascii: "J")])
}

@Test func consoleLogAppends() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("wled-dj-console-\(UUID().uuidString).log")
    defer { try? FileManager.default.removeItem(at: url) }
    for text in ["first\n", "second\n"] {
        let handle = try appendingHandle(for: url)
        handle.write(Data(text.utf8))
        try handle.close()
    }
    #expect(try String(contentsOf: url, encoding: .utf8) == "first\nsecond\n")
}

@Test func recognisesOnlyAWLEDDJEngine() {
    #expect(isWLEDDJEngine(command: "/x/dj/.venv/bin/python -m wled_dj --capture '/x/WLED DJ.app/Contents/MacOS/wled-capture'"))
    #expect(!isWLEDDJEngine(command: "/usr/bin/python3 server.py"))
    #expect(!isWLEDDJEngine(command: ""))
}
