import Foundation

/// Messages on the helper's stdout: 1-byte type, UInt32 LE payload length, payload.
/// "A": Float64 host time of the first sample, UInt32 cumulative drops, Float32 LE interleaved stereo.
/// "J": UTF-8 JSON object with an "event" key.
private func frame(_ type: Character, _ payload: Data) -> Data {
    var data = Data(capacity: 5 + payload.count)
    data.append(type.asciiValue!)
    var length = UInt32(payload.count).littleEndian
    withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
    data.append(payload)
    return data
}

public func encodeAudio(hostTime: Double, drops: UInt32, interleaved: [Float]) -> Data {
    var payload = Data(capacity: 12 + interleaved.count * 4)
    var time = hostTime.bitPattern.littleEndian
    var lost = drops.littleEndian
    withUnsafeBytes(of: &time) { payload.append(contentsOf: $0) }
    withUnsafeBytes(of: &lost) { payload.append(contentsOf: $0) }
    interleaved.withUnsafeBytes { payload.append(contentsOf: $0) }
    return frame("A", payload)
}

public func encodeEvent(_ object: [String: Any]) -> Data {
    let payload = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{\"event\":\"error\"}".utf8)
    return frame("J", payload)
}
