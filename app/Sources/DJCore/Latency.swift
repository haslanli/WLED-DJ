/// Total output latency in milliseconds, as Core Audio reports it for an output device:
/// device latency + safety offset + I/O buffer + stream latency, all in frames at the nominal rate.
public func totalOutputLatencyMs(device: UInt32, safety: UInt32, buffer: UInt32, stream: UInt32, rate: Double) -> Double {
    guard rate > 0 else { return 0 }
    return Double(UInt64(device) + UInt64(safety) + UInt64(buffer) + UInt64(stream)) / rate * 1000
}
