import Foundation

/// The helper's stdout, written in order on one queue. Audio is dropped (and counted) once `limit` audio
/// messages are waiting, so a stalled engine cannot make the helper's memory grow; events are never dropped.
public final class BoundedWriter {
    private let queue = DispatchQueue(label: "local.wled.dj.writer")
    private let sink: (Data) -> Void
    private let limit: Int
    private let lock = NSLock()
    private var pending = 0
    private var droppedCount: UInt32 = 0

    public init(limit: Int, sink: @escaping (Data) -> Void) {
        self.limit = limit
        self.sink = sink
    }

    public var dropped: UInt32 {
        lock.lock(); defer { lock.unlock() }
        return droppedCount
    }

    @discardableResult
    public func writeAudio(_ data: Data) -> Bool {
        lock.lock()
        if pending >= limit {
            droppedCount &+= 1
            lock.unlock()
            return false
        }
        pending += 1
        lock.unlock()
        queue.async {
            self.sink(data)
            self.lock.lock(); self.pending -= 1; self.lock.unlock()
        }
        return true
    }

    public func writeEvent(_ data: Data) {
        queue.async { self.sink(data) }
    }

    /// Wait until everything queued so far has been written.
    public func flush() {
        queue.sync {}
    }
}

/// Open a log for appending, creating it if needed (crash output from earlier runs is kept).
public func appendingHandle(for url: URL) throws -> FileHandle {
    if !FileManager.default.fileExists(atPath: url.path) {
        FileManager.default.createFile(atPath: url.path, contents: nil)
    }
    let handle = try FileHandle(forWritingTo: url)
    try handle.seekToEnd()
    return handle
}

/// Whether a process command line is a WLED DJ engine (so a stale one may be stopped).
public func isWLEDDJEngine(command: String) -> Bool {
    command.contains("-m wled_dj")
}
