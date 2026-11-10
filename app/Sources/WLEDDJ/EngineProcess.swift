import DJCore
import Foundation

/// The Python engine as a child process: JSON lines on stdout, console output kept for crash reports.
final class EngineProcess {
    private let process = Process()
    private var buffer = Data()
    var onEvent: (([String: Any]) -> Void)?
    var onExit: ((Int32) -> Void)?

    init(python: URL, arguments: [String], root: URL, consoleLog: URL) {
        process.executableURL = python
        process.arguments = arguments
        process.currentDirectoryURL = root
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONPATH"] = root.appendingPathComponent("engine").path   // works even if the folder moved
        process.environment = environment
        let output = Pipe()
        process.standardOutput = output
        process.standardError = (try? appendingHandle(for: consoleLog)) ?? FileHandle.nullDevice   // keep old crashes
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard let self, !chunk.isEmpty else { return }
            self.buffer.append(chunk)
            while let newline = self.buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = self.buffer.subdata(in: self.buffer.startIndex..<newline)
                self.buffer.removeSubrange(self.buffer.startIndex...newline)
                if let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any] {
                    DispatchQueue.main.async { self.onEvent?(object) }
                }
            }
        }
        process.terminationHandler = { [weak self] finished in
            output.fileHandleForReading.readabilityHandler = nil
            let status = finished.terminationStatus
            DispatchQueue.main.async { self?.onExit?(status) }
        }
    }

    var isRunning: Bool { process.isRunning }

    func start() throws { try process.run() }

    /// SIGTERM, then SIGKILL after `timeout` seconds. Blocks the caller until the engine is gone.
    func stop(timeout: TimeInterval = 6) {
        guard process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        process.waitUntilExit()
    }
}
