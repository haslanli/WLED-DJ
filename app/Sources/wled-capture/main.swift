import DJCore
import Foundation

// stdout carries the framed protocol; stderr carries human-readable diagnostics.
let stdout = FileHandle.standardOutput
let writer = BoundedWriter(limit: 64) { data in       // about 0.7 s of audio may wait; then audio is dropped
    do { try stdout.write(contentsOf: data) } catch { exit(0) }  // the engine went away
}

/// Tell the engine what went wrong, make sure it was written, then exit.
func fail(_ message: String) -> Never {
    writer.writeEvent(encodeEvent(["event": "error", "message": message]))
    writer.flush()
    exit(1)
}

signal(SIGPIPE, SIG_IGN)
signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
if ProcessInfo.processInfo.environment["WLED_DJ_CAPTURE_SELFTEST"] == "fail" {
    fail("Self-test failure")                    // exercises the failure path without touching audio
}
let capture = Capture(writer: writer, fail: fail)
var sources: [DispatchSourceSignal] = []
for number in [SIGTERM, SIGINT] {
    let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
    source.setEventHandler {
        capture.stop()
        writer.flush()
        exit(0)
    }
    source.resume()
    sources.append(source)
}
let parent = getppid()
let watchdog = DispatchSource.makeTimerSource(queue: .main)
watchdog.schedule(deadline: .now() + 1, repeating: 1)
watchdog.setEventHandler {
    if getppid() != parent { capture.stop(); exit(0) }
}
watchdog.resume()

do {
    writer.writeEvent(encodeEvent(["event": "ready"]))
    try capture.run()
} catch {
    capture.stop()
    fail("\(error)")
}
dispatchMain()
