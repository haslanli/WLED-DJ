import Foundation

/// The project folder holding pyproject.toml: the app's own folder first, else the path baked in at build time.
public func resolveProjectRoot(bundleURL: URL, bakedRoot: String?, exists: (String) -> Bool) -> URL? {
    let parent = bundleURL.deletingLastPathComponent()
    if exists(parent.appendingPathComponent("pyproject.toml").path) { return parent }
    if let bakedRoot, exists(URL(fileURLWithPath: bakedRoot).appendingPathComponent("pyproject.toml").path) {
        return URL(fileURLWithPath: bakedRoot)
    }
    return nil
}

/// uv's usual install locations, in order (apps launched from Finder get no shell PATH).
public func findUV(home: String, exists: (String) -> Bool) -> String? {
    ["\(home)/.local/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv"].first(where: exists)
}

/// Restart a crashed engine after 1, 2 and 4 seconds; give up after three restarts within a minute.
public struct RestartPolicy {
    private var restarts: [Date] = []
    private let delays: [TimeInterval] = [1, 2, 4]
    private let window: TimeInterval = 60

    public init() {}

    public mutating func next(now: Date) -> TimeInterval? {
        restarts.removeAll { now.timeIntervalSince($0) > window }
        guard restarts.count < delays.count else { return nil }
        let delay = delays[restarts.count]
        restarts.append(now)
        return delay
    }
}
