import AppKit
import DJCore

/// Menu-bar app: owns the System Audio Recording permission, runs the engine, opens the studio.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private let statusLine = NSMenuItem(title: "Starting…", action: nil, keyEquivalent: "")
    private let logItem = NSMenuItem(title: "Show Log", action: #selector(showLog), keyEquivalent: "")
    private var engine: EngineProcess?
    private var root: URL!
    private var studio: URL?
    private var openedStudio = false
    private var quitting = false
    private var restarts = RestartPolicy()
    private var pendingLaunch: DispatchWorkItem?
    private var replacedStale = false
    private let fileManager = FileManager.default
    private let logs = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/WLED DJ")
    private let support = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/WLED DJ")

    func applicationDidFinishLaunching(_ notification: Notification) {
        if NSRunningApplication.runningApplications(withBundleIdentifier: "local.wled.dj").count > 1 {
            if let url = runningStudioURL() { NSWorkspace.shared.open(url) }
            NSApp.terminate(nil)
            return
        }
        buildMenu()
        try? fileManager.createDirectory(at: logs, withIntermediateDirectories: true)
        let baked = Bundle.main.object(forInfoDictionaryKey: "WLEDDJProjectRoot") as? String
        guard let found = resolveProjectRoot(bundleURL: Bundle.main.bundleURL, bakedRoot: baked,
                                             exists: { self.fileManager.fileExists(atPath: $0) }) else {
            fail("WLED DJ can't find its project folder", "Keep WLED DJ.app inside its dj folder, or rebuild it with app/build.sh.")
            return
        }
        root = found
        Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.refreshStatus() }
        setUp()
    }

    private var python: URL { root.appendingPathComponent(".venv/bin/python") }

    private func buildMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let image = NSImage(systemSymbolName: "light.beacon.max", accessibilityDescription: "WLED DJ") {
            image.isTemplate = true
            statusItem.button?.image = image
        } else {
            statusItem.button?.title = "WLED DJ"
        }
        let menu = NSMenu()
        statusLine.isEnabled = false
        menu.addItem(statusLine)
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Open Studio", action: #selector(openStudio), keyEquivalent: "o"))
        menu.addItem(NSMenuItem(title: "Restart Engine", action: #selector(restartEngine), keyEquivalent: "r"))
        menu.addItem(logItem)
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit WLED DJ", action: #selector(quit), keyEquivalent: "q"))
        for item in menu.items { item.target = self }
        statusItem.menu = menu
    }

    /// Bring the Python environment in line with uv.lock (quick when nothing changed), then launch.
    private func setUp() {
        let firstRun = !fileManager.fileExists(atPath: python.path)
        statusLine.title = firstRun ? "Setting up… (first run, about a minute)" : "Starting…"
        guard let uv = findUV(home: NSHomeDirectory(), exists: { self.fileManager.fileExists(atPath: $0) }) else {
            if !firstRun { launchEngine(); return }
            fail("WLED DJ needs uv to set itself up",
                 "Install it in Terminal with:\ncurl -LsSf https://astral.sh/uv/install.sh | sh\nthen open WLED DJ again.")
            return
        }
        let setupLog = logs.appendingPathComponent("setup.log")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: uv)
        process.arguments = ["sync", "--frozen"]
        process.currentDirectoryURL = root
        let handle = try? appendingHandle(for: setupLog)
        process.standardOutput = handle
        process.standardError = handle
        process.terminationHandler = { [weak self] finished in
            DispatchQueue.main.async {
                guard let self else { return }
                if finished.terminationStatus == 0 || !firstRun { self.launchEngine() } else {
                    self.fail("WLED DJ setup failed", "See \(setupLog.path) for details.")
                }
            }
        }
        do { try process.run() } catch { fail("WLED DJ setup failed", error.localizedDescription) }
    }

    private func launchEngine() {
        pendingLaunch?.cancel()
        pendingLaunch = nil
        guard engine == nil else { return }
        statusLine.title = "Starting…"
        let capture = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/wled-capture").path
        let quoted = "'" + capture.replacingOccurrences(of: "'", with: "'\\''") + "'"   // parsed with shlex
        let process = EngineProcess(python: python, arguments: ["-m", "wled_dj", "--watch-parent", "--capture", quoted],
                                    root: root, consoleLog: logs.appendingPathComponent("engine-console.log"))
        process.onEvent = { [weak self] event in self?.handle(event) }
        process.onExit = { [weak self] status in self?.engineExited(status) }
        do {
            try process.start()
            engine = process
        } catch {
            fail("WLED DJ could not start its engine", error.localizedDescription)
        }
    }

    private func handle(_ event: [String: Any]) {
        guard let kind = event["event"] as? String else { return }
        if let text = event["url"] as? String, let url = URL(string: text) { studio = url }
        if kind == "ready" && !openedStudio, let studio {
            openedStudio = true
            NSWorkspace.shared.open(studio)
        } else if kind == "already-running", let studio {
            NSWorkspace.shared.open(studio)
        }
    }

    private func engineExited(_ status: Int32) {
        engine = nil
        if quitting || status == 0 { NSApp.terminate(nil); return }
        if status == 3 {                         // an engine left over from a force-quit app, or a dev run
            if !replacedStale, stopStaleEngine() {
                replacedStale = true
                launchEngine()
            } else {
                statusLine.title = "Another program is running a WLED DJ engine"
            }
            return
        }
        if let delay = restarts.next(now: Date()) {
            statusLine.title = "Engine stopped (code \(status)), restarting…"
            let work = DispatchWorkItem { [weak self] in self?.launchEngine() }
            pendingLaunch = work
            DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
        } else {
            statusLine.title = "Engine stopped. Use Show Log, then Restart Engine."
        }
    }

    private func refreshStatus() {
        guard let studio, engine?.isRunning == true else { return }
        let health = studio.appendingPathComponent("api/health")
        URLSession.shared.dataTask(with: health) { [weak self] data, _, _ in
            guard let data, let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let summary = body["summary"] as? String else { return }
            DispatchQueue.main.async { self?.statusLine.title = summary }
        }.resume()
    }

    /// Stop the engine recorded in engine.json if it really is a WLED DJ engine. Returns true if it is gone.
    private func stopStaleEngine() -> Bool {
        guard let data = try? Data(contentsOf: support.appendingPathComponent("engine.json")),
              let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pid = (body["pid"] as? NSNumber)?.int32Value, pid > 0, kill(pid, 0) == 0 else { return false }
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-o", "command=", "-p", String(pid)]
        let pipe = Pipe()
        ps.standardOutput = pipe
        guard (try? ps.run()) != nil else { return false }
        ps.waitUntilExit()
        let command = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        guard isWLEDDJEngine(command: command) else { return false }
        kill(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(6)
        while kill(pid, 0) == 0 && Date() < deadline { Thread.sleep(forTimeInterval: 0.1) }
        if kill(pid, 0) == 0 { kill(pid, SIGKILL); Thread.sleep(forTimeInterval: 0.3) }
        return kill(pid, 0) != 0
    }

    private func runningStudioURL() -> URL? {
        guard let data = try? Data(contentsOf: support.appendingPathComponent("engine.json")),
              let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = body["url"] as? String else { return nil }
        return URL(string: text)
    }

    private func fail(_ title: String, _ detail: String) {
        statusLine.title = title
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = detail
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    @objc private func openStudio() {
        if let studio { NSWorkspace.shared.open(studio) }
    }

    @objc private func restartEngine() {
        restarts = RestartPolicy()
        pendingLaunch?.cancel()
        pendingLaunch = nil
        replacedStale = false
        statusLine.title = "Restarting…"
        let old = engine
        engine = nil
        old?.onExit = nil
        DispatchQueue.global().async { [weak self] in        // stopping can take seconds; keep the menu live
            old?.stop()
            DispatchQueue.main.async { self?.launchEngine() }
        }
    }

    @objc private func showLog() {
        NSWorkspace.shared.open(logs.appendingPathComponent("engine.log"))
    }

    @objc private func quit() {
        quitting = true
        if let engine, engine.isRunning {
            statusLine.title = "Stopping…"
            DispatchQueue.global().async { engine.stop() }   // engineExited terminates the app
        } else {
            NSApp.terminate(nil)
        }
    }

    /// Opening the app again while it runs (Finder, Dock, Spotlight) brings up the studio.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openStudio()
        return false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if let engine, engine.isRunning {
            quitting = true
            engine.onExit = nil
            engine.stop()
        }
        return .terminateNow
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
