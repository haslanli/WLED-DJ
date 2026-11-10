import AVFoundation
import CoreAudio
import DJCore
import Foundation

struct CaptureError: Error, CustomStringConvertible {
    let description: String
}

func check(_ status: OSStatus, _ operation: String) throws {
    if status != noErr { throw CaptureError(description: "\(operation) failed (Core Audio \(status))") }
}

private func address(_ selector: AudioObjectPropertySelector,
                     _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

private func read<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                     _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal, _ value: inout T) -> Bool {
    var where_ = address(selector, scope)
    var size = UInt32(MemoryLayout<T>.size)
    return AudioObjectGetPropertyData(object, &where_, 0, nil, &size, &value) == noErr
}

/// What the engine needs to know about the device the music is heard on.
struct OutputDevice: Equatable {
    let id: AudioObjectID
    let name: String
    let uid: String
    let transport: String
    let latencyMs: Double

    static func current() -> OutputDevice? {
        var id = AudioObjectID(0)
        guard read(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, &id), id != 0 else {
            return nil
        }
        var name = "" as CFString, uid = "" as CFString
        _ = read(id, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, &name)
        _ = read(id, kAudioDevicePropertyDeviceUID, kAudioObjectPropertyScopeGlobal, &uid)
        var transport: UInt32 = 0, latency: UInt32 = 0, safety: UInt32 = 0, buffer: UInt32 = 0, rate: Float64 = 0
        _ = read(id, kAudioDevicePropertyTransportType, kAudioObjectPropertyScopeGlobal, &transport)
        _ = read(id, kAudioDevicePropertyLatency, kAudioObjectPropertyScopeOutput, &latency)
        _ = read(id, kAudioDevicePropertySafetyOffset, kAudioObjectPropertyScopeOutput, &safety)
        _ = read(id, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal, &buffer)
        _ = read(id, kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, &rate)
        var streamLatency: UInt32 = 0
        var streams = address(kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput)
        var size: UInt32 = 0
        if AudioObjectGetPropertyDataSize(id, &streams, 0, nil, &size) == noErr, size >= 4 {
            var ids = [AudioStreamID](repeating: 0, count: Int(size) / 4)
            if AudioObjectGetPropertyData(id, &streams, 0, nil, &size, &ids) == noErr, let first = ids.first {
                _ = read(first, kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal, &streamLatency)
            }
        }
        let code = String(bytes: [24, 16, 8, 0].map { UInt8((transport >> $0) & 0xff) }, encoding: .ascii) ?? "?"
        return OutputDevice(id: id, name: name as String, uid: uid as String, transport: code,
                            latencyMs: totalOutputLatencyMs(device: latency, safety: safety, buffer: buffer,
                                                            stream: streamLatency, rate: rate))
    }

    var event: [String: Any] {
        ["event": "output", "name": name, "uid": uid, "transport": transport, "latency_ms": latencyMs]
    }
}

/// Global, private, non-muting process tap on everything the Mac plays, converted to 48 kHz stereo float.
/// Rebuilt when the default output device or the tap's format changes.
final class Capture {
    private let writer: BoundedWriter
    private let fail: (String) -> Never
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: false)!
    private let converterQueue = DispatchQueue(label: "local.wled.dj.capture.convert")
    private let slots = DispatchSemaphore(value: Capture.slotCount)
    private var tap = AudioObjectID(0)
    private var device = AudioObjectID(0)
    private var io: AudioDeviceIOProcID?
    private var converter: AVAudioConverter?
    private var dropped: UInt32 = 0
    /// Buffers for the real-time callback, made before audio starts: the callback must not allocate.
    /// As many as there are slots, reused in order (the converter queue releases them in the same order).
    private static let slotCount = 4
    private static let maxFrames: AVAudioFrameCount = 8192
    private var ring: [AVAudioPCMBuffer] = []
    private var ringIndex = 0
    private var output: OutputDevice?
    private var pendingRebuild: DispatchWorkItem?
    private var formatListener: AudioObjectPropertyListenerBlock?
    private var formatAddress = address(kAudioTapPropertyFormat)
    private var defaultAddress = address(kAudioHardwarePropertyDefaultOutputDevice)

    init(writer: BoundedWriter, fail: @escaping (String) -> Never) {
        self.writer = writer
        self.fail = fail
    }

    /// Called once on the main queue.
    func run() throws {
        try start()
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.scheduleRebuild() }
        try check(AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &defaultAddress, .main, listener),
                  "Watch default output")
        Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.reportOutput(force: false) }
    }

    private func scheduleRebuild() {
        pendingRebuild?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.stop()
            do { try self.start() } catch { self.fail("\(error)") }
        }
        pendingRebuild = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3, execute: work)
    }

    private func reportOutput(force: Bool) {
        guard let now = OutputDevice.current() else { return }
        if force || now.id != output?.id || abs(now.latencyMs - (output?.latencyMs ?? -100)) > 1 {
            output = now
            writer.writeEvent(encodeEvent(now.event))
        }
    }

    private func start() throws {
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.name = "WLED DJ System Audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(description, &tap), "Create system audio tap")
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(AudioObjectGetPropertyData(tap, &formatAddress, 0, nil, &size, &asbd), "Read tap format")
        guard asbd.mChannelsPerFrame == 2, asbd.mBytesPerFrame > 0,
              let source = AVAudioFormat(streamDescription: &asbd),
              let convert = AVAudioConverter(from: source, to: target) else {
            throw CaptureError(description: "Cannot convert the tap format to 48 kHz stereo")
        }
        converter = convert
        ring = (0..<Capture.slotCount).compactMap { _ in AVAudioPCMBuffer(pcmFormat: source, frameCapacity: Capture.maxFrames) }
        ringIndex = 0
        guard ring.count == Capture.slotCount else { throw CaptureError(description: "Cannot allocate audio buffers") }
        let ring = self.ring
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.scheduleRebuild() }
        formatListener = listener
        try check(AudioObjectAddPropertyListenerBlock(tap, &formatAddress, .main, listener), "Watch tap format")
        let settings: [String: Any] = [
            kAudioAggregateDeviceNameKey: "WLED DJ Private Capture",
            kAudioAggregateDeviceUIDKey: "local.wled.dj.capture.\(UUID().uuidString)",
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: description.uuid.uuidString,
                                               kAudioSubTapDriftCompensationKey: true]],
        ]
        try check(AudioHardwareCreateAggregateDevice(settings as CFDictionary, &device), "Create private tap device")
        let bytesPerFrame = asbd.mBytesPerFrame
        try check(AudioDeviceCreateIOProcIDWithBlock(&io, device, nil) { [weak self] _, input, inputTime, _, _ in
            guard let self else { return }
            let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
            guard let first = buffers.first, first.mDataByteSize > 0 else { return }
            guard self.slots.wait(timeout: .now()) == .success else { self.dropped &+= 1; return }
            let frames = first.mDataByteSize / bytesPerFrame
            guard frames <= Capture.maxFrames else { self.dropped &+= 1; self.slots.signal(); return }
            let copy = ring[self.ringIndex % ring.count]
            self.ringIndex &+= 1
            copy.frameLength = frames
            let destination = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
            for index in 0..<min(buffers.count, destination.count) {
                if let from = buffers[index].mData, let to = destination[index].mData {
                    memcpy(to, from, Int(min(buffers[index].mDataByteSize, destination[index].mDataByteSize)))
                }
            }
            let hostTime = inputTime.pointee.mHostTime != 0
                ? AVAudioTime.seconds(forHostTime: inputTime.pointee.mHostTime)
                : ProcessInfo.processInfo.systemUptime
            let losses = self.dropped
            self.converterQueue.async {
                defer { self.slots.signal() }
                self.convertAndEmit(copy, sourceRate: source.sampleRate, hostTime: hostTime, drops: losses)
            }
        }, "Create tap callback")
        try check(AudioDeviceStart(device, io), "Start tap")
        reportOutput(force: true)
    }

    private func convertAndEmit(_ input: AVAudioPCMBuffer, sourceRate: Double, hostTime: Double, drops: UInt32) {
        guard let converter else { return }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * 48000 / sourceRate) + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, outStatus in
            if supplied { outStatus.pointee = .noDataNow; return nil }
            supplied = true
            outStatus.pointee = .haveData
            return input
        }
        if status == .error {
            fail("Audio conversion failed: \(error?.localizedDescription ?? "unknown")")
        }
        let count = Int(out.frameLength)
        guard count > 0, let planes = out.floatChannelData else { return }
        var interleaved = [Float](repeating: 0, count: count * 2)
        for i in 0..<count {
            interleaved[2 * i] = planes[0][i]
            interleaved[2 * i + 1] = planes[1][i]
        }
        writer.writeAudio(encodeAudio(hostTime: hostTime, drops: drops &+ writer.dropped, interleaved: interleaved))
    }

    func stop() {
        if device != 0, let io {
            AudioDeviceStop(device, io)
            AudioDeviceDestroyIOProcID(device, io)
        }
        io = nil
        converterQueue.sync {}
        if device != 0 { AudioHardwareDestroyAggregateDevice(device); device = 0 }
        if tap != 0 {
            if let formatListener { AudioObjectRemovePropertyListenerBlock(tap, &formatAddress, .main, formatListener) }
            formatListener = nil
            AudioHardwareDestroyProcessTap(tap)
            tap = 0
        }
        converter = nil
        ring = []
    }
}
