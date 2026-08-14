import Foundation
import SwiftUI
import Metal
import Combine
import AVFoundation
import ImageIO

/// A text/image-to-video model: the same component machinery as image generation
/// (download, flags, paths) plus the temporal settings sd-cli needs for `vid_gen`.
struct VideoGenModel: Identifiable {
    let name: String
    let detailES: String
    let detailEN: String
    let components: [ImageGenComponent]
    let defaultSteps: Int
    let cfgScale: Double
    /// Flow-matching shift; Wan wants 3.0 at 480p and 5.0 at 720p.
    let flowShift: Double
    let minVRAMGB: Double
    /// Model native long edge. Going past it costs time without adding detail.
    var maxLongEdge: Int = 832
    /// Accepts an init image as the first frame (`-i`).
    var supportsI2V: Bool = false
    var recommendable: Bool = true
    var extraArgs: [String] = []

    var id: String { name }
    var totalGB: Double { components.reduce(0) { $0 + $1.sizeGB } }

    /// What actually stays in VRAM: the text encoder runs on the CPU, so only the
    /// diffusion model and the VAE are resident. Measured on Wan 2.1 1.3B: 2951 MB
    /// against 9615 MB with the encoder in VRAM.
    var vramResidentGB: Double {
        components.filter { $0.kind != .textEncoder && $0.kind != .t5 && $0.kind != .connectors }
            .reduce(0) { $0 + $1.sizeGB }
    }

    func fitsVRAMClass(_ vramGB: Double) -> Bool { vramGB >= minVRAMGB * 0.9 }

    func detail(_ spanish: Bool) -> String { spanish ? detailES : detailEN }
}

enum VideoGenCatalog {
    /// Shared by every Wan model, so switching between them is a free download.
    private static let umt5 = ImageGenComponent(kind: .t5,
        urlString: "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf",
        fileName: "umt5-xxl-encoder-Q4_K_M.gguf", sizeGB: 3.66)

    private static let wan21VAE = ImageGenComponent(kind: .vae,
        urlString: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        fileName: "wan_2.1_vae.safetensors", sizeGB: 0.25)

    private static let wan22VAE = ImageGenComponent(kind: .vae,
        urlString: "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
        fileName: "wan2.2_vae.safetensors", sizeGB: 1.41)

    /// Wan's own negative prompt. It is not optional in practice: without it the
    /// model returns burnt, oversaturated frames that read as a broken app.
    static let defaultNegative =
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        + "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        + "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    static let wan21T2V13B = VideoGenModel(
        name: "Wan 2.1 T2V 1.3B",
        detailES: "1.3B, 480p. El más ligero: entra en 8 GB y es el más rápido de los tres.",
        detailEN: "1.3B, 480p. The lightest: fits 8 GB and the fastest of the three.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
                fileName: "wan2.1_t2v_1.3B_fp16.safetensors", sizeGB: 2.84),
            wan21VAE, umt5,
        ],
        defaultSteps: 30, cfgScale: 6.0, flowShift: 3.0, minVRAMGB: 8, maxLongEdge: 832)

    /// Only safetensors: every Wan GGUF stores patch_embedding.weight with 5 dims
    /// (it is a 3D convolution) and ggml's GGUF reader caps at 4, so the file fails
    /// to load and the run returns noise. Checked on QuantStack, unsloth and the
    /// isfs build that advertises sd.cpp support: all three are 5D.
    static let wan22TI2V5B = VideoGenModel(
        name: "Wan 2.2 TI2V 5B",
        detailES: "5B, 720p. Texto e imagen a vídeo en un solo modelo; pide una tarjeta grande.",
        detailEN: "5B, 720p. Text and image to video in one model; needs a large card.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
                fileName: "wan2.2_ti2v_5B_fp16.safetensors", sizeGB: 10.0),
            wan22VAE, umt5,
        ],
        defaultSteps: 30, cfgScale: 5.0, flowShift: 5.0, minVRAMGB: 24,
        maxLongEdge: 1280, supportsI2V: true,
        // keeps the VAE from allocating its whole working set at once (sd.cpp #868)
        extraArgs: ["--vae-conv-direct"])

    static let wan21I2V14B = VideoGenModel(
        name: "Wan 2.1 I2V 14B",
        detailES: "14B, 480p. Máxima calidad de imagen a vídeo; necesita una tarjeta grande.",
        detailEN: "14B, 480p. Top image-to-video quality; needs a large card.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
                fileName: "wan2.1_i2v_480p_14B_fp8_scaled.safetensors", sizeGB: 16.40),
            wan21VAE, umt5,
        ],
        defaultSteps: 30, cfgScale: 6.0, flowShift: 3.0, minVRAMGB: 32,
        maxLongEdge: 832, supportsI2V: true,
        extraArgs: ["--vae-conv-direct"])

    /// LTX-2.3 distilled (22B). The only one here that also generates audio.
    static let ltx23Distilled = VideoGenModel(
        name: "LTX-2.3 distilled",
        detailES: "22B destilado, 720p y con audio. Para Mac Pro con tarjetas de 32 GB.",
        detailEN: "22B distilled, 720p and with audio. For Mac Pros with 32 GB cards.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf",
                fileName: "ltx-2.3-22b-distilled-Q4_K_M.gguf", sizeGB: 14.33),
            ImageGenComponent(kind: .vae,
                urlString: "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_video_vae.safetensors",
                fileName: "ltx-2.3-22b-distilled_video_vae.safetensors", sizeGB: 1.45),
            ImageGenComponent(kind: .audioVAE,
                urlString: "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
                fileName: "ltx-2.3-22b-distilled_audio_vae.safetensors", sizeGB: 0.36),
            ImageGenComponent(kind: .connectors,
                urlString: "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                fileName: "ltx-2.3-22b-distilled_embeddings_connectors.safetensors", sizeGB: 2.31),
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf",
                fileName: "gemma-3-12b-it-Q4_K_M.gguf", sizeGB: 7.30),
        ],
        defaultSteps: 20, cfgScale: 6.0, flowShift: 3.0, minVRAMGB: 32,
        maxLongEdge: 1280, supportsI2V: true, recommendable: false,
        extraArgs: ["--offload-to-cpu"])

    /// HunyuanVideo 1.5, 720p. Qwen2.5-VL 7B does the conditioning, so it is heavy.
    static let hunyuanVideo15 = VideoGenModel(
        name: "HunyuanVideo 1.5",
        detailES: "720p, movimiento y física muy naturales. Pide 32 GB y 24 GB de descarga.",
        detailEN: "720p, very natural motion and physics. Needs 32 GB and a 24 GB download.",
        components: [
            ImageGenComponent(kind: .diffusion,
                urlString: "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors",
                fileName: "hunyuanvideo1.5_720p_t2v_fp16.safetensors", sizeGB: 16.65),
            ImageGenComponent(kind: .vae,
                urlString: "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
                fileName: "hunyuanvideo15_vae_fp16.safetensors", sizeGB: 2.52),
            ImageGenComponent(kind: .textEncoder,
                urlString: "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
                fileName: "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf", sizeGB: 4.68),
        ],
        defaultSteps: 30, cfgScale: 6.0, flowShift: 5.0, minVRAMGB: 32,
        maxLongEdge: 1280, recommendable: false,
        extraArgs: ["--offload-to-cpu", "--vae-tiling"])

    static let all: [VideoGenModel] = [wan21T2V13B, wan22TI2V5B, wan21I2V14B,
                                       ltx23Distilled, hunyuanVideo15]

    /// Largest model the card can hold, falling back to the lightest.
    static func recommended(vramGB: Double) -> VideoGenModel {
        all.filter { $0.recommendable && $0.fitsVRAMClass(vramGB) }
            .max(by: { $0.minVRAMGB < $1.minVRAMGB }) ?? wan21T2V13B
    }
}

enum VideoGenLimits {
    /// Wan's VAE compresses time 4x, so the frame count has to be 4n+1 or the
    /// last chunk is dropped.
    static let frameCounts = [17, 33, 49, 65, 81]

    static func nearestFrameCount(_ n: Int) -> Int {
        frameCounts.min(by: { abs($0 - n) < abs($1 - n) }) ?? 33
    }

    /// Seconds of output: Wan is trained at 16 fps.
    static func seconds(frames: Int, fps: Int = 16) -> Double {
        Double(frames) / Double(fps)
    }

    /// Peak VRAM (GB): what stays resident plus the activations, which scale with
    /// the latent volume. Calibrated 08-13 on Wan 2.1 1.3B at 480x272: params
    /// measured 2951 MB with the encoder on CPU, and the activation term from the
    /// 11.7 GB total observed at 49 frames.
    static func estVRAMGB(px: Int, frames: Int, residentGB: Double) -> Double {
        residentGB + 0.35e-6 * Double(px) * Double(frames)
    }

    static func vramFraction(width: Int, height: Int, frames: Int,
                             vramGB: Double, residentGB: Double) -> Double {
        estVRAMGB(px: width * height, frames: frames, residentGB: residentGB) / max(0.1, vramGB)
    }

    /// Every size the model is good for. Not filtered by VRAM: the estimate is too
    /// rough to hide the 720p a 720p model exists for, so the UI warns instead.
    static func baseSizes(maxLongEdge: Int) -> [Int] {
        [480, 640, 832, 960, 1280].filter { $0 <= maxLongEdge }
    }
}

/// Drives one text/image-to-video run. Mirrors ImageGenerator, with the temporal
/// settings sd-cli needs and a PNG sequence as the output: macOS has no VP8
/// decoder, so the app animates the frames itself instead of playing a container.
@MainActor
final class VideoGenerator: ObservableObject {
    enum State: Equatable { case idle, generating, done, failed(String) }
    enum Stage { case loading, sampling, decoding }

    @Published var state: State = .idle
    @Published var stage: Stage = .loading
    @Published var progress: Double = 0
    @Published var stepText: String = ""
    /// Decoded frames in order, ready for the player.
    @Published var frames: [NSImage] = []
    @Published var frameURLs: [URL] = []
    @Published var lastDuration: Int = 0

    private(set) var lastPrompt = ""
    private(set) var lastSeed = -1
    private(set) var lastFPS = 16

    private var process: Process?
    private var startedAt: Date?
    private var firstStepAt: Date?
    private var logTail = ""
    private let fileLog = RotatingFileLog(name: "videogen")

    var isBusy: Bool { if case .generating = state { return true }; return false }
    var elapsed: Int { startedAt.map { Int(-$0.timeIntervalSinceNow) } ?? 0 }

    /// The VAE decode is a much bigger share of a video run than of an image one,
    /// so the tail allowance is larger here.
    var etaSeconds: Int? {
        guard let first = firstStepAt, progress > 0.001, progress < 1 else { return nil }
        let spent = -first.timeIntervalSinceNow
        return max(0, Int(spent / progress * (1 - progress) + spent * 0.35))
    }

    static var binary: String { ImageGenerator.binary }
    static var engineInstalled: Bool { ImageGenerator.engineInstalled }

    static func installed(_ model: VideoGenModel, in models: ModelStore) -> Bool {
        model.components.allSatisfy {
            FileManager.default.fileExists(atPath: $0.path(in: models.imagenDirectory).path)
        }
    }

    func generate(model: VideoGenModel, models: ModelStore, prompt: String,
                  width: Int, height: Int, frames frameCount: Int, steps: Int,
                  seed: Int, fps: Int = 16, gpuIndex: Int,
                  initImagePath: String = "") {
        guard !isBusy else { return }
        lastPrompt = prompt; lastSeed = seed; lastFPS = fps
        let dir = models.imagenDirectory

        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH.mm.ss"
        let token = String(UUID().uuidString.prefix(4)).lowercased()
        let runDir = dir.appendingPathComponent("video/toshllm_\(fmt.string(from: Date()))_\(token)",
                                                isDirectory: true)
        try? FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)

        var args: [String] = ["-M", "vid_gen"]
        for comp in model.components {
            args += [comp.flag, comp.path(in: dir).path]
        }
        args += [
            "-p", prompt,
            "-n", VideoGenCatalog.defaultNegative,
            "--cfg-scale", String(format: "%.1f", model.cfgScale),
            "--flow-shift", String(format: "%.1f", model.flowShift),
            "--sampling-method", "euler",
            "--steps", String(steps),
            "-W", String(width), "-H", String(height),
            "--video-frames", String(frameCount),
            "--fps", String(fps),
            "--seed", String(seed),
            // measured 6% faster sampling on RDNA2 with byte-identical output
            "--diffusion-fa",
            // umt5 is 6.66 GB and does nothing after the first 20 s; in RAM it costs
            // 12 s of encode once and frees the VRAM that caps resolution
            "--params-backend", "te=cpu",
            // a PNG sequence, not a container: the app animates the frames
            "-o", runDir.appendingPathComponent("frame_%03d.png").path,
        ]
        if !initImagePath.isEmpty && model.supportsI2V {
            args += ["-i", initImagePath]
        }
        args += model.extraArgs

        var env = ProcessInfo.processInfo.environment
        // The wide matmul tile is tuned for LLM prefill shapes; on diffusion it costs
        // 1.7% on SDXL and 3% on Wan, with byte-identical output. Measured 08-14.
        env["TOSH_MM_WIDE_DISABLE"] = "1"
        let devices = MTLCopyAllDevices()
        if gpuIndex >= 0 && devices.count > 1 {
            env["GGML_METAL_DEVICE_INDEX"] = String(gpuIndex)
        }
        let external = gpuIndex >= 0 && gpuIndex < devices.count && devices[gpuIndex].location == .external
        if UserDefaults.standard.bool(forKey: SettingsKeys.forcePrivateBuffers) || external {
            env["GGML_METAL_SHARED_BUFFERS_DISABLE"] = "1"
        }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: Self.binary)
        p.arguments = args
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.consume(text) }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in self?.finish(status: proc.terminationStatus, runDir: runDir) }
        }

        fileLog.startSession()
        fileLog.append("$ sd-cli " + args.joined(separator: " ") + "\n\n")

        frames = []
        frameURLs = []
        progress = 0
        stepText = ""
        stage = .loading
        state = .generating
        startedAt = Date()
        firstStepAt = nil
        do {
            try p.run()
            process = p
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func cancel() { process?.terminate() }

    /// Playback copy: full frames are only needed for the mp4, which reads the PNGs.
    nonisolated static func displayFrame(_ url: URL) -> NSImage? {
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 960,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
        else { return NSImage(contentsOf: url) }
        return NSImage(cgImage: cg, size: .zero)
    }

    private func consume(_ text: String) {
        logTail = String((logTail + text).suffix(4000))
        fileLog.append(text)
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = String(raw)
            if line.lowercased().contains("decod") { stage = .decoding }
            guard line.contains("s/it"),
                  let r = line.range(of: #"(\d+)/(\d+)"#, options: .regularExpression) else { continue }
            let parts = line[r].split(separator: "/")
            if parts.count == 2, let a = Int(parts[0]), let b = Int(parts[1]), b > 0 {
                if firstStepAt == nil { firstStepAt = Date() }
                stage = .sampling
                progress = Double(a) / Double(b)
                stepText = "\(a)/\(b)"
            }
        }
    }

    private func finish(status: Int32, runDir: URL) {
        process = nil
        let urls = ((try? FileManager.default.contentsOfDirectory(at: runDir,
                                                                  includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard status == 0, !urls.isEmpty else {
            if status == 15 || status == 2 { state = .failed(""); return }
            if logTail.contains("failed to allocate") { state = .failed("OOM"); return }
            let timedOut = logTail.contains("Timeout") || logTail.contains("status 5")
            state = .failed(timedOut ? "TIMEOUT" : "exit \(status)")
            return
        }
        frameURLs = urls
        progress = 1
        lastDuration = elapsed
        state = .done
        // Off the main actor, and downsampled for playback: 81 frames at 720p held
        // at full size are ~300 MB of NSImage for a view a few hundred points wide.
        // The PNGs stay on disk as the source of truth, so export is unaffected.
        Task.detached(priority: .userInitiated) {
            let decoded = urls.compactMap { Self.displayFrame($0) }
            await MainActor.run { self.frames = decoded }
        }
    }
}

/// Writes the frame sequence as an H.264 mp4. AVFoundation can encode it natively,
/// which is the whole reason the engine's own .webm is not what we hand the user.
enum VideoExporter {
    enum ExportError: Error { case noFrames, writerFailed(String) }

    static func writeMP4(frames: [NSImage], fps: Int, to url: URL) throws {
        guard let first = frames.first else { throw ExportError.noFrames }
        // H.264 needs even dimensions.
        let w = Int(first.size.width) & ~1
        let h = Int(first.size.height) & ~1
        guard w > 0, h > 0 else { throw ExportError.noFrames }

        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for (i, image) in frames.enumerated() {
            guard let buffer = pixelBuffer(from: image, width: w, height: h) else { continue }
            // the input drains asynchronously; spin instead of dropping frames
            while !input.isReadyForMoreMediaData { usleep(2000) }
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i),
                                                                timescale: CMTimeScale(max(1, fps))))
        }
        input.markAsFinished()

        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        if writer.status != .completed {
            throw ExportError.writerFailed(writer.error?.localizedDescription ?? "unknown")
        }
    }

    private static func pixelBuffer(from image: NSImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue),
              let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
