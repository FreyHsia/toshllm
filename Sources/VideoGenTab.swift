// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UniformTypeIdentifiers

// Video studio. Same split as the image one (controls in the sidebar, canvas in
// the detail), but the result is a PNG sequence the canvas animates itself:
// macOS ships no VP8 decoder, so playing the engine's own container is not an
// option.

/// Sidebar: model, prompt and the temporal settings.
struct VideoControls: View {
    @ObservedObject var gen: VideoGenerator
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var models: ModelStore

    @AppStorage(SettingsKeys.videoModel) private var modelName = ""
    @AppStorage(SettingsKeys.videoFrames) private var frames = 33
    @AppStorage(SettingsKeys.videoSteps) private var steps = 30
    @AppStorage(SettingsKeys.videoBase) private var base = 480
    @AppStorage(SettingsKeys.videoGPU) private var gpuIndex = -1
    @State private var prompt = ""
    @State private var seed = -1
    @State private var initImage = ""

    private var vram: Double { ImageControls.vram(of: gpuIndex) }

    private var model: VideoGenModel {
        VideoGenCatalog.all.first { $0.name == modelName } ?? VideoGenCatalog.recommended(vramGB: vram)
    }
    private var installed: Bool { VideoGenerator.installed(model, in: models) }

    private var sizes: [Int] { VideoGenLimits.baseSizes(maxLongEdge: model.maxLongEdge) }

    private var vramFraction: Double {
        VideoGenLimits.vramFraction(width: base, height: height, frames: frames,
                                    vramGB: vram, totalGB: model.totalGB)
    }
    private var height: Int { Self.shortEdge(base) }

    /// 16:9 short edge, rounded to the latent grid.
    static func shortEdge(_ base: Int) -> Int {
        max(240, Int((Double(base) * 9 / 16 / 16).rounded()) * 16)
    }

    static func sizeLabel(_ base: Int) -> String { "\(base)x\(shortEdge(base))" }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                modelCard
                promptCard
                settingsCard
                actionRow
            }
            .padding(14)
        }
        .onAppear { if modelName.isEmpty { modelName = VideoGenCatalog.recommended(vramGB: vram).name } }
    }

    private var modelCard: some View {
        Card(title: loc.t("Modelo", "Model"), icon: "film") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("", selection: $modelName) {
                    ForEach(VideoGenCatalog.all) { m in
                        Text(m.name).tag(m.name)
                    }
                }
                .labelsHidden()
                .help(loc.t("Los Wan comparten codificador de texto, así que cambiar entre ellos no vuelve a descargarlo.",
                            "The Wan models share a text encoder, so switching between them does not download it again."))

                Text(model.detail(loc.isSpanish))
                    .font(.caption).foregroundStyle(.secondary)

                if !installed { installBox }
                if !model.fitsVRAMClass(vram) {
                    Label(loc.t("Pide \(Int(model.minVRAMGB)) GB de VRAM y tienes \(Int(vram))",
                                "Wants \(Int(model.minVRAMGB)) GB of VRAM and you have \(Int(vram))"),
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
        }
    }

    /// Same component list and downloader as the image studio: a Wan model is
    /// several files and the encoder is shared, so partial state has to be visible.
    private var installBox: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(model.components) { componentRow($0) }
            Button {
                for comp in model.components where !componentPresent(comp) {
                    models.downloadImageComponent(urlString: comp.urlString, fileName: comp.fileName)
                }
            } label: {
                Label(loc.t("Descargar todo (\(String(format: "%.1f", missingGB)) GB)",
                            "Download all (\(String(format: "%.1f", missingGB)) GB)"),
                      systemImage: "arrow.down.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .help(loc.t("Solo descarga lo que falte: el codificador es común a los Wan.",
                        "Only downloads what is missing: the encoder is shared across the Wan models."))
        }
        .padding(10).frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
    }

    private func componentPresent(_ comp: ImageGenComponent) -> Bool {
        FileManager.default.fileExists(atPath: comp.path(in: models.imagenDirectory).path)
    }

    private var missingGB: Double {
        model.components.filter { !componentPresent($0) }.reduce(0) { $0 + $1.sizeGB }
    }

    private func componentRow(_ comp: ImageGenComponent) -> some View {
        let present = componentPresent(comp)
        return HStack(spacing: 8) {
            Image(systemName: present ? "checkmark.circle.fill" : "circle.dashed")
                .foregroundStyle(present ? .green : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(comp.label(loc.isSpanish)).font(.caption)
                Text(comp.fileName).font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            Spacer()
            if let item = models.imageDownload(fileName: comp.fileName) {
                InlineDownloadProgress(item: item)
            } else if !present {
                Text(String(format: "%.1f GB", comp.sizeGB))
                    .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            }
        }
    }

    private var promptCard: some View {
        Card(title: loc.t("Descripción", "Prompt"), icon: "text.alignleft") {
            VStack(alignment: .leading, spacing: 8) {
                TextEditor(text: $prompt)
                    .font(.system(size: 12))
                    .frame(height: 80)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))

                if model.supportsI2V {
                    HStack {
                        Text(loc.t("Imagen inicial", "Init image")).font(.caption)
                        Spacer()
                        Button(initImage.isEmpty ? loc.t("Elegir…", "Choose…")
                                                 : URL(fileURLWithPath: initImage).lastPathComponent) {
                            pickInitImage()
                        }
                        .font(.caption)
                        if !initImage.isEmpty {
                            Button { initImage = "" } label: { Image(systemName: "xmark.circle.fill") }
                                .buttonStyle(.plain)
                        }
                    }
                    .help(loc.t("Este modelo puede animar una imagen: será el primer fotograma.",
                                "This model can animate an image: it becomes the first frame."))
                }
            }
        }
    }

    private var settingsCard: some View {
        Card(title: loc.t("Ajustes", "Settings"), icon: "slider.horizontal.3") {
            VStack(alignment: .leading, spacing: 10) {
                LabeledContent(loc.t("Tamaño", "Size")) {
                    Picker("", selection: $base) {
                        ForEach(sizes, id: \.self) { b in
                            Text(Self.sizeLabel(b)).tag(b)
                        }
                    }.labelsHidden().frame(width: 130)
                }
                LabeledContent(loc.t("Fotogramas", "Frames")) {
                    Picker("", selection: $frames) {
                        ForEach(VideoGenLimits.frameCounts, id: \.self) {
                            Text("\($0)  ·  \(String(format: "%.1f", VideoGenLimits.seconds(frames: $0)))s").tag($0)
                        }
                    }.labelsHidden().frame(width: 130)
                }
                .help(loc.t("El VAE de Wan comprime el tiempo 4×, así que solo valen cuentas 4n+1.",
                            "Wan's VAE compresses time 4x, so only 4n+1 counts are valid."))

                LabeledContent(loc.t("Pasos", "Steps")) {
                    Stepper("\(steps)", value: $steps, in: 8...60, step: 2).frame(width: 130)
                }
                .help(loc.t("Menos de 20 pasos quema el color con este modelo.",
                            "Below 20 steps this model burns the colors."))

                if vramFraction > 0.9 {
                    Label(loc.t("Estimo ~\(Int(vramFraction * 100))% de la VRAM. Si falla, baja los fotogramas antes que el tamaño.",
                                "I estimate ~\(Int(vramFraction * 100))% of VRAM. If it fails, drop the frame count before the size."),
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }

                if hardware.gpus.count > 1 {
                    LabeledContent(loc.t("GPU", "GPU")) {
                        Picker("", selection: $gpuIndex) {
                            Text(loc.t("Automática", "Automatic")).tag(-1)
                            ForEach(Array(hardware.gpus.enumerated()), id: \.offset) { i, g in
                                Text(g.name).tag(i)
                            }
                        }.labelsHidden().frame(width: 130)
                    }
                }
            }
        }
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            if gen.isBusy {
                ProgressView(value: gen.progress)
                HStack {
                    Text(stageText).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if let eta = gen.etaSeconds {
                        Text(loc.t("~\(eta)s restantes", "~\(eta)s left"))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Button(loc.t("Cancelar", "Cancel")) { gen.cancel() }
            } else {
                Button {
                    gen.generate(model: model, models: models, prompt: prompt,
                                 width: base, height: height, frames: frames,
                                 steps: steps, seed: seed, gpuIndex: gpuIndex,
                                 initImagePath: initImage)
                } label: {
                    Label(loc.t("Generar vídeo", "Generate video"), systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(prompt.isEmpty || !installed || !VideoGenerator.engineInstalled)
            }
            if case .failed(let why) = gen.state, !why.isEmpty {
                Text(failureText(why)).font(.caption).foregroundStyle(.red)
            }
        }
    }

    private var stageText: String {
        switch gen.stage {
        case .loading:  return loc.t("Cargando el modelo…", "Loading the model…")
        case .sampling: return loc.t("Generando \(gen.stepText)", "Sampling \(gen.stepText)")
        case .decoding: return loc.t("Decodificando los fotogramas…", "Decoding the frames…")
        }
    }

    private func failureText(_ why: String) -> String {
        switch why {
        case "OOM":     return loc.t("Sin VRAM. Baja el tamaño o los fotogramas.",
                                     "Out of VRAM. Lower the size or the frame count.")
        case "TIMEOUT": return loc.t("La GPU agotó el tiempo. Baja el tamaño o los fotogramas.",
                                     "The GPU timed out. Lower the size or the frame count.")
        default:        return why
        }
    }

    private func pickInitImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { initImage = url.path }
    }
}

/// Detail: plays the frame sequence and exports it.
struct VideoCanvas: View {
    @ObservedObject var gen: VideoGenerator
    @EnvironmentObject var loc: Localizer

    @State private var playing = true
    @State private var frameIndex = 0
    @State private var exporting = false
    @State private var exportError = ""

    var body: some View {
        VStack(spacing: 12) {
            if gen.frames.isEmpty {
                emptyState
            } else {
                player
                controls
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(16)
    }

    private func exportMP4() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = "toshllm-video.mp4"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let frames = gen.frames
        let fps = gen.lastFPS
        exporting = true
        Task.detached {
            do {
                try VideoExporter.writeMP4(frames: frames, fps: fps, to: url)
                await MainActor.run {
                    exporting = false
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
            } catch {
                await MainActor.run {
                    exporting = false
                    exportError = error.localizedDescription
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "film").font(.system(size: 40)).foregroundStyle(.tertiary)
            Text(loc.t("Escribe una descripción y genera un vídeo.",
                       "Write a prompt and generate a video."))
                .foregroundStyle(.secondary)
            if gen.isBusy {
                Text(loc.t("Los fotogramas aparecen al terminar el decodificado.",
                           "The frames appear once decoding finishes."))
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    /// Animates the PNG sequence at the model's frame rate. The closure re-reads
    /// gen.frames on every tick, and the list can empty between the body's check and
    /// that tick, so the bounds check belongs here: `% 0` is SIGILL, not a trap.
    private var player: some View {
        TimelineView(.periodic(from: .now, by: 1.0 / Double(max(1, gen.lastFPS)))) { ctx in
            let count = gen.frames.count
            if count > 0 {
                let fps = max(1, gen.lastFPS)
                let tick = Int(ctx.date.timeIntervalSinceReferenceDate * Double(fps))
                let idx = playing ? ((tick % count) + count) % count
                                  : min(max(0, frameIndex), count - 1)
                Image(nsImage: gen.frames[idx])
                    .resizable().scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Color.clear
            }
        }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button { playing.toggle() } label: {
                Image(systemName: playing ? "pause.fill" : "play.fill")
            }
            if !playing {
                Slider(value: Binding(get: { Double(frameIndex) },
                                      set: { frameIndex = Int($0) }),
                       in: 0...Double(max(1, gen.frames.count - 1)))
                Text("\(frameIndex + 1)/\(gen.frames.count)")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            } else {
                Text(loc.t("\(gen.frames.count) fotogramas · \(gen.lastDuration)s en generarse",
                           "\(gen.frames.count) frames · \(gen.lastDuration)s to generate"))
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            Button(loc.t("Exportar mp4…", "Export mp4…")) { exportMP4() }
                .disabled(exporting)
                .help(loc.t("Codifica los fotogramas en H.264, que macOS reproduce de serie.",
                            "Encodes the frames as H.264, which macOS plays out of the box."))
            Button(loc.t("Abrir carpeta", "Reveal")) {
                if let first = gen.frameURLs.first {
                    NSWorkspace.shared.activateFileViewerSelecting([first])
                }
            }
            .help(loc.t("Los fotogramas se guardan como PNG sueltos, listos para montarlos donde quieras.",
                        "The frames are saved as individual PNGs, ready to assemble anywhere."))
        }
        .frame(maxWidth: 700)
    }
}
