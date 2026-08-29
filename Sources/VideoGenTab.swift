// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UniformTypeIdentifiers

// Video studio controls and animated PNG-sequence preview.

/// Sidebar: model, prompt and the temporal settings.
struct VideoControls: View {
    @ObservedObject var gen: VideoGenerator
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var models: ModelStore

    @AppStorage(SettingsKeys.videoModel) private var modelName = ""
    @AppStorage(SettingsKeys.videoFrames) private var frames = 33
    @AppStorage(SettingsKeys.videoSteps) private var steps = 30
    @AppStorage(SettingsKeys.videoRecipeVersion) private var recipeVersion = 0
    @AppStorage(SettingsKeys.videoSize) private var sizeLabel = ""
    @AppStorage(SettingsKeys.videoGPU) private var gpuIndex = -1
    @AppStorage(SettingsKeys.videoVAETiling) private var vaeTiling = true
    @AppStorage(SettingsKeys.videoNegativePrompt) private var negativePrompt = ""
    @AppStorage(SettingsKeys.videoNegativeSeeded) private var negativeSeeded = false
    @State private var prompt = ""
    @State private var seed = -1
    @State private var initImage = ""

    private var vram: Double { ImageControls.vram(of: gpuIndex) }

    private var model: VideoGenModel {
        VideoGenCatalog.all.first { $0.name == modelName } ?? VideoGenCatalog.recommended(vramGB: vram)
    }
    private var installed: Bool { VideoGenerator.installed(model, in: models) }

    private var sizes: [VideoGenSize] { model.sizes }

    private var size: VideoGenSize {
        model.sizes.first { $0.label == sizeLabel } ?? model.sizes[0]
    }

    private var vramFraction: Double {
        VideoGenLimits.vramFraction(width: size.width, height: size.height, frames: frames,
                                    vramGB: vram, model: model)
    }

    private var nativeNote: String {
        let list = model.sizes.map(\.label).joined(separator: ", ")
        guard model.nativeFrames > 0 else {
            return loc.t("Tamaños del modelo: %@. A %@ fps.", "The model's sizes: %@. At %@ fps.", "\(list)", "\(model.fps)")
        }
        let secs = String(format: "%.1f", VideoGenLimits.seconds(frames: model.nativeFrames, fps: model.fps))
        if frames > model.nativeFrames {
            return loc.t("Por encima de los %@ fotogramas que aprendió el modelo: el movimiento puede irse. Es límite del modelo, no de la app.", "Past the %@ frames the model learned: the motion may drift. That is the model's limit, not the app's.", "\(model.nativeFrames)")
        }
        return loc.t("Tamaños del modelo: %@, %@ fotogramas (%@ s) a %@ fps.", "The model's sizes: %@, %@ frames (%@ s) at %@ fps.", "\(list)", "\(model.nativeFrames)", "\(secs)", "\(model.fps)")
    }

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
        .onAppear {
            if modelName.isEmpty { modelName = VideoGenCatalog.recommended(vramGB: vram).name }
            if !negativeSeeded {
                negativePrompt = model.negativePrompt
                negativeSeeded = true
            }
            // Preserve manual choices after the one-time recipe migration.
            if recipeVersion < 1 {
                steps = model.defaultSteps
                recipeVersion = 1
            }
        }
        .onChange(of: modelName) { _, _ in
            steps = model.defaultSteps
            sizeLabel = ""
            if VideoGenCatalog.isDefaultNegative(negativePrompt) { negativePrompt = model.negativePrompt }
        }
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
                    Label(loc.t("Pide %@ GB de VRAM y tienes %@", "Wants %@ GB of VRAM and you have %@", "\(Int(model.minVRAMGB))", "\(Int(vram))"),
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
                Label(loc.t("Descargar todo (%@ GB)", "Download all (%@ GB)", "\(String(format: "%.1f", missingGB))"),
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

                negativePromptRow

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
                            Button { initImage = "" } label: { Label(loc.t("Quitar", "Clear"), systemImage: "xmark.circle.fill") }
                                .labelStyle(.iconOnly)
                                .buttonStyle(.plain)
                        }
                    }
                    .help(loc.t("Este modelo puede animar una imagen: será el primer fotograma.",
                                "This model can animate an image: it becomes the first frame."))
                }
            }
        }
    }

    private var negativePromptRow: some View {
        let tip = loc.t("Lo que NO debe aparecer. Viene relleno con el negativo recomendado del modelo; vaciarlo suele quemar la imagen.",
                        "What must NOT appear. Prefilled with the model's recommended negative; emptying it usually burns the image out.")
        return DisclosureGroup {
            VStack(alignment: .leading, spacing: 6) {
                TextEditor(text: $negativePrompt)
                    .font(.system(size: 11))
                    .frame(height: 60)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                Button(loc.t("Restaurar el del modelo", "Restore the model's")) {
                    negativePrompt = model.negativePrompt
                }
                .font(.caption).buttonStyle(.borderless)
                .disabled(negativePrompt == model.negativePrompt)
                .help(loc.t("Vuelve a poner el negativo recomendado de %@.",
                            "Puts %@'s recommended negative back.", model.name))
            }
            .padding(.top, 4)
        } label: {
            Text(loc.t("Prompt negativo", "Negative prompt")).font(.caption)
        }
        .help(tip)
    }

    private var settingsCard: some View {
        Card(title: loc.t("Ajustes", "Settings"), icon: "slider.horizontal.3") {
            VStack(alignment: .leading, spacing: 10) {
                LabeledContent(loc.t("Tamaño", "Size")) {
                    Picker("", selection: $sizeLabel) {
                        ForEach(sizes) { s in
                            Text(s.label).tag(s.label)
                        }
                    }.labelsHidden().frame(width: 130)
                }
                .help(loc.t("Solo los tamaños que el modelo declara. Fuera de ellos la imagen sale blanda o el modelo ni acepta la forma.",
                            "Only the sizes the model states. Outside them the picture comes back soft, or the model does not accept the shape at all."))
                LabeledContent(loc.t("Fotogramas", "Frames")) {
                    Picker("", selection: $frames) {
                        ForEach(VideoGenLimits.frameCounts, id: \.self) {
                            Text("\($0)  ·  \(String(format: "%.1f", VideoGenLimits.seconds(frames: $0, fps: model.fps)))s").tag($0)
                        }
                    }.labelsHidden().frame(width: 130)
                }
                .help(loc.t("Los VAE temporales admitidos por el motor requieren cuentas 4n+1.",
                            "The temporal VAEs supported by the engine require 4n+1 frame counts."))

                if let recommended = VideoGenLimits.recommendedVRAMGB(model: model, frames: frames) {
                    Label(loc.t("Este ajuste recomienda %@ GB de VRAM. En 12 GB queda menos de 2 GB libres para el escritorio y otras apps.", "This setting recommends %@ GB of VRAM. On 12 GB, less than 2 GB remains for the desktop and other apps.", "\(Int(recommended))"),
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }

                LabeledContent(loc.t("Pasos", "Steps")) {
                    Stepper("\(steps)", value: $steps, in: 8...60, step: 2).frame(width: 130)
                }
                .help(loc.t("Recomendación para %@: %@ pasos.",
                            "Recommended for %@: %@ steps.", model.name, String(model.defaultSteps)))

                Toggle(loc.t("Decodificar por trozos", "Decode in tiles"), isOn: $vaeTiling)
                    .help(loc.t("Recomendado. Sin esto el decodificado aclara un fotograma de cada cuatro y el clip parpadea, y además pide hasta 16 GB en vez de 3.4. Cuesta un 26% del decodificado.",
                                "Recommended. Without it the decode brightens one frame in every four and the clip flickers, and it asks for up to 16 GB instead of 3.4. It costs 26% of the decode."))

                Text(nativeNote).font(.caption2).foregroundStyle(.secondary)

                if vramFraction > 0.9 && VideoGenLimits.recommendedVRAMGB(model: model, frames: frames) == nil {
                    Label(loc.t("Estimo ~%@% de la VRAM. Si falla, baja los fotogramas antes que el tamaño.", "I estimate ~%@% of VRAM. If it fails, drop the frame count before the size.", "\(Int(vramFraction * 100))"),
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
                        Text(loc.t("~%@s restantes", "~%@s left", "\(eta)"))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Button(loc.t("Cancelar", "Cancel")) { gen.cancel() }
            } else {
                Button {
                    gen.generate(model: model, models: models, prompt: prompt,
                                 negativePrompt: negativePrompt,
                                 width: size.width, height: size.height, frames: frames,
                                 steps: steps, seed: seed, fps: model.fps, gpuIndex: gpuIndex,
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
        case .sampling: return loc.t("Generando %@", "Sampling %@", "\(gen.stepText)")
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
            if !gen.frames.isEmpty {
                player
                controls
            } else if !gen.frameURLs.isEmpty {
                // the frames decode off the main actor, so the run is done before
                // they are ready; without this the canvas reads as "nothing here"
                VStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text(loc.t("Preparando los fotogramas…", "Preparing the frames…"))
                        .foregroundStyle(.secondary)
                }
            } else {
                emptyState
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
        let urls = gen.frameURLs
        let fps = gen.lastFPS
        exporting = true
        Task.detached {
            do {
                let full = urls.compactMap { NSImage(contentsOf: $0) }
                try VideoExporter.writeMP4(frames: full, fps: fps, to: url)
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
            Button {
                playing.toggle()
            } label: {
                Label(playing ? loc.t("Pausar", "Pause") : loc.t("Reproducir", "Play"),
                      systemImage: playing ? "pause.fill" : "play.fill")
            }
                .labelStyle(.iconOnly)
            if !playing {
                Slider(value: Binding(get: { Double(frameIndex) },
                                      set: { frameIndex = Int($0) }),
                       in: 0...Double(max(1, gen.frames.count - 1)))
                Text("\(frameIndex + 1)/\(gen.frames.count)")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            } else {
                Text(loc.t("%@ fotogramas · %@s en generarse", "%@ frames · %@s to generate", "\(gen.frames.count)", "\(gen.lastDuration)"))
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
