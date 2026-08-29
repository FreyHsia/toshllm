// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import AVFoundation

/// Subtitle appearance, previewed over a real frame of the video with the
/// longest caption, which is the one that used to get cut off.
struct SubtitleStyleSheet: View {
    let sourceURL: URL?
    let cues: [SubtitleCue]

    @EnvironmentObject var loc: Localizer
    @Environment(\.dismiss) private var dismiss

    @State private var style = SubtitleStyle.load()
    @State private var frame: NSImage? = nil
    @State private var rendered: NSImage? = nil

    /// The worst case is what you want to look at, not an average one.
    private var sampleText: String {
        cues.max { $0.text.count < $1.text.count }?.text
            ?? loc.t("Un subtítulo de ejemplo, lo bastante largo como para que tenga que partirse en varias líneas y se vea cómo queda.",
                     "A sample caption, long enough that it has to wrap onto several lines so you can see how it lands.")
    }

    var body: some View {
        VStack(spacing: 12) {
            preview
            Divider()
            controls
            footer
        }
        .padding(16)
        .frame(minWidth: 620, minHeight: 560)
        .task { await loadFrame() }
        .onChange(of: style) { _, _ in render() }
    }

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(.black)
            if let rendered {
                Image(nsImage: rendered)
                    .resizable().interpolation(.high).aspectRatio(contentMode: .fit)
            } else {
                ProgressView()
            }
        }
        .frame(minHeight: 260)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var controls: some View {
        Form {
            Picker(loc.t("Tipografía", "Typeface"), selection: $style.fontName) {
                ForEach(SubtitleStyle.fontChoices, id: \.name) { Text($0.label).tag($0.name) }
            }
            .help(loc.t("Fuentes que trae macOS y se leen bien sobre imagen en movimiento.",
                        "Faces macOS ships that stay readable over moving pictures."))

            LabeledContent(loc.t("Tamaño", "Size")) {
                HStack {
                    Slider(value: $style.relativeSize, in: 0.02...0.08)
                    Text("\(Int(style.relativeSize * 1000))")
                        .font(.system(size: 11, design: .monospaced)).frame(width: 30)
                }
            }
            .help(loc.t("Proporción del alto del fotograma, así se ve igual en 720p que en 4K.",
                        "A fraction of the frame height, so it reads the same at 720p and at 4K."))

            ColorPicker(loc.t("Color del texto", "Text colour"),
                        selection: Binding(get: { style.textColor.color },
                                           set: { style.textColor = .init($0) }))

            Picker(loc.t("Fondo", "Background"), selection: $style.background) {
                Text(loc.t("Caja", "Box")).tag(SubtitleStyle.Background.box)
                Text(loc.t("Contorno", "Outline")).tag(SubtitleStyle.Background.outline)
                Text(loc.t("Ninguno", "None")).tag(SubtitleStyle.Background.none)
            }
            .pickerStyle(.segmented)
            .help(loc.t("La caja se lee siempre; el contorno tapa menos imagen.",
                        "The box always reads; the outline covers less of the picture."))

            if style.background != .none {
                LabeledContent(loc.t("Opacidad", "Opacity")) {
                    Slider(value: $style.backgroundOpacity, in: 0.2...1)
                }
            }

            Picker(loc.t("Posición", "Position"), selection: $style.position) {
                Text(loc.t("Abajo", "Bottom")).tag(SubtitleStyle.Position.bottom)
                Text(loc.t("Arriba", "Top")).tag(SubtitleStyle.Position.top)
            }
            .pickerStyle(.segmented)

            LabeledContent(loc.t("Margen", "Margin")) {
                Slider(value: $style.margin, in: 0.01...0.2)
            }

            LabeledContent(loc.t("Ancho máximo", "Maximum width")) {
                Slider(value: $style.maxWidth, in: 0.5...0.98)
            }
            .help(loc.t("Cuánto del ancho puede ocupar el texto antes de partir de línea.",
                        "How much of the width the text may take before it wraps."))
        }
        .formStyle(.grouped)
    }

    private var footer: some View {
        HStack {
            Button(loc.t("Restaurar", "Reset")) { style = .default }
                .help(loc.t("Vuelve a la apariencia de fábrica.", "Back to the built-in look."))
            Spacer()
            Button(loc.t("Cancelar", "Cancel")) { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button(loc.t("Guardar", "Save")) {
                style.save()
                dismiss()
            }
            .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
        }
    }

    private func loadFrame() async {
        guard let sourceURL else { render(); return }
        let asset = AVURLAsset(url: sourceURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1280, height: 1280)
        // The middle of the clip beats the first frame, which is often black.
        let at = (try? await asset.load(.duration)).map { CMTime(seconds: $0.seconds / 2, preferredTimescale: 600) }
        guard let at, let cg = try? await generator.image(at: at).image else { render(); return }
        frame = NSImage(cgImage: cg, size: .zero)
        render()
    }

    private func render() {
        guard let frame else {
            rendered = SubtitleLayout.preview(text: sampleText, style: style,
                                              over: Self.placeholder)
            return
        }
        rendered = SubtitleLayout.preview(text: sampleText, style: style, over: frame)
    }

    /// Audio-only projects still get to set the look.
    private static let placeholder: NSImage = {
        let size = CGSize(width: 1280, height: 720)
        let img = NSImage(size: size)
        img.lockFocus()
        NSColor(calibratedWhite: 0.22, alpha: 1).setFill()
        NSBezierPath(rect: CGRect(origin: .zero, size: size)).fill()
        img.unlockFocus()
        return img
    }()
}
