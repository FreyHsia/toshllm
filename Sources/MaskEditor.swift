// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import AppKit

/// One painted stroke, in coordinates normalised to the image (0...1) so the
/// preview and the rasterised PNG agree at any view size.
struct MaskStroke: Equatable {
    var points: [CGPoint]
    /// Fraction of the image width, so the brush keeps its size when zoomed.
    var radius: Double
    var erases: Bool
}

/// Paints the inpainting mask over the init image: white repaints, black keeps.
/// Writes a PNG the engine takes with `--mask`.
struct MaskEditorView: View {
    let initImagePath: String
    let outputDirectory: URL
    @Binding var maskPath: String

    @EnvironmentObject var loc: Localizer
    @Environment(\.dismiss) private var dismiss

    @State private var strokes: [MaskStroke] = []
    @State private var radius = 0.06
    @State private var erasing = false
    @State private var showMaskOnly = false
    @State private var error = ""
    @State private var image: NSImage? = nil
    @State private var pixelSize = CGSize(width: 512, height: 512)

    var body: some View {
        VStack(spacing: 12) {
            header
            canvas
            if !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(.red)
            }
            controls
        }
        .padding(16)
        .frame(minWidth: 560, minHeight: 560)
        .task(id: initImagePath) { load() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(loc.t("Pinta la zona a retocar", "Paint the area to repaint"))
                .font(.headline)
            Text(loc.t("Lo pintado se vuelve a generar; el resto de la foto se conserva.",
                       "What you paint is regenerated; the rest of the photo is kept."))
                .font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var canvas: some View {
        MaskPaintSurface(image: image, pixelSize: pixelSize, strokes: $strokes,
                         radius: radius, erasing: erasing, maskOnly: showMaskOnly)
            .frame(minHeight: 380)
    }

    private var controls: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Picker("", selection: $erasing) {
                    Label(loc.t("Pincel", "Brush"), systemImage: "paintbrush.pointed").tag(false)
                    Label(loc.t("Borrador", "Eraser"), systemImage: "eraser").tag(true)
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 200)
                .help(loc.t("El pincel marca lo que se repinta; el borrador devuelve la zona a conservada.",
                            "The brush marks what gets repainted; the eraser puts an area back to kept."))

                Text(loc.t("Grosor", "Size")).font(.caption)
                Slider(value: $radius, in: 0.01...0.25).frame(width: 130)
                    .help(loc.t("Diámetro del pincel, relativo al ancho de la imagen.",
                                "Brush diameter, relative to the image width."))

                Toggle(loc.t("Ver máscara", "View mask"), isOn: $showMaskOnly)
                    .toggleStyle(.switch).font(.caption)
                    .help(loc.t("Muestra la máscara en blanco y negro, como la recibe el motor.",
                                "Shows the mask in black and white, the way the engine gets it."))
            }

            HStack(spacing: 10) {
                Button(loc.t("Deshacer", "Undo"), systemImage: "arrow.uturn.backward") {
                    if !strokes.isEmpty { strokes.removeLast() }
                }
                .disabled(strokes.isEmpty)
                .help(loc.t("Quita el último trazo.", "Removes the last stroke."))

                Button(loc.t("Limpiar", "Clear"), systemImage: "trash") { strokes.removeAll() }
                    .disabled(strokes.isEmpty)
                    .help(loc.t("Borra todos los trazos.", "Removes every stroke."))

                Spacer()

                Button(loc.t("Cancelar", "Cancel")) { dismiss() }
                    .keyboardShortcut(.cancelAction)

                Button(loc.t("Usar máscara", "Use mask")) { save() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(strokes.isEmpty)
                    .help(loc.t("Guarda la máscara y la aplica a esta instancia.",
                                "Saves the mask and applies it to this instance."))
            }
        }
    }

    // MARK: geometry

    /// Where the image lands inside the view once fitted, so painting and the
    /// rasterised PNG use the same frame.
    static func fitRect(_ pixels: CGSize, in bounds: CGSize) -> CGRect {
        guard pixels.width > 0, pixels.height > 0, bounds.width > 0, bounds.height > 0 else {
            return CGRect(origin: .zero, size: bounds)
        }
        let scale = min(bounds.width / pixels.width, bounds.height / pixels.height)
        let size = CGSize(width: pixels.width * scale, height: pixels.height * scale)
        return CGRect(x: (bounds.width - size.width) / 2, y: (bounds.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    static func normalise(_ point: CGPoint, in rect: CGRect) -> CGPoint {
        guard rect.width > 0, rect.height > 0 else { return .zero }
        return CGPoint(x: (point.x - rect.minX) / rect.width,
                       y: (point.y - rect.minY) / rect.height)
    }

    static func path(for stroke: MaskStroke, in rect: CGRect) -> Path {
        var p = Path()
        let pts = stroke.points.map {
            CGPoint(x: rect.minX + $0.x * rect.width, y: rect.minY + $0.y * rect.height)
        }
        guard let first = pts.first else { return p }
        // A single tap has no segment to stroke, so give it one of zero length.
        p.move(to: first)
        for pt in pts.dropFirst() { p.addLine(to: pt) }
        if pts.count == 1 { p.addLine(to: first) }
        return p
    }

    // MARK: file

    private func load() {
        let img = NSImage(contentsOfFile: initImagePath)
        image = img
        if let rep = img?.representations.first {
            pixelSize = CGSize(width: rep.pixelsWide, height: rep.pixelsHigh)
        }
        strokes = []
    }

    private func save() {
        let size = pixelSize
        guard let png = MaskEditorView.render(strokes: strokes, size: size) else {
            error = loc.t("No se pudo generar la máscara.", "The mask could not be generated.")
            return
        }
        let url = outputDirectory.appendingPathComponent("mask-\(UUID().uuidString.prefix(8)).png")
        do {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            try png.write(to: url)
            maskPath = url.path
            dismiss()
        } catch {
            self.error = loc.t("No se pudo guardar: %@", "Could not save: %@", error.localizedDescription)
        }
    }

    /// Black canvas with the brush strokes in white, at the init image's pixel size.
    static func render(strokes: [MaskStroke], size: CGSize) -> Data? {
        let w = max(1, Int(size.width)), h = max(1, Int(size.height))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)
        for stroke in strokes {
            guard let first = stroke.points.first else { continue }
            ctx.setStrokeColor(gray: stroke.erases ? 0 : 1, alpha: 1)
            ctx.setLineWidth(stroke.radius * 2 * Double(w))
            // The context's origin is bottom-left and the strokes are top-left.
            func map(_ p: CGPoint) -> CGPoint {
                CGPoint(x: p.x * Double(w), y: Double(h) - p.y * Double(h))
            }
            ctx.beginPath()
            ctx.move(to: map(first))
            for p in stroke.points.dropFirst() { ctx.addLine(to: map(p)) }
            if stroke.points.count == 1 { ctx.addLine(to: map(first)) }
            ctx.strokePath()
        }
        guard let cg = ctx.makeImage() else { return nil }
        let rep = NSBitmapImageRep(cgImage: cg)
        return rep.representation(using: .png, properties: [:])
    }
}
