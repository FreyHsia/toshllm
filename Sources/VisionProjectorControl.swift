// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UniformTypeIdentifiers

/// Per-model vision control: an on/off switch plus, when on, the projector to
/// pin or auto-pair. `inline` is one compact row; `settings` is a form section.
struct VisionProjectorControl: View {
    enum Layout { case inline, settings }

    let modelPath: String
    /// Anchor the switch to the leading edge (model cards, left-aligned) or the
    /// trailing edge (server card, right-aligned) so revealing the menu never
    /// shifts the switch's position.
    var switchLeading: Bool = false
    var layout: Layout = .inline
    @EnvironmentObject var loc: Localizer
    @State private var version = 0

    @ViewBuilder
    var body: some View {
        let _ = version
        let override = ServerSettings.mmprojOverride(forModel: modelPath)
        let enabled = override != ""   // nil (auto) or a pinned path = on; "" = off
        let resolved = ServerSettings.mmprojPath(forModel: modelPath)
        let mismatch = resolved.map { ServerSettings.mmprojIncompatible(model: modelPath, projector: $0) } ?? false
        let current = resolved.map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent }
            ?? loc.t("automático", "automatic")
        if layout == .settings {
            settingsRows(enabled: enabled, current: current, mismatch: mismatch)
        } else {
            inlineRow(enabled: enabled, current: current, mismatch: mismatch)
        }
    }

    private func inlineRow(enabled: Bool, current: String, mismatch: Bool) -> some View {
        HStack(spacing: 8) {
            if switchLeading { toggle(enabled) }
            if enabled {
                Menu {
                    Button(loc.t("Elegir archivo…", "Choose file…")) { pick() }
                    Button(loc.t("Automático", "Automatic")) { set(nil) }
                } label: {
                    HStack(spacing: 4) {
                        Text("mmproj:").foregroundStyle(.secondary)
                        Text(current).lineLimit(1).truncationMode(.middle)
                        if mismatch {
                            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                                .help(loc.t("El proyector no coincide con la dimensión del modelo; la visión podría fallar.",
                                            "The projector doesn't match the model's dimension; vision may fail."))
                        }
                    }
                    .font(.caption)
                }
                .menuStyle(.button).buttonStyle(.bordered).controlSize(.small).fixedSize()
            } else {
                Text(loc.t("solo texto", "text-only")).font(.caption).foregroundStyle(.secondary)
            }
            if !switchLeading { toggle(enabled) }
        }
    }

    private func settingsRows(enabled: Bool, current: String, mismatch: Bool) -> some View {
        Group {
            Toggle(loc.t("Leer imágenes", "Read images"),
                   isOn: Binding(get: { enabled }, set: { set($0 ? nil : "") }))
                .help(loc.t("Con la visión apagada el modelo responde solo a texto y no carga el proyector.",
                            "With vision off the model answers text only and the projector is not loaded."))
            if enabled {
                LabeledContent(loc.t("Proyector", "Projector")) {
                    Menu(current) {
                        Button(loc.t("Automático", "Automatic")) { set(nil) }
                        Button(loc.t("Elegir archivo…", "Choose file…")) { pick() }
                    }
                    .fixedSize()
                }
                if mismatch {
                    Label(loc.t("El proyector no coincide con la dimensión del modelo; la visión podría fallar.",
                                "The projector doesn't match the model's dimension; vision may fail."),
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func toggle(_ enabled: Bool) -> some View {
        Toggle("", isOn: Binding(get: { enabled }, set: { set($0 ? nil : "") }))
            .labelsHidden().toggleStyle(.switch).controlSize(.small)
            .help(loc.t("Activar o desactivar la visión para este modelo", "Turn vision on or off for this model"))
    }

    private func set(_ value: String?) {
        ServerSettings.setMmprojOverride(value, forModel: modelPath)
        // Enabling (auto or a file) re-arms the vision load path.
        if value != "" { UserDefaults.standard.set(true, forKey: SettingsKeys.loadVision) }
        version += 1
    }

    private func pick() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf")].compactMap { $0 }
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { set(url.path) }
    }
}
