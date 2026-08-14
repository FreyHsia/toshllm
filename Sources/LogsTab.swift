// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

// MARK: - Logs

/// Picks which server's log to view (when more than one exists) and shows it.
struct LogsView: View {
    @EnvironmentObject var manager: ServerManager
    @State private var selectedID: UUID?

    private var selected: ServerController {
        manager.servers.first { $0.id == selectedID } ?? manager.servers[0]
    }

    var body: some View {
        VStack(spacing: 0) {
            if manager.servers.count > 1 {
                GlassSegmentedControl(selection: Binding(get: { selected.id }, set: { selectedID = $0 }),
                                      segments: manager.servers.map { .init(value: $0.id, title: $0.name) })
                    .padding(.horizontal, 12).padding(.top, 10)
            }
            ServerLogView(server: selected)
        }
    }
}

/// Dedicated, full-height server-log viewer with search, severity filtering,
/// toggleable auto-follow, copy and diagnostics export. The wrapper (`LogsView`)
/// picks which server and passes it in as an observed object.
struct ServerLogView: View {
    @ObservedObject var server: ServerController
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var loc: Localizer
    @AppStorage(SettingsKeys.modelPath) private var modelPath = ""
    @AppStorage(SettingsKeys.ncmoe) private var ncmoe = 0

    @State private var query = ""
    /// Minimum severity to show: 0 = everything, 1 = warnings+, 2 = errors only.
    @State private var minLevel = 0
    @State private var autoFollow = true
    @State private var copied = false
    /// Which engine's log to show: the chat server, or the image studio.
    @State private var logSource = "server"
    @State private var imageLog = ""

    /// The raw log for the selected source. The image log is read from disk (the
    /// image studio runs in another window).
    private var rawLog: String { logSource == "images" ? imageLog : server.log }
    private func reloadImageLog() {
        imageLog = ImageGenerator.latestLogURL.flatMap { try? String(contentsOf: $0, encoding: .utf8) } ?? ""
    }

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider()
            logBody
        }
        // The image log lives in a file written by the other window; poll it while shown.
        .onReceive(Timer.publish(every: 1.5, on: .main, in: .common).autoconnect()) { _ in
            if logSource == "images" { reloadImageLog() }
        }
    }

    // MARK: controls

    private var controls: some View {
        VStack(spacing: 8) {
            GlassSegmentedControl(selection: $logSource, segments: [
                .init(value: "server", title: loc.t("Servidor", "Server"), systemImage: "server.rack"),
                .init(value: "images", title: loc.t("Imágenes", "Images"), systemImage: "photo"),
            ])
            .help(loc.t("Registro del servidor de chat o del motor de imágenes.",
                        "Chat server log or the image engine log."))
            .onChange(of: logSource) { if logSource == "images" { reloadImageLog() } }
            serverControls
            HStack(spacing: 10) {
                GlassSearchField(placeholder: loc.t("Filtrar en el registro…", "Filter the log…"), text: $query)

                GlassSegmentedControl(selection: $minLevel, segments: [
                    .init(value: 0, title: loc.t("Todo", "All")),
                    .init(value: 1, title: loc.t("Avisos", "Warnings"), systemImage: "exclamationmark.triangle"),
                    .init(value: 2, title: loc.t("Errores", "Errors"), systemImage: "xmark.octagon"),
                ])
                .help(loc.t("Filtra por severidad mínima de cada línea.",
                            "Filter by each line's minimum severity."))
            }

            HStack(spacing: 12) {
                Toggle(isOn: $autoFollow) {
                    Label(loc.t("Seguir", "Follow"), systemImage: "arrow.down.to.line")
                }
                .toggleStyle(.button)
                .glassButton()
                .controlSize(.small)
                .help(loc.t("Sigue automáticamente las líneas nuevas al final.",
                            "Automatically follow new lines at the bottom."))

                Spacer()

                Text(loc.t("^[\(matchCount) línea](inflect: true)", "^[\(matchCount) line](inflect: true)"))
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()

                Button(copied ? loc.t("Copiado", "Copied") : loc.t("Copiar", "Copy"),
                       systemImage: copied ? "checkmark" : "doc.on.doc") { copy() }
                    .glassButton()
                    .controlSize(.small)
                    .contentTransition(.symbolEffect(.replace))
                    .help(loc.t("Copia lo que se muestra (con los filtros aplicados).",
                                "Copies what's shown (with filters applied)."))

                Menu(loc.t("Más acciones", "More actions"), systemImage: "ellipsis") {
                    Button(loc.t("Logs en Finder", "Logs in Finder"), systemImage: "folder") {
                        let file = logSource == "images"
                            ? (ImageGenerator.latestLogURL ?? server.logsDirectory) : server.logFileURL
                        revealInFinder(file: file, folder: server.logsDirectory)
                    }
                    Button(loc.t("Exportar diagnóstico…", "Export diagnostics…"),
                           systemImage: "square.and.arrow.up") { exportDiagnostics() }
                    Divider()
                    Button(loc.t("Limpiar en pantalla", "Clear on screen"),
                           systemImage: "trash", role: .destructive) {
                        if logSource == "images" { imageLog = "" } else { server.log = "" }
                    }
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                    .labelStyle(.iconOnly)
                .fixedSize()
                .help(loc.t("Abrir la carpeta de registros, exportar un diagnóstico o vaciar la vista. El archivo en disco se conserva.",
                            "Open the logs folder, export diagnostics, or clear the view. The file on disk is kept."))
            }
            .font(.callout)
        }
        .padding(12)
    }

    // MARK: server controls

    /// Start/stop the server and pick a model right here, so you can drive a debug
    /// session without leaving the log you're watching.
    private var serverControls: some View {
        HStack(spacing: 10) {
            // The model picker edits the global config, so only the primary server can
            // change its model here; an added server shows its own model read-only.
            if server.profile == nil {
                Menu {
                    if models.models.isEmpty {
                        Text(loc.t("No hay modelos descargados", "No downloaded models"))
                    } else {
                        ForEach(models.models) { m in
                            Button {
                                modelPath = m.url.path
                                ncmoe = Estimator.estimateCurrent(spec: Catalog.spec(forLocal: m), hw: hardware).suggestedNcmoe
                            } label: {
                                Label(m.name + (ModelTraitsCache.cached(for: m.url.path)?.pickerSuffix(spanish: loc.isSpanish) ?? ""),
                                      systemImage: modelPath == m.url.path ? "checkmark" : "cpu")
                            }
                        }
                    }
                } label: {
                    Label(selectedModelName, systemImage: "cpu")
                        .lineLimit(1).truncationMode(.middle)
                }
                .menuStyle(.button)
                .glassButton()
                .controlSize(.small)
                .fixedSize()
                .help(loc.t("Selecciona el modelo a cargar.", "Pick the model to load."))
            } else {
                Label(selectedModelName, systemImage: "cpu")
                    .lineLimit(1).truncationMode(.middle)
            }

            Spacer()

            statusDot
            switch server.state {
            case .running, .starting:
                Button(loc.t("Detener", "Stop"), systemImage: "stop.fill", role: .destructive) {
                    server.stop()
                }
                .glassButton()
            default:
                Button(loc.t("Iniciar servidor", "Start server"), systemImage: "play.fill") {
                    server.start(server.effectiveSettings())
                }
                .glassButton(prominent: true)
                .disabled(server.effectiveSettings().modelPath.isEmpty)
            }
        }
        .font(.callout)
        .padding(.bottom, 2)
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(filteredLog, forType: .string)
        copied = true
        Task { try? await Task.sleep(for: .seconds(1.5)); copied = false }
    }

    private var selectedModelName: String {
        let mp = server.effectiveSettings().modelPath
        guard !mp.isEmpty else { return loc.t("Seleccionar modelo…", "Select model…") }
        return URL(fileURLWithPath: mp).lastPathComponent
    }

    @ViewBuilder private var statusDot: some View {
        switch server.state {
        case .running:  Circle().fill(.green).frame(width: 8, height: 8)
        case .starting: Circle().fill(.orange).frame(width: 8, height: 8)
        case .failed:   Circle().fill(.red).frame(width: 8, height: 8)
        case .stopped:  Circle().fill(.secondary).frame(width: 8, height: 8)
        }
    }

    // MARK: log body

    @ViewBuilder
    private var logBody: some View {
        if filteredLog.isEmpty {
            emptyState
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(.background.secondary)
        } else {
            scrollingLog
        }
    }

    private var scrollingLog: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(filteredLog)
                    .font(.caption.monospaced())
                    .lineSpacing(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(12)
                    .id("logEnd")
            }
            .background(.background.secondary)
            .onChange(of: server.log) { _, _ in
                if autoFollow { proxy.scrollTo("logEnd", anchor: .bottom) }
            }
            .onChange(of: autoFollow) { _, on in
                if on { proxy.scrollTo("logEnd", anchor: .bottom) }
            }
            .onAppear { proxy.scrollTo("logEnd", anchor: .bottom) }
        }
    }

    @ViewBuilder private var emptyState: some View {
        if rawLog.isEmpty {
            ContentUnavailableView(loc.t("Sin registro todavía", "No log yet"),
                                   systemImage: "text.alignleft",
                                   description: Text(loc.t("Inicia el servidor y su salida aparecerá aquí en vivo.",
                                                           "Start the server and its output shows up here live.")))
        } else if !query.trimmingCharacters(in: .whitespaces).isEmpty {
            ContentUnavailableView.search(text: query)
        } else {
            ContentUnavailableView(loc.t("Nada en este nivel", "Nothing at this level"),
                                   systemImage: "line.3.horizontal.decrease.circle",
                                   description: Text(loc.t("El registro no tiene líneas de esa severidad.",
                                                           "The log has no lines of that severity.")))
        }
    }

    // MARK: filtering

    /// llama-server prefixes each line with a timestamp then a severity char
    /// (`I`/`W`/`E`). Continuation lines (multi-line dumps, app stdout) have no
    /// such marker; they count as info so they only show in the "All" view.
    private func rank(_ line: Substring) -> Int {
        let parts = line.split(separator: " ", maxSplits: 2, omittingEmptySubsequences: true)
        guard parts.count >= 2 else { return 0 }
        switch parts[1] {
        case "E": return 2
        case "W": return 1
        default:  return 0
        }
    }

    private var filteredLog: String {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard minLevel > 0 || !q.isEmpty else { return rawLog }
        let lines = rawLog.split(separator: "\n", omittingEmptySubsequences: false)
        let out = lines.filter { line in
            if minLevel > 0, rank(line) < minLevel { return false }
            if !q.isEmpty, !line.lowercased().contains(q) { return false }
            return true
        }
        return out.joined(separator: "\n")
    }

    private var matchCount: Int {
        filteredLog.isEmpty ? 0 : filteredLog.split(separator: "\n", omittingEmptySubsequences: false).count
    }

    // MARK: diagnostics

    private func exportDiagnostics() {
        let settings = ServerSettings.fromDefaults()
        let logTail = (try? String(contentsOf: server.logFileURL, encoding: .utf8))?
            .split(separator: "\n").suffix(250).joined(separator: "\n") ?? server.log
        let gpu = hardware.bestGPU.map { "\($0.name) (\($0.vramMB) MB VRAM)" } ?? "—"
        let report = """
        ToshLLM \(AppInfo.version) — diagnostics
        Date: \(Date().formatted(.iso8601))

        ## Hardware
        CPU: \(hardware.cpuBrand) (\(hardware.physicalCores)c/\(hardware.logicalCores)t)
        RAM: \(Int(hardware.ramGB)) GB
        GPU: \(gpu)
        Arch: \(hardware.arch)

        ## Configuration
        model: \(URL(fileURLWithPath: settings.modelPath).lastPathComponent)
        engine: \(settings.serverBinary)
        args: \(settings.arguments.joined(separator: " "))

        ## Recent log
        \(logTail)
        """
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "toshllm-diagnostics.txt"
        if panel.runModal() == .OK, let url = panel.url {
            try? report.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
