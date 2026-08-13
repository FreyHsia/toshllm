// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

// MARK: - Models

struct ModelsView: View {
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var search: SearchStore
    @EnvironmentObject var modelUpdates: ModelUpdateChecker
    @EnvironmentObject var loc: Localizer
    @State private var tab: Tab = .recommended
    @State private var refreshing = false

    enum Tab: Hashable { case recommended, browse, mine }

    var body: some View {
        VStack(spacing: 0) {
            GlassSegmentedControl(selection: $tab, segments: [
                .init(value: .recommended, title: loc.t("Recomendados", "Recommended"), systemImage: "star"),
                .init(value: .browse, title: loc.t("Buscar", "Browse"), systemImage: "magnifyingglass"),
                .init(value: .mine, title: loc.t("Mis modelos", "My models"),
                      systemImage: models.downloads.contains { $0.phase == .downloading }
                          ? "arrow.down.circle.fill" : "internaldrive"),
            ])
            .padding(12)
            Divider()

            ScrollView {
                switch tab {
                case .recommended: RecommendedTab()
                case .browse: BrowseTab()
                case .mine: MyModelsTab()
                }
            }
        }
        .toolbar {
            ToolbarItem(id: "modelUpdateCheck") {
                Button {
                    Task { await modelUpdates.check(models.models) }
                } label: {
                    Label(loc.t("Buscar actualizaciones", "Check for updates"),
                          systemImage: "arrow.triangle.2.circlepath")
                        .spinningSymbol(modelUpdates.checking)
                }
                .disabled(modelUpdates.checking || models.models.isEmpty)
                .help(loc.t("Comprueba si los modelos descargados se han vuelto a publicar en su repositorio de Hugging Face.",
                            "Checks whether the downloaded models have been re-published in their Hugging Face repo."))
            }

            ToolbarItem(id: "modelFolderRefresh") {
                Button {
                    models.refresh()
                    withAnimation { refreshing = true }
                    Task { try? await Task.sleep(for: .seconds(0.8)); withAnimation { refreshing = false } }
                } label: {
                    Label(loc.t("Actualizar", "Refresh"),
                          systemImage: refreshing ? "checkmark" : "arrow.clockwise")
                        .contentTransition(.symbolEffect(.replace))
                }
                .disabled(refreshing)
                .help(loc.t("Vuelve a escanear la carpeta de modelos para detectar archivos añadidos o eliminados.",
                            "Re-scans the models folder to pick up files added or removed outside the app."))
            }
        }
    }
}

/// Adaptive grid of model cards used across the tabs.
private struct ModelGrid<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: CardMetrics.minWidth), spacing: 12)],
                  alignment: .leading, spacing: 12) {
            content
        }
    }
}

// MARK: - Recommended tab

private struct RecommendedTab: View {
    @EnvironmentObject var loc: Localizer
    @State private var filter: CatalogFilter = .all

    enum CatalogFilter: CaseIterable, Hashable { case all, vision, coder, moe }

    private func label(_ f: CatalogFilter) -> String {
        switch f {
        case .all: return loc.t("Todos", "All")
        case .vision: return loc.t("Visión", "Vision")
        case .coder: return "Coder"
        case .moe: return "MoE"
        }
    }
    private func icon(_ f: CatalogFilter) -> String {
        switch f {
        case .all: return "square.grid.2x2"
        case .vision: return "eye"
        case .coder: return "chevron.left.forwardslash.chevron.right"
        case .moe: return "square.stack.3d.up"
        }
    }
    private func matches(_ m: CatalogModel) -> Bool {
        switch filter {
        case .all: return true
        case .vision: return m.isVision
        case .coder: return m.isCoder
        case .moe: return m.isMoE
        }
    }

    var body: some View {
        let recs = Catalog.recommendations(for: hardware).filter { matches($0.model) }
        let recIDs = Set(recs.map(\.id))
        let rest = Catalog.models.filter { !recIDs.contains($0.id) && matches($0) }

        VStack(alignment: .leading, spacing: 16) {
            GlassSegmentedControl(selection: $filter, segments: CatalogFilter.allCases.map {
                .init(value: $0, title: label($0), systemImage: icon($0))
            })

            if !recs.isEmpty {
                SectionHeader(icon: "star.fill",
                              title: loc.t("Para tu equipo", "For your machine"),
                              subtitle: loc.t("Elegidos según tu GPU/RAM, según lo que necesites.",
                                              "Picked for your GPU/RAM, by what you need."))
                ModelGrid {
                    ForEach(recs) { rec in
                        CatalogCard(model: rec.model,
                                    est: rec.est,
                                    role: rec.role)
                    }
                }
            }

            if !rest.isEmpty {
                SectionHeader(icon: "square.grid.2x2",
                              title: loc.t("Resto del catálogo", "Rest of the catalog"),
                              subtitle: loc.t("Modelos curados con estimaciones medidas para tu equipo.",
                                              "Curated models with measured estimates for your machine."))
                ModelGrid {
                    ForEach(rest) { m in
                        CatalogCard(model: m,
                                    est: Estimator.estimateCurrent(spec: m.spec, hw: hardware),
                                    role: nil)
                    }
                }
            }
        }
        .padding(16)
    }
}

// MARK: - Browse / Trending tab

private struct BrowseTab: View {
    @EnvironmentObject var search: SearchStore
    @EnvironmentObject var loc: Localizer

    private func sortTitle(_ order: HFSortOrder) -> String {
        switch order {
        case .trending: return loc.t("Tendencia", "Trending")
        case .downloads: return loc.t("Descargas", "Downloads")
        case .likes: return loc.t("Favoritos", "Likes")
        case .recent: return loc.t("Recientes", "Recent")
        }
    }

    private func sortIcon(_ order: HFSortOrder) -> String {
        switch order {
        case .trending: return "flame"
        case .downloads: return "arrow.down.circle"
        case .likes: return "heart"
        case .recent: return "clock"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                GlassSearchField(placeholder: loc.t("Buscar GGUF en Hugging Face…", "Search GGUF on Hugging Face…"),
                                 text: $search.query)
                    .onSubmit { Task { await search.search() } }
                    .onChange(of: search.query) {
                        if search.query.isEmpty { search.didSearch = false }
                    }

                Button(loc.t("Buscar", "Search"), systemImage: "magnifyingglass") {
                    Task { await search.search() }
                }
                .glassButton(prominent: true)
                .labelStyle(.titleOnly)
                .disabled(search.query.isEmpty || search.searching)
                .opacity(search.searching ? 0.6 : 1)
            }

            HStack(spacing: 10) {
                GlassSegmentedControl(selection: $search.sort, segments: HFSortOrder.allCases.map {
                    .init(value: $0, title: sortTitle($0), systemImage: sortIcon($0))
                })
                .onChange(of: search.sort) { Task { await search.reload() } }
                .help(loc.t("Orden en el que Hugging Face devuelve los repositorios; siempre limitado a los que publican GGUF. «Recientes» además exige un mínimo de descargas, para no llenarse de repositorios recién subidos que nadie usa.",
                            "The order Hugging Face returns repositories in, always limited to those publishing GGUF. “Recent” also requires a minimum number of downloads, so it doesn't fill up with freshly pushed repos nobody uses."))
                if search.searching || search.loadingTrending {
                    ProgressView().controlSize(.small)
                }
                Spacer(minLength: 0)
            }

            if search.didSearch && !search.query.isEmpty {
                if search.results.isEmpty && !search.searching {
                    ContentUnavailableView.search(text: search.query)
                } else {
                    SectionHeader(icon: "magnifyingglass",
                                  title: loc.t("Resultados", "Results"), subtitle: nil)
                    LazyVStack(spacing: 12) {
                        ForEach(search.results) { RepoCard(repo: $0) }
                    }
                }
            } else {
                SectionHeader(icon: sortIcon(search.sort),
                              title: loc.t("Modelos GGUF en Hugging Face", "GGUF models on Hugging Face"),
                              subtitle: loc.t("Despliega un repositorio para ver sus cuantizaciones y si caben.",
                                              "Expand a repo to see its quants and whether they fit."))
                LazyVStack(spacing: 12) {
                    ForEach(search.trending) { RepoCard(repo: $0) }
                }
            }
        }
        .padding(16)
        .task { await search.loadTrending() }
    }
}

/// An expandable Hugging Face repo: header with stats, expands to its GGUF
/// files with per-quant fit badges. Shared by search results and trending.
private struct RepoCard: View {
    let repo: HFRepo
    @EnvironmentObject var search: SearchStore
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var loc: Localizer

    private var isVisionRepo: Bool { search.visionRepos.contains(repo.id) }
    private var verifiedVision: Bool {
        Catalog.models.contains { $0.isVision && $0.urlString.contains(repo.id) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                Task { await search.toggleFiles(repo: repo.id) }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: search.expanded == repo.id ? "chevron.down" : "chevron.right")
                        .font(.caption).foregroundStyle(.secondary).frame(width: 12)
                    Text(repo.id).font(.callout.weight(.medium)).lineLimit(1)
                    // Once expanded, the file list is loaded; a sibling mmproj means
                    // it's a vision model (the projector is fetched with the model).
                    if isVisionRepo {
                        TagBadge(text: loc.t("Visión", "Vision"), icon: "eye", color: .purple)
                        if verifiedVision {
                            Label(loc.t("Verificado", "Verified"), systemImage: "checkmark.seal.fill")
                                .labelStyle(.titleAndIcon)
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 7).padding(.vertical, 2)
                                .background(Color.green.opacity(0.16), in: Capsule())
                                .foregroundStyle(.green)
                        } else {
                            Label(loc.t("Sin verificar", "Unverified"), systemImage: "exclamationmark.triangle.fill")
                                .labelStyle(.titleAndIcon)
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 7).padding(.vertical, 2)
                                .background(Color.orange.opacity(0.18), in: Capsule())
                                .foregroundStyle(.orange)
                        }
                    }
                    Spacer()
                    if let l = repo.likes {
                        Label("\(l)", systemImage: "heart").font(.caption).foregroundStyle(.secondary)
                    }
                    if let d = repo.downloads {
                        Label(compact(d), systemImage: "arrow.down.circle")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let updated = repo.lastModified, updated > .distantPast {
                        Text(updated, format: .relative(presentation: .named))
                            .font(.caption).foregroundStyle(.secondary)
                            .help(loc.t("Última actualización del repositorio", "Repository last updated"))
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(12)

            if search.expanded == repo.id {
                Divider()
                if isVisionRepo && !verifiedVision {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(loc.t("Visión sin verificar. Con el botón de visión se descarga el proyector (mmproj) que mejor coincida, pero no se garantiza la compatibilidad. Si la visión falla, comprueba que el mmproj corresponda a este modelo.",
                                   "Unverified vision. The vision button downloads the best-matching projector (mmproj), but compatibility isn't guaranteed. If vision fails, check that the mmproj matches this model."))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(.caption)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.12))
                }
                Group {
                    if let files = search.files[repo.id] {
                        if files.isEmpty {
                            Text(loc.t("Sin archivos .gguf directos", "No direct .gguf files"))
                                .font(.caption).foregroundStyle(.secondary).padding(12)
                        } else {
                            LazyVStack(spacing: 8) {
                                ForEach(files) { FileRow(repo: repo.id, file: $0) }
                            }
                            .padding(12)
                        }
                    } else {
                        ProgressView().controlSize(.small).padding(12)
                    }
                }
            }
        }
        .cardSurface()
    }

    private func compact(_ n: Int) -> String {
        n >= 1_000_000 ? String(format: "%.1fM", Double(n) / 1e6)
            : n >= 1_000 ? String(format: "%.0fk", Double(n) / 1e3) : "\(n)"
    }
}

private struct FileRow: View {
    let repo: String
    let file: HFFile
    @EnvironmentObject var search: SearchStore
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var loc: Localizer
    @State private var visionBusy = false

    private var repoHasVision: Bool { search.visionRepos.contains(repo) }
    private var stem: String { URL(fileURLWithPath: file.path).deletingPathExtension().lastPathComponent }
    private var projName: String { stem + ".mmproj.gguf" }
    private var draftName: String { stem + ".dflash.gguf" }

    var body: some View {
        let est = Estimator.estimateCurrent(
            spec: .estimated(fileBytes: file.sizeBytes, isMoE: search.isMoE(repo: repo, file: file),
                             name: URL(fileURLWithPath: file.path).lastPathComponent), hw: hardware)
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                ModelTitleLabel(ModelName.forPath(file.path), titleFont: .caption)
                EstimateLine(est: est)
            }
            Spacer()
            Text(file.sizeGB).font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                modelPill(est: est)
                if repoHasVision { visionPill }
                if let draft = search.draftRepos[repo] { dflashPill(draft) }
            }
        }
    }

    private func modelPill(est: MemoryEstimate) -> some View {
        let names = file.paths.map { URL(fileURLWithPath: $0).lastPathComponent }
        return AssetDownloadButton(
            icon: "arrow.down.circle", label: loc.t("Modelo", "Model"), prominent: true,
            help: loc.t("Descargar el modelo base", "Download the base model"),
            downloaded: names.allSatisfy { models.isDownloaded(fileName: $0) },
            item: names.compactMap { models.downloadItem(fileName: $0) }.first,
            busy: false, disabledReason: est.level == .no ? loc.t("No cabe", "Won't fit") : nil) {
            for path in file.paths where !models.isDownloaded(fileName: URL(fileURLWithPath: path).lastPathComponent) {
                models.download(urlString: search.downloadURL(repo: repo, file: path), fetchVisionProjector: false)
            }
        }
    }

    // Vision (mmproj) is opt-in. The spinner covers the HF tree lookup before the
    // transfer is queued (the projector lives as a sibling in the same repo).
    private var visionPill: some View {
        AssetDownloadButton(
            icon: "eye.circle", label: loc.t("Visión", "Vision"),
            help: loc.t("Descargar el proyector de visión (mmproj)", "Download the vision projector (mmproj)"),
            downloaded: models.isDownloaded(fileName: projName),
            item: models.downloadItem(fileName: projName), busy: visionBusy, disabledReason: nil) {
            visionBusy = true
            Task {
                await models.autoFetchProjector(for: URL(string: search.downloadURL(repo: repo, file: file.path))!)
                visionBusy = false
            }
        }
    }

    private func dflashPill(_ draft: SearchStore.DraftInfo) -> some View {
        AssetDownloadButton(
            icon: "bolt.circle", label: "DFlash",
            help: loc.t("Descargar el draft DFlash para decodificación especulativa (experimental). Por ahora acelera la generación en modelos MoE con expertos en CPU (offload); en denso a GPU completa puede ser más lento.",
                        "Download the DFlash draft for speculative decoding (experimental). For now it speeds up generation on MoE models with CPU-offloaded experts; on full-GPU dense models it can be slower."),
            downloaded: models.isDownloaded(fileName: draftName),
            item: models.downloadItem(fileName: draftName), busy: false, disabledReason: nil) {
            models.downloadDflashDraft(repo: draft.repo, file: draft.file, modelStem: stem)
        }
    }
}

/// One download control in a consistent visual language across the base model,
/// vision projector and DFlash draft: a tinted pill that becomes a live progress
/// bar while transferring and a green check once present.
private struct AssetDownloadButton: View {
    let icon: String
    let label: String
    var prominent = false
    let help: String
    let downloaded: Bool
    let item: DownloadItem?
    let busy: Bool
    let disabledReason: String?
    let action: () -> Void

    var body: some View {
        Group {
            if let item, !item.finished, item.error == nil {
                InlineDownloadProgress(item: item)
            } else if downloaded {
                Label(label, systemImage: "checkmark.circle.fill")
                    .labelStyle(.titleAndIcon).font(.caption).foregroundStyle(.green)
            } else if busy {
                HStack(spacing: 5) {
                    ProgressView().controlSize(.small)
                    Text(label).font(.caption).foregroundStyle(.secondary)
                }
            } else if let reason = disabledReason {
                Text(reason).font(.caption).foregroundStyle(.red)
            } else {
                Button(label, systemImage: icon, action: action)
                    .font(.caption)
                    .glassButton(prominent: prominent)
                    .controlSize(.small)
            }
        }
        .fixedSize()
        .help(help)
    }
}

// MARK: - My models tab

private struct MyModelsTab: View {
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var modelUpdates: ModelUpdateChecker
    @EnvironmentObject var loc: Localizer
    @AppStorage(SettingsKeys.modelPath) private var modelPath = ""
    @AppStorage(SettingsKeys.ncmoe) private var ncmoe = 0
    @State private var customURL = ""
    @State private var pendingDelete: LocalModel?
    @State private var pendingUpdate: LocalModel?
    @State private var filter: LocalFilter = .all

    enum LocalFilter: CaseIterable, Hashable { case all, vision, mtp, dflash, moe }

    private func label(_ f: LocalFilter) -> String {
        switch f {
        case .all: return loc.t("Todos", "All")
        case .vision: return loc.t("Visión", "Vision")
        case .mtp: return "MTP"
        case .dflash: return "DFlash"
        case .moe: return "MoE"
        }
    }

    private func icon(_ f: LocalFilter) -> String {
        switch f {
        case .all: return "square.grid.2x2"
        case .vision: return "eye"
        case .mtp: return "hare"
        case .dflash: return "bolt"
        case .moe: return "square.stack.3d.up"
        }
    }

    private var shown: [LocalModel] {
        guard filter != .all else { return models.models }
        return models.models.filter { model in
            let traits = ModelTraitsCache.traits(for: model.url.path)
            switch filter {
            case .all: return true
            case .vision: return traits.hasVision
            case .mtp: return traits.hasMTP
            case .dflash: return traits.hasDflash
            case .moe: return traits.isMoE
            }
        }
    }

    private var modelsFolderShort: String {
        (models.directory.path as NSString).abbreviatingWithTildeInPath
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !models.downloads.isEmpty {
                SectionHeader(icon: "arrow.down.circle", title: loc.t("Descargas", "Downloads"), subtitle: nil)
                VStack(spacing: 8) {
                    ForEach(models.downloads) { DownloadRow(item: $0) }
                }
                Button(loc.t("Limpiar terminadas", "Clear finished"), systemImage: "checkmark.circle") {
                    models.clearFinishedDownloads()
                }
                .glassButton()
                .controlSize(.small)
            }

            SectionHeader(icon: "internaldrive",
                          title: loc.t("Archivos locales en \(modelsFolderShort)",
                                       "Local files in \(modelsFolderShort)"),
                          subtitle: nil)
            if !models.models.isEmpty {
                GlassSegmentedControl(selection: $filter, segments: LocalFilter.allCases.map {
                    .init(value: $0, title: label($0), systemImage: icon($0))
                })
            }
            if models.models.isEmpty {
                ContentUnavailableView(loc.t("Todavía no hay modelos", "No models yet"),
                                       systemImage: "internaldrive",
                                       description: Text(loc.t("Descarga uno desde Recomendados o Buscar y aparecerá aquí.",
                                                               "Download one from Recommended or Browse and it will show up here.")))
            } else if shown.isEmpty {
                ContentUnavailableView(loc.t("Ningún modelo con esa característica", "No model with that trait"),
                                       systemImage: icon(filter),
                                       description: Text(loc.t("Ninguno de tus modelos descargados la tiene.",
                                                               "None of your downloaded models has it.")))
            } else {
                ModelGrid {
                    ForEach(shown) { m in
                        LocalModelCard(model: m, pendingDelete: $pendingDelete, pendingUpdate: $pendingUpdate)
                    }
                }
            }

            SectionHeader(icon: "link",
                          title: loc.t("URL personalizada (GGUF directo)", "Custom URL (direct GGUF)"),
                          subtitle: nil)
            HStack {
                TextField("https://huggingface.co/…/resolve/main/model.gguf", text: $customURL)
                    .textFieldStyle(.roundedBorder)
                Button(loc.t("Descargar", "Download"), systemImage: "arrow.down.circle") {
                    models.download(urlString: customURL)
                    customURL = ""
                }
                .glassButton(prominent: true)
                .disabled(!customURL.hasPrefix("http"))
            }
        }
        .padding(16)
        .task { await modelUpdates.checkIfStale(models.models) }
        .confirmationDialog(
            loc.t("¿Actualizar \(pendingUpdate?.name ?? "")?", "Update \(pendingUpdate?.name ?? "")?"),
            isPresented: Binding(get: { pendingUpdate != nil }, set: { if !$0 { pendingUpdate = nil } })
        ) {
            Button(loc.t("Descargar y reemplazar", "Download and replace")) {
                if let m = pendingUpdate { models.update(m) }
                pendingUpdate = nil
            }
            Button(loc.t("Cancelar", "Cancel"), role: .cancel) { pendingUpdate = nil }
        } message: {
            Text(loc.t("Se descarga de nuevo desde su repositorio y el archivo actual se reemplaza al verificarse el checksum. Si el modelo está cargado, reinicia el servidor al terminar.",
                       "It is downloaded again from its repo and the current file is replaced once the checksum verifies. If the model is loaded, restart the server afterwards."))
        }
        .confirmationDialog(
            loc.t("¿Eliminar \(pendingDelete?.name ?? "")?", "Delete \(pendingDelete?.name ?? "")?"),
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })
        ) {
            Button(loc.t("Mover a la Papelera", "Move to Trash"), role: .destructive) {
                if let m = pendingDelete {
                    if modelPath == m.url.path { modelPath = "" }
                    models.delete(m)
                }
                pendingDelete = nil
            }
            Button(loc.t("Cancelar", "Cancel"), role: .cancel) { pendingDelete = nil }
        } message: {
            if let m = pendingDelete, ServerSettings.mmprojPath(forModel: m.url.path) != nil {
                Text(loc.t("Se eliminará también su archivo de visión (mmproj).",
                           "Its vision file (mmproj) will be removed too."))
            }
        }
    }
}

// MARK: - Cards

/// A catalog model as a card: optional recommendation role chip, name, size,
/// MoE badge, blurb, fit estimate and the download/use action.
private struct CatalogCard: View {
    let model: CatalogModel
    let est: MemoryEstimate
    let role: Catalog.Recommendation.Role?
    @EnvironmentObject var loc: Localizer

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                if let role { RoleChip(role: role) }
                if model.isMoE { MoEBadge() }
                if model.isVision { TagBadge(text: loc.t("Visión", "Vision"), icon: "eye", color: .purple) }
                if model.isCoder { TagBadge(text: "Coder", icon: "chevron.left.forwardslash.chevron.right", color: .blue) }
                Spacer(minLength: 0)
                Text(String(format: "%.1f GB", model.spec.fileGB))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text(ModelName(model.name).title).font(.headline)
            Text(model.detail(loc.isSpanish))
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            EstimateLine(est: est)
            Spacer(minLength: 2)
            HStack { Spacer(); CatalogActionButton(model: model, est: est) }
        }
        .padding(CardMetrics.padding)
        .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
        .cardSurface()
    }
}

// MARK: - Small components

private struct SectionHeader: View {
    let icon: String
    let title: String
    let subtitle: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(title, systemImage: icon)
                .font(.headline)
            if let subtitle {
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

private struct RoleChip: View {
    let role: Catalog.Recommendation.Role
    @EnvironmentObject var loc: Localizer

    static func color(_ role: Catalog.Recommendation.Role) -> Color {
        switch role {
        case .fast: return .green
        case .balanced: return .blue
        case .quality: return .purple
        case .coding: return .orange
        }
    }

    var body: some View {
        let (text, icon): (String, String) = {
            switch role {
            case .fast:     return (loc.t("Más rápido", "Fastest"), "hare.fill")
            case .balanced: return (loc.t("Equilibrado", "Balanced"), "scalemass.fill")
            case .quality:  return (loc.t("Máxima calidad", "Top quality"), "sparkles")
            case .coding:   return (loc.t("Programación", "Coding"), "chevron.left.forwardslash.chevron.right")
            }
        }()
        let color = Self.color(role)
        return Label(text, systemImage: icon)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 7).padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}

struct MoEBadge: View {
    var body: some View { TagBadge(text: "MoE", icon: "square.stack.3d.up", color: Color.appAccent) }
}

/// Small capsule tag (MoE / Vision / MTP / DFlash / Coder) shown on model cards.
/// The icon carries the meaning where color alone would.
struct TagBadge: View {
    let text: String
    var icon: String?
    let color: Color

    var body: some View {
        Group {
            if let icon {
                Label(text, systemImage: icon)
            } else {
                Text(text)
            }
        }
        .font(.caption)
        .lineLimit(1)
        .fixedSize()
        .foregroundStyle(color)
        .padding(.horizontal, 7).padding(.vertical, 2)
        .background(color.opacity(0.16), in: Capsule())
    }
}

struct DownloadRow: View {
    @ObservedObject var item: DownloadItem
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var models: ModelStore

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(item.fileName).font(.callout)
                Spacer()
                switch item.phase {
                case .preparing:
                    Text(loc.t("Preparando…", "Preparing…"))
                        .font(.caption).foregroundStyle(.secondary)
                case .verifying:
                    ProgressView().controlSize(.small)
                    Text(loc.t("Verificando SHA-256…", "Verifying SHA-256…"))
                        .font(.caption).foregroundStyle(.secondary)
                case .finished:
                    Label(loc.t("Completada y verificada", "Done and verified"),
                          systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green).font(.caption)
                case .failed(let message):
                    Text(message).font(.caption).foregroundStyle(.red)
                        .lineLimit(2).frame(maxWidth: 300, alignment: .trailing)
                    Button(loc.t("Reintentar", "Retry"), systemImage: "arrow.clockwise") {
                        models.retry(item)
                    }
                    .glassButton()
                    .controlSize(.small)
                    .help(loc.t("Reintentar la descarga desde cero.", "Retry the download from scratch."))
                case .downloading, .paused:
                    Text(String(format: "%.0f / %.0f MB", item.receivedMB, item.totalMB))
                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    if item.phase == .paused {
                        Button { item.resume() } label: { Image(systemName: "play.circle") }
                            .buttonStyle(.borderless)
                            .iconHelp(loc.t("Reanudar", "Resume"))
                    } else {
                        Button { item.pause() } label: { Image(systemName: "pause.circle") }
                            .buttonStyle(.borderless)
                            .iconHelp(loc.t("Pausar (reanudable)", "Pause (resumable)"))
                    }
                    Button { item.cancel() } label: { Image(systemName: "xmark.circle") }
                        .buttonStyle(.borderless)
                        .iconHelp(loc.t("Cancelar", "Cancel"))
                }
            }
            if item.phase == .downloading || item.phase == .paused {
                ProgressView(value: item.progress)
                    .tint(item.phase == .paused ? .orange : .accentColor)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }
}
