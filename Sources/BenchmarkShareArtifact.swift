// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <e.alex.vd@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct BenchmarkShareArtifact: Identifiable {
    let name: String
    let sha256: String
    let sizeBytes: Int64

    var id: String { "\(name):\(sha256)" }

    var formattedSize: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}
