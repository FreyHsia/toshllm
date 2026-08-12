// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <e.alex.vd@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct QueuedChatMessage: Identifiable, Equatable {
    let conversationID: UUID
    var text: String
    var attachments: [ChatAttachment]
    var imageURIs: [String]
    var id: UUID { conversationID }
}
