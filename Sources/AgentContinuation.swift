// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <e.alex.vd@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct PendingAgentContinuation: Identifiable, Equatable {
    let conversationID: UUID
    var id: UUID { conversationID }
}
