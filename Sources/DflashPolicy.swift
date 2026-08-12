// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <e.alex.vd@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

enum DflashPolicy {
    static func autoEligible(isMoE _: Bool, ncmoe _: Int) -> Bool {
        true
    }

    static func shouldWarn(fractions: [Double]) -> Bool {
        fractions.count(where: { $0 >= 0.95 }) >= 3
    }
}
