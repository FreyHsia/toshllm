// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM
final class DraftFilterTests: XCTestCase {
    func testDraftsStayOutOfThePicker() {
        for name in ["DeepSeek-V4-DSpark-Q4_K.gguf", "Nemotron3.5-dspark-draft.gguf",
                     "Qwen3-8B.dflash.gguf", "mtp-Qwen3.gguf", "Qwen3.mtp.gguf"] {
            XCTAssertTrue(GGUFFile.isDraft("/models/\(name)"), name)
        }
        for name in ["Qwen3-8B-Q4_K_M.gguf", "DeepSeek-V4-Flash-Q4_K.gguf"] {
            XCTAssertFalse(GGUFFile.isDraft("/models/\(name)"), name)
        }
    }
}
