// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <e.alex.vd@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ChatCapabilityTests: XCTestCase {
    func testThinkingSupportRecognizesTemplateControls() {
        XCTAssertTrue(ThinkingSupportDetector.supportsThinking("{% if enable_thinking %}<think>{% endif %}"))
        XCTAssertTrue(ThinkingSupportDetector.supportsThinking("{{ reasoning_effort }}"))
        XCTAssertFalse(ThinkingSupportDetector.supportsThinking("{{ messages | tojson }}"))
    }

    func testReasoningEffortBudgetsMatchLlamaUILevels() {
        XCTAssertNil(ChatStore.reasoningBudget(for: "off"))
        XCTAssertEqual(ChatStore.reasoningBudget(for: "low"), 512)
        XCTAssertEqual(ChatStore.reasoningBudget(for: "medium"), 2_048)
        XCTAssertEqual(ChatStore.reasoningBudget(for: "high"), 8_192)
        XCTAssertNil(ChatStore.reasoningBudget(for: "max"))
    }
}
