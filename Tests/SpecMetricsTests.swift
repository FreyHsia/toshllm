// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class SpecMetricsTests: XCTestCase {
    /// Verbatim from llama-server 84e908c62 running a 4B with an MTP head.
    private let sample = """
    # HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
    # TYPE llamacpp:prompt_tokens_total counter
    llamacpp:prompt_tokens_total 44
    # HELP llamacpp:spec_decode_num_draft_tokens_total Total draft tokens generated
    # TYPE llamacpp:spec_decode_num_draft_tokens_total counter
    llamacpp:spec_decode_num_draft_tokens_total 337
    # HELP llamacpp:spec_decode_num_accepted_tokens_total Total draft tokens accepted by the target model
    # TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
    llamacpp:spec_decode_num_accepted_tokens_total 259
    # HELP llamacpp:spec_decode_num_drafts_total Total speculative decoding verification steps
    # TYPE llamacpp:spec_decode_num_drafts_total counter
    llamacpp:spec_decode_num_drafts_total 113
    # HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position
    # TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
    llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 105
    llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 83
    llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 71
    """

    func testParsesCountersAndPerPositionBreakdown() {
        let m = SpecDecodeMetrics.parse(sample)
        XCTAssertTrue(m.ran)
        XCTAssertEqual(m.draftTokens, 337)
        XCTAssertEqual(m.acceptedTokens, 259)
        XCTAssertEqual(m.drafts, 113)
        XCTAssertEqual(m.acceptedPerPosition, [105, 83, 71])
        XCTAssertEqual(m.acceptance ?? 0, 259.0 / 337.0, accuracy: 1e-9)
        XCTAssertEqual(m.meanAccepted ?? 0, 259.0 / 113.0, accuracy: 1e-9)
        XCTAssertEqual(m.acceptance(atPosition: 0) ?? 0, 105.0 / 113.0, accuracy: 1e-9)
        XCTAssertEqual(m.acceptance(atPosition: 2) ?? 0, 71.0 / 113.0, accuracy: 1e-9)
        XCTAssertNil(m.acceptance(atPosition: 3))
    }

    /// Speculation off: upstream still exposes the counters, all at zero.
    func testCountersAtZeroDoNotCountAsRun() {
        let m = SpecDecodeMetrics.parse("""
        llamacpp:spec_decode_num_draft_tokens_total 0
        llamacpp:spec_decode_num_accepted_tokens_total 0
        llamacpp:spec_decode_num_drafts_total 0
        """)
        XCTAssertFalse(m.ran)
        XCTAssertNil(m.acceptance)
        XCTAssertNil(m.acceptance(atPosition: 0))
    }

    func testSkipsCommentsAndUnrelatedMetrics() {
        let m = SpecDecodeMetrics.parse("""
        # llamacpp:spec_decode_num_drafts_total 999
        llamacpp:kv_cache_usage_ratio 0.5
        llamacpp:spec_decode_num_drafts_total 7
        """)
        XCTAssertEqual(m.drafts, 7)
        XCTAssertEqual(m.draftTokens, 0)
    }

    /// A gap in the position labels must not shift the later positions.
    func testMissingPositionKeepsIndexAlignment() {
        let m = SpecDecodeMetrics.parse("""
        llamacpp:spec_decode_num_drafts_total 10
        llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 9
        llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 4
        """)
        XCTAssertEqual(m.acceptedPerPosition, [9, 0, 4])
    }

    func testGarbageIsIgnoredRatherThanCrashing() {
        let m = SpecDecodeMetrics.parse("llamacpp:spec_decode_num_drafts_total notanumber\n\n   \n")
        XCTAssertEqual(m.drafts, 0)
        XCTAssertFalse(m.ran)
    }
}
