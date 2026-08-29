// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class SubtitleLayoutTests: XCTestCase {
    private let frame = CGSize(width: 1920, height: 1080)

    /// The old exporter used a fixed box, so a long caption was clipped. The box
    /// has to grow with the text instead.
    func testLongCaptionGetsATallerBoxThanAShortOne() {
        let short = SubtitleLayout.layout(text: "Hola.", style: .default, frame: frame)
        let long = SubtitleLayout.layout(
            text: String(repeating: "Una frase larga que tiene que partirse en varias líneas. ", count: 4),
            style: .default, frame: frame)
        XCTAssertGreaterThan(long.box.height, short.box.height * 2)
    }

    /// Nothing may be clipped: the text must fit inside its own box at every length.
    func testTextAlwaysFitsInsideItsBox() {
        for count in [1, 3, 8, 20, 60] {
            let text = String(repeating: "palabra ", count: count * 3)
            let l = SubtitleLayout.layout(text: text, style: .default, frame: frame)
            let inner = l.box.insetBy(dx: l.textInset, dy: l.textInset)
            let measured = l.attributed.boundingRect(
                with: CGSize(width: inner.width, height: .greatestFiniteMagnitude),
                options: [.usesLineFragmentOrigin, .usesFontLeading])
            XCTAssertLessThanOrEqual(ceil(measured.height), ceil(inner.height) + 1,
                                     "clipped at \(count) chunks")
        }
    }

    func testVeryLongCaptionShrinksRatherThanOverflow() {
        let text = String(repeating: "texto ", count: 400)
        let l = SubtitleLayout.layout(text: text, style: .default, frame: frame)
        XCTAssertLessThan(l.pointSize, frame.height * SubtitleStyle.default.relativeSize)
        XCTAssertGreaterThanOrEqual(l.pointSize, SubtitleStyle.default.minPointSize)
        // Past the minimum size the box grows rather than clip, but it stays on screen.
        XCTAssertLessThanOrEqual(l.box.maxY, frame.height)
        XCTAssertGreaterThanOrEqual(l.box.minY, 0)
    }

    func testPositionAndMarginAreHonoured() {
        var top = SubtitleStyle.default
        top.position = .top
        let t = SubtitleLayout.layout(text: "Arriba", style: top, frame: frame)
        let b = SubtitleLayout.layout(text: "Abajo", style: .default, frame: frame)
        XCTAssertGreaterThan(t.box.minY, b.box.minY)
        XCTAssertEqual(b.box.minY, frame.height * SubtitleStyle.default.margin, accuracy: 0.5)
    }

    func testStyleRoundTripsThroughItsPersistence() throws {
        var s = SubtitleStyle.default
        s.fontName = "Georgia-Bold"; s.relativeSize = 0.055; s.background = .outline
        let data = try JSONEncoder().encode(s)
        XCTAssertEqual(try JSONDecoder().decode(SubtitleStyle.self, from: data), s)
    }
}
