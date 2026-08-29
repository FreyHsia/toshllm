// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM
final class MaskEditorTests: XCTestCase {
    func testFitRectCentersAndPreservesAspect() {
        let r = MaskEditorView.fitRect(CGSize(width: 1024, height: 512), in: CGSize(width: 400, height: 400))
        XCTAssertEqual(r.width, 400, accuracy: 0.01)
        XCTAssertEqual(r.height, 200, accuracy: 0.01)
        XCTAssertEqual(r.minY, 100, accuracy: 0.01)
    }

    func testNormaliseMapsCorners() {
        let rect = CGRect(x: 50, y: 20, width: 200, height: 100)
        XCTAssertEqual(MaskEditorView.normalise(CGPoint(x: 50, y: 20), in: rect), .zero)
        let br = MaskEditorView.normalise(CGPoint(x: 250, y: 120), in: rect)
        XCTAssertEqual(br.x, 1, accuracy: 0.001)
        XCTAssertEqual(br.y, 1, accuracy: 0.001)
    }

    /// A centre stroke must come out white in the middle and black at the edges,
    /// which is the polarity the engine expects.
    func testRenderPolarityAndOrientation() throws {
        let stroke = MaskStroke(points: [CGPoint(x: 0.5, y: 0.25)], radius: 0.1, erases: false)
        let data = try XCTUnwrap(MaskEditorView.render(strokes: [stroke], size: CGSize(width: 128, height: 128)))
        let rep = try XCTUnwrap(NSBitmapImageRep(data: data))
        XCTAssertEqual(rep.pixelsWide, 128)
        // Painted at a quarter down the image: white there, black at the bottom.
        XCTAssertGreaterThan(try XCTUnwrap(rep.colorAt(x: 64, y: 32)).whiteComponent, 0.9)
        XCTAssertLessThan(try XCTUnwrap(rep.colorAt(x: 64, y: 120)).whiteComponent, 0.1)
        XCTAssertLessThan(try XCTUnwrap(rep.colorAt(x: 4, y: 4)).whiteComponent, 0.1)
    }

    func testEraserCutsBackToBlack() throws {
        let paint = MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], radius: 0.3, erases: false)
        let erase = MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], radius: 0.15, erases: true)
        let data = try XCTUnwrap(MaskEditorView.render(strokes: [paint, erase], size: CGSize(width: 128, height: 128)))
        let rep = try XCTUnwrap(NSBitmapImageRep(data: data))
        XCTAssertLessThan(try XCTUnwrap(rep.colorAt(x: 64, y: 64)).whiteComponent, 0.1)
        XCTAssertGreaterThan(try XCTUnwrap(rep.colorAt(x: 64, y: 36)).whiteComponent, 0.9)
    }
}
