// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import CoreText

/// Where a caption lands and how big its text ends up. The exporter and the
/// preview both go through here so what you see is what gets burned in.
enum SubtitleLayout {
    struct Result {
        var box: CGRect
        var pointSize: Double
        var attributed: NSAttributedString
        var textInset: Double
    }

    /// Text is measured, never assumed to fit: the box grows to what the string
    /// needs, and only if it would pass `maxHeightFraction` does the size come
    /// down. A fixed box silently clips the long captions.
    static func layout(text: String, style: SubtitleStyle, frame: CGSize,
                       maxHeightFraction: Double = 0.4) -> Result {
        let width = frame.width * style.maxWidth
        let inset = max(6, frame.height * 0.012)
        let textWidth = max(1, width - inset * 2)
        let maxTextHeight = max(1, frame.height * maxHeightFraction - inset * 2)

        var size = max(style.minPointSize, frame.height * style.relativeSize)
        var attributed = string(text, style: style, pointSize: size)
        var measured = measure(attributed, width: textWidth)
        while measured.height > maxTextHeight, size > style.minPointSize {
            size = max(style.minPointSize, size - 1)
            attributed = string(text, style: style, pointSize: size)
            measured = measure(attributed, width: textWidth)
        }

        // Growing past the cap beats clipping, so the box takes what the text
        // needs and is only pushed back on screen if that overflows the frame.
        let boxHeight = min(frame.height, measured.height + inset * 2)
        let margin = frame.height * style.margin
        let wanted = style.position == .bottom ? margin : frame.height - margin - boxHeight
        let y = min(max(0, wanted), max(0, frame.height - boxHeight))
        return Result(box: CGRect(x: (frame.width - width) / 2, y: y, width: width, height: boxHeight),
                      pointSize: size, attributed: attributed, textInset: inset)
    }

    static func string(_ text: String, style: SubtitleStyle, pointSize: Double) -> NSAttributedString {
        let font = NSFont(name: style.fontName, size: pointSize)
            ?? NSFont.boldSystemFont(ofSize: pointSize)
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .center
        paragraph.lineBreakMode = .byWordWrapping
        var attrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor(cgColor: style.textColor.cgColor) ?? .white,
            .paragraphStyle: paragraph,
        ]
        if style.background == .outline {
            // Negative width strokes and fills, which is what keeps the glyph readable.
            attrs[.strokeColor] = NSColor.black.withAlphaComponent(style.backgroundOpacity)
            attrs[.strokeWidth] = -max(2.0, pointSize * 0.12)
        }
        return NSAttributedString(string: text, attributes: attrs)
    }

    /// The two APIs disagree by a few points on long strings, and the framesetter
    /// is the one that under-reports, so the taller answer wins. Under-measuring
    /// here is what clips the caption.
    private static func measure(_ attributed: NSAttributedString, width: Double) -> CGSize {
        let constraint = CGSize(width: width, height: .greatestFiniteMagnitude)
        let framesetter = CTFramesetterCreateWithAttributedString(attributed)
        let suggested = CTFramesetterSuggestFrameSizeWithConstraints(
            framesetter, CFRange(location: 0, length: 0), nil, constraint, nil)
        let bounds = attributed.boundingRect(
            with: constraint, options: [.usesLineFragmentOrigin, .usesFontLeading])
        return CGSize(width: ceil(max(suggested.width, bounds.width)),
                      height: ceil(max(suggested.height, bounds.height)))
    }
}

extension SubtitleLayout {
    /// The caption drawn over a still, by the same layout the exporter uses, so
    /// the preview cannot drift from the burned-in result.
    static func preview(text: String, style: SubtitleStyle, over frame: NSImage) -> NSImage {
        let size = frame.representations.first.map {
            CGSize(width: $0.pixelsWide, height: $0.pixelsHigh)
        } ?? frame.size
        let out = NSImage(size: size)
        out.lockFocus()
        defer { out.unlockFocus() }
        frame.draw(in: CGRect(origin: .zero, size: size))
        guard let ctx = NSGraphicsContext.current?.cgContext else { return out }

        let l = layout(text: text, style: style, frame: size)
        if style.background == .box {
            ctx.setFillColor(CGColor(gray: 0, alpha: style.backgroundOpacity))
            ctx.addPath(CGPath(roundedRect: l.box, cornerWidth: 8, cornerHeight: 8, transform: nil))
            ctx.fillPath()
        }
        let textRect = l.box.insetBy(dx: l.textInset, dy: l.textInset)
        let framesetter = CTFramesetterCreateWithAttributedString(l.attributed)
        let path = CGPath(rect: textRect, transform: nil)
        let ctFrame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), path, nil)
        CTFrameDraw(ctFrame, ctx)
        return out
    }
}
