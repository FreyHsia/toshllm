// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Renders the installer window background. Same gradient and glyph as
// AppIcon.icon, so the DMG and the app read as one thing.
//   swiftc -O scripts/dmg-background.swift -o /tmp/dmgbg && /tmp/dmgbg out.png 2

import AppKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: dmgbg <out.png> <glyph.png> [scale]\n".data(using: .utf8)!)
    exit(2)
}
let outPath = args[1], glyphPath = args[2]
let scale = args.count > 3 ? (Double(args[3]) ?? 1) : 1

let W = 640.0, H = 400.0
// icon centres; the cards below also have to hold the name Finder draws under them
let appSlot = CGPoint(x: 165, y: 206)
let dstSlot = CGPoint(x: 475, y: 206)
let cardCentreY = 186.0

func srgb(_ r: Double, _ g: Double, _ b: Double, _ a: Double = 1) -> NSColor {
    NSColor(srgbRed: r, green: g, blue: b, alpha: a)
}

guard let ctx = CGContext(data: nil, width: Int(W * scale), height: Int(H * scale),
                          bitsPerComponent: 8, bytesPerRow: 0,
                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { exit(1) }
ctx.scaleBy(x: scale, y: scale)
NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)

// the icon's two colors, deepened so white text and icons carry
let deep = srgb(0.212, 0.192, 0.318)
let plum = srgb(0.478, 0.180, 0.322)
NSGradient(colors: [plum, deep])!.draw(in: NSRect(x: 0, y: 0, width: W, height: H), angle: 58)

/// Radial wash with a real falloff: `drawRadialGradient` fades to clear instead
/// of filling a path, which would leave a visible disc edge.
func wash(_ center: CGPoint, radius: CGFloat, color: NSColor, alpha: CGFloat) {
    let colors = [color.withAlphaComponent(alpha).cgColor,
                  color.withAlphaComponent(0).cgColor] as CFArray
    guard let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                             colors: colors, locations: [0, 1]) else { return }
    ctx.drawRadialGradient(g, startCenter: center, startRadius: 0,
                           endCenter: center, endRadius: radius, options: [])
}

// aurora: the brand colors bleeding in from opposite corners
wash(CGPoint(x: -20, y: H + 40), radius: 380, color: srgb(0.902, 0.373, 0.576), alpha: 0.70)
wash(CGPoint(x: W + 40, y: -40), radius: 380, color: srgb(0.267, 0.251, 0.478), alpha: 0.75)

// brand glyph, cropped into the corner furthest from the title and the icons
if let glyph = NSImage(contentsOfFile: glyphPath) {
    let side = 260.0
    glyph.draw(in: NSRect(x: -side * 0.44, y: -side * 0.56, width: side, height: side),
               from: .zero, operation: .sourceOver, fraction: 0.05)
}

// light pooled under each icon, brighter under the one being dragged
wash(appSlot, radius: 122, color: .white, alpha: 0.16)
wash(dstSlot, radius: 122, color: .white, alpha: 0.09)

// cards give the two drop points a deliberate place to be
func plate(_ c: CGPoint, fill: CGFloat, stroke: CGFloat, dashed: Bool = false) {
    let side = 156.0, height = 180.0
    let r = NSBezierPath(roundedRect: NSRect(x: c.x - side / 2, y: cardCentreY - height / 2,
                                             width: side, height: height),
                         xRadius: 26, yRadius: 26)
    NSColor(white: 1, alpha: fill).setFill()
    r.fill()
    if dashed { r.setLineDash([7, 6], count: 2, phase: 0) }
    r.lineWidth = 1.5
    NSColor(white: 1, alpha: stroke).setStroke()
    r.stroke()
}
plate(appSlot, fill: 0.09, stroke: 0.20)
plate(dstSlot, fill: 0.05, stroke: 0.22, dashed: true)

// vignette
let vignette = [NSColor(white: 0, alpha: 0).cgColor, NSColor(white: 0, alpha: 0.30).cgColor] as CFArray
if let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                      colors: vignette, locations: [0.55, 1]) {
    ctx.drawRadialGradient(g, startCenter: CGPoint(x: W / 2, y: H / 2), startRadius: 0,
                           endCenter: CGPoint(x: W / 2, y: H / 2), endRadius: 430, options: [])
}

// the arrow explains the gesture without a sentence in any one language
let midY = appSlot.y
let from = NSPoint(x: appSlot.x + 98, y: midY), tip = NSPoint(x: dstSlot.x - 98, y: midY)
NSColor(white: 1, alpha: 0.62).setStroke()
let shaft = NSBezierPath()
shaft.move(to: from)
shaft.line(to: NSPoint(x: tip.x - 6, y: midY))
shaft.lineWidth = 2.5
shaft.lineCapStyle = .round
shaft.stroke()
let head = NSBezierPath()
head.move(to: NSPoint(x: tip.x - 14, y: midY + 9))
head.line(to: tip)
head.line(to: NSPoint(x: tip.x - 14, y: midY - 9))
head.lineWidth = 2.5
head.lineCapStyle = .round
head.lineJoinStyle = .round
head.stroke()

func draw(_ s: String, size: CGFloat, weight: NSFont.Weight, alpha: CGFloat, y: CGFloat,
          tracking: CGFloat = 0, shadow: Bool = false) {
    var attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: size, weight: weight),
        .foregroundColor: NSColor(white: 1, alpha: alpha),
        .kern: tracking,
    ]
    if shadow {
        let sh = NSShadow()
        sh.shadowColor = NSColor(white: 0, alpha: 0.28)
        sh.shadowBlurRadius = 12
        sh.shadowOffset = NSSize(width: 0, height: -1)
        attrs[.shadow] = sh
    }
    let str = NSAttributedString(string: s, attributes: attrs)
    str.draw(at: NSPoint(x: (W - str.size().width) / 2, y: y))
}

draw("ToshLLM", size: 40, weight: .semibold, alpha: 0.98, y: H - 96, tracking: 0.4, shadow: true)
draw("Run large language models locally on your Mac", size: 12.5, weight: .regular,
     alpha: 0.70, y: H - 114, tracking: 0.2)
draw("Drag ToshLLM onto Applications", size: 11.5, weight: .medium, alpha: 0.55, y: 48,
     tracking: 0.3)

guard let image = ctx.makeImage(),
      let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
else { exit(1) }
try! png.write(to: URL(fileURLWithPath: outPath))
