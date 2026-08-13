// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Renders the installer window background. The cinematic artwork and the
// app's brand details are composed here so typography stays pixel-perfect.
//   swiftc -O scripts/dmg-background.swift -o /tmp/dmgbg
//   /tmp/dmgbg out.png Assets/dmg/obsidian-hardware.png AppIcon.icon/Assets/glyph.png 2

import AppKit

let args = CommandLine.arguments
guard args.count >= 4 else {
    FileHandle.standardError.write(
        "usage: dmgbg <out.png> <artwork.png> <glyph.png> [scale]\n".data(using: .utf8)!
    )
    exit(2)
}
let outPath = args[1], artworkPath = args[2], glyphPath = args[3]
let scale = args.count > 4 ? (Double(args[4]) ?? 1) : 1

let W = 640.0, H = 400.0
// These centres match the Finder positions saved in Assets/dmg/DS_Store.
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
let canvas = NSRect(x: 0, y: 0, width: W, height: H)

// A safe fallback fill means corrupt or missing art can never yield transparency.
NSGradient(colors: [srgb(0.09, 0.04, 0.12), srgb(0.12, 0.08, 0.18)])!
    .draw(in: canvas, angle: 42)
if let artwork = NSImage(contentsOfFile: artworkPath) {
    artwork.draw(in: canvas, from: .zero, operation: .sourceOver, fraction: 0.94)
}

// A controlled veil keeps Finder labels and the title readable over the art.
NSGradient(colorsAndLocations:
    (srgb(0.02, 0.01, 0.04, 0.66), 0.0),
    (srgb(0.04, 0.02, 0.07, 0.12), 0.48),
    (srgb(0.02, 0.01, 0.04, 0.58), 1.0)
)!.draw(in: canvas, angle: -90)

func wash(_ center: CGPoint, radius: CGFloat, color: NSColor, alpha: CGFloat) {
    let colors = [color.withAlphaComponent(alpha).cgColor,
                  color.withAlphaComponent(0).cgColor] as CFArray
    guard let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    colors: colors, locations: [0, 1]) else { return }
    ctx.drawRadialGradient(gradient, startCenter: center, startRadius: 0,
                           endCenter: center, endRadius: radius, options: [])
}

// Brand light is restrained: the machined hardware artwork remains the hero.
wash(CGPoint(x: W / 2, y: 150), radius: 285,
     color: srgb(0.76, 0.30, 0.72), alpha: 0.08)
wash(CGPoint(x: appSlot.x, y: appSlot.y + 2), radius: 112,
     color: srgb(0.90, 0.37, 0.58), alpha: 0.11)
wash(CGPoint(x: dstSlot.x, y: dstSlot.y + 2), radius: 112,
     color: srgb(0.58, 0.48, 1.0), alpha: 0.09)

// A quiet watermark ties the installer to the app icon without adding clutter.
if let glyph = NSImage(contentsOfFile: glyphPath) {
    let side = 210.0
    glyph.draw(in: NSRect(x: -side * 0.48, y: -side * 0.58, width: side, height: side),
               from: .zero, operation: .sourceOver, fraction: 0.035)
}

// Glass landing zones make the gesture obvious without competing with Finder.
func plate(_ center: CGPoint, accent: NSColor, dashed: Bool = false) {
    let side = 154.0, height = 166.0
    let rect = NSRect(x: center.x - side / 2, y: cardCentreY - height / 2,
                      width: side, height: height)

    let shadow = NSShadow()
    shadow.shadowColor = NSColor(white: 0, alpha: 0.28)
    shadow.shadowBlurRadius = 16
    shadow.shadowOffset = NSSize(width: 0, height: -5)
    NSGraphicsContext.saveGraphicsState()
    shadow.set()
    NSColor(white: 0.03, alpha: 0.24).setFill()
    NSBezierPath(roundedRect: rect, xRadius: 28, yRadius: 28).fill()
    NSGraphicsContext.restoreGraphicsState()

    let outline = NSBezierPath(roundedRect: rect, xRadius: 28, yRadius: 28)
    NSGradient(colors: [NSColor(white: 1, alpha: 0.065),
                        accent.withAlphaComponent(0.025)])!.draw(in: outline, angle: -72)
    NSColor(white: 1, alpha: 0.018).setFill()
    outline.fill()
    if dashed { outline.setLineDash([5, 5], count: 2, phase: 0) }
    outline.lineWidth = dashed ? 1.1 : 0.8
    (dashed ? accent.withAlphaComponent(0.34) : NSColor(white: 1, alpha: 0.13)).setStroke()
    outline.stroke()

    let highlight = NSBezierPath()
    highlight.move(to: NSPoint(x: rect.minX + 32, y: rect.maxY - 0.75))
    highlight.line(to: NSPoint(x: rect.maxX - 32, y: rect.maxY - 0.75))
    highlight.lineWidth = 0.7
    NSColor(white: 1, alpha: dashed ? 0.055 : 0.14).setStroke()
    highlight.stroke()

    // Finder chooses the item-label colour and currently renders it black.
    // Give that system-owned text a quiet light nameplate so it stays readable
    // on every part of the dark artwork.
    let captionRect = NSRect(x: center.x - 59, y: rect.minY + 10,
                             width: 118, height: 24)
    let caption = NSBezierPath(roundedRect: captionRect, xRadius: 12, yRadius: 12)
    NSGradient(colors: [NSColor(white: 1, alpha: 0.72),
                        NSColor(white: 1, alpha: 0.50)])!.draw(in: caption, angle: -90)
    caption.lineWidth = 0.6
    NSColor(white: 1, alpha: 0.48).setStroke()
    caption.stroke()
}
plate(appSlot, accent: srgb(0.90, 0.37, 0.58))
plate(dstSlot, accent: srgb(0.69, 0.60, 1.0), dashed: true)

// A dark edge vignette makes the centre feel illuminated and dimensional.
let vignette = [NSColor(white: 0, alpha: 0).cgColor,
                NSColor(white: 0, alpha: 0.48).cgColor] as CFArray
if let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                             colors: vignette, locations: [0.55, 1]) {
    ctx.drawRadialGradient(gradient, startCenter: CGPoint(x: W / 2, y: H / 2), startRadius: 0,
                           endCenter: CGPoint(x: W / 2, y: H / 2), endRadius: 430, options: [])
}

// The luminous arrow echoes the neural ribbon and clearly explains the gesture.
let midY = appSlot.y + 1
let from = NSPoint(x: appSlot.x + 100, y: midY)
let tip = NSPoint(x: dstSlot.x - 100, y: midY)
let arrowGlow = NSShadow()
arrowGlow.shadowColor = srgb(0.86, 0.62, 1.0, 0.52)
arrowGlow.shadowBlurRadius = 7
arrowGlow.shadowOffset = .zero
NSGraphicsContext.saveGraphicsState()
arrowGlow.set()
srgb(0.93, 0.89, 1.0, 0.82).setStroke()
let shaft = NSBezierPath()
shaft.move(to: from)
shaft.line(to: NSPoint(x: tip.x - 6, y: midY))
shaft.lineWidth = 1.9
shaft.lineCapStyle = .round
shaft.stroke()
let head = NSBezierPath()
head.move(to: NSPoint(x: tip.x - 14, y: midY + 9))
head.line(to: tip)
head.line(to: NSPoint(x: tip.x - 14, y: midY - 9))
head.lineWidth = 1.9
head.lineCapStyle = .round
head.lineJoinStyle = .round
head.stroke()
NSGraphicsContext.restoreGraphicsState()

func draw(_ string: String, size: CGFloat, weight: NSFont.Weight, alpha: CGFloat, y: CGFloat,
          tracking: CGFloat = 0, shadow: Bool = false) {
    var attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: size, weight: weight),
        .foregroundColor: NSColor(white: 1, alpha: alpha),
        .kern: tracking,
    ]
    if shadow {
        let textShadow = NSShadow()
        textShadow.shadowColor = NSColor(white: 0, alpha: 0.62)
        textShadow.shadowBlurRadius = 14
        textShadow.shadowOffset = NSSize(width: 0, height: -2)
        attributes[.shadow] = textShadow
    }
    let text = NSAttributedString(string: string, attributes: attributes)
    text.draw(at: NSPoint(x: (W - text.size().width) / 2, y: y))
}

draw("ToshLLM", size: 36, weight: .semibold, alpha: 0.99,
     y: H - 78, tracking: 0.2, shadow: true)
draw("PRIVATE AI. NATIVE SPEED.", size: 9.5, weight: .semibold, alpha: 0.66,
     y: H - 97, tracking: 2.1)
draw("Drag to install", size: 11.5, weight: .medium, alpha: 0.70,
     y: 36, tracking: 0.55, shadow: true)

guard let image = ctx.makeImage(),
      let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
else { exit(1) }
try png.write(to: URL(fileURLWithPath: outPath))
