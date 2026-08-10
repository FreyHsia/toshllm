// Can a compute kernel read a buffer that lives on another GPU of the same peer group?
// Metal documents remote buffer views for blit copies; if a shader can read one directly, the
// tensor-split allreduce can fuse its copy and its add into a single kernel.
//
//   swift scripts/peer-compute-probe.swift          two GPUs of one peer group (W6800X/Vega II Duo)
//   swift scripts/peer-compute-probe.swift --self   one GPU, checks the harness itself
//
// Prints PASS or FAIL and touches nothing else.

import Metal
import Foundation

let selfMode = CommandLine.arguments.contains("--self")
let n = 1 << 16
let bytes = n * MemoryLayout<Float>.size

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void add_inplace(device float * dst [[buffer(0)]],
                        device const float * src [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    dst[tid] += src[tid];
}
"""

let devices = MTLCopyAllDevices()
for d in devices {
    print("device: \(d.name) peerGroupID=\(d.peerGroupID) peerIndex=\(d.peerIndex) peerCount=\(d.peerCount)")
}

var devA: MTLDevice?
var devB: MTLDevice?
if selfMode {
    devA = devices.first
    devB = devices.first
} else {
    outer: for a in devices {
        guard a.peerGroupID != 0 else { continue }
        for b in devices where b !== a && b.peerGroupID == a.peerGroupID {
            devA = a; devB = b
            break outer
        }
    }
}

guard let a = devA, let b = devB else {
    print("FAIL: no two devices in a peer group (run with --self to check the harness)")
    exit(1)
}
print("source GPU: \(a.name)\ntarget GPU: \(b.name)")

// Both buffers private, which is what a remote view requires.
let bufA = a.makeBuffer(length: bytes, options: .storageModePrivate)!
let bufB = b.makeBuffer(length: bytes, options: .storageModePrivate)!

func fill(_ dev: MTLDevice, _ queue: MTLCommandQueue, _ dst: MTLBuffer, _ value: Float) {
    var host = [Float](repeating: value, count: n)
    let stage = dev.makeBuffer(bytes: &host, length: bytes, options: .storageModeShared)!
    let cmd = queue.makeCommandBuffer()!
    let blit = cmd.makeBlitCommandEncoder()!
    blit.copy(from: stage, sourceOffset: 0, to: dst, destinationOffset: 0, size: bytes)
    blit.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

let queueA = a.makeCommandQueue()!
let queueB = b.makeCommandQueue()!
fill(a, queueA, bufA, 1.0)
fill(b, queueB, bufB, 2.0)

// The question: a view of A's buffer, bound as a kernel argument on B.
var remote: MTLBuffer?
if selfMode {
    remote = bufA
} else {
    remote = bufA.makeRemoteBufferView(b)
    if remote == nil {
        print("FAIL: newRemoteBufferViewForDevice returned nil")
        exit(1)
    }
    print("remote view created, remoteStorageBuffer=\(remote!.remoteStorageBuffer != nil)")
}

let lib = try! b.makeLibrary(source: source, options: nil)
let pipe = try! b.makeComputePipelineState(function: lib.makeFunction(name: "add_inplace")!)

let cmd = queueB.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipe)
enc.setBuffer(bufB, offset: 0, index: 0)
enc.setBuffer(remote!, offset: 0, index: 1)
enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if let error = cmd.error {
    print("FAIL: command buffer error: \(error)")
    exit(1)
}

let readback = b.makeBuffer(length: bytes, options: .storageModeShared)!
let cmdR = queueB.makeCommandBuffer()!
let blit = cmdR.makeBlitCommandEncoder()!
blit.copy(from: bufB, sourceOffset: 0, to: readback, destinationOffset: 0, size: bytes)
blit.endEncoding()
cmdR.commit()
cmdR.waitUntilCompleted()

let out = readback.contents().bindMemory(to: Float.self, capacity: n)
var bad = 0
for i in 0..<n where out[i] != 3.0 {
    if bad < 4 { print("  [\(i)] = \(out[i]), expected 3.0") }
    bad += 1
}

if bad == 0 {
    print("PASS: the kernel read the \(selfMode ? "local" : "remote") buffer, \(n) values correct")
} else {
    print("FAIL: \(bad)/\(n) values wrong")
    exit(1)
}
