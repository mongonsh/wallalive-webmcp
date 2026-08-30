import AppKit
import AVFoundation
import CoreVideo
import Foundation

struct Manifest: Decodable {
  let fps: Int
  let width: Int
  let height: Int
  let frameCount: Int
}

guard CommandLine.arguments.count == 7 else {
  fputs("Usage: assemble-demo.swift FRAMES MANIFEST CARD VOICE SILENT_OUTPUT FINAL_OUTPUT\n", stderr)
  exit(2)
}

let framesDirectory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let manifestURL = URL(fileURLWithPath: CommandLine.arguments[2])
let cardURL = URL(fileURLWithPath: CommandLine.arguments[3])
let voiceURL = URL(fileURLWithPath: CommandLine.arguments[4])
let silentURL = URL(fileURLWithPath: CommandLine.arguments[5])
let finalURL = URL(fileURLWithPath: CommandLine.arguments[6])
let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
let fileManager = FileManager.default
try? fileManager.removeItem(at: silentURL)
try? fileManager.removeItem(at: finalURL)

let writer = try AVAssetWriter(outputURL: silentURL, fileType: .mp4)
let settings: [String: Any] = [
  AVVideoCodecKey: AVVideoCodecType.h264,
  AVVideoWidthKey: manifest.width,
  AVVideoHeightKey: manifest.height,
  AVVideoCompressionPropertiesKey: [
    AVVideoAverageBitRateKey: 6_000_000,
    AVVideoExpectedSourceFrameRateKey: manifest.fps,
    AVVideoMaxKeyFrameIntervalKey: manifest.fps * 2,
  ],
]
let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
input.expectsMediaDataInRealTime = false
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
  assetWriterInput: input,
  sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: manifest.width,
    kCVPixelBufferHeightKey as String: manifest.height,
  ]
)
guard writer.canAdd(input) else { throw NSError(domain: "WallAliveVideo", code: 1) }
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)

func pixelBuffer(from url: URL) throws -> CVPixelBuffer {
  guard let image = NSImage(contentsOf: url),
        let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    throw NSError(domain: "WallAliveVideo", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not read \(url.path)"])
  }
  var optionalBuffer: CVPixelBuffer?
  let status = CVPixelBufferCreate(
    kCFAllocatorDefault,
    manifest.width,
    manifest.height,
    kCVPixelFormatType_32BGRA,
    [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
    &optionalBuffer
  )
  guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
    throw NSError(domain: "WallAliveVideo", code: 3)
  }
  CVPixelBufferLockBaseAddress(buffer, [])
  defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
  guard let context = CGContext(
    data: CVPixelBufferGetBaseAddress(buffer),
    width: manifest.width,
    height: manifest.height,
    bitsPerComponent: 8,
    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
  ) else { throw NSError(domain: "WallAliveVideo", code: 4) }
  context.setFillColor(NSColor(calibratedRed: 0.969, green: 0.941, blue: 0.875, alpha: 1).cgColor)
  context.fill(CGRect(x: 0, y: 0, width: manifest.width, height: manifest.height))
  let sourceRatio = CGFloat(cgImage.width) / CGFloat(cgImage.height)
  let targetRatio = CGFloat(manifest.width) / CGFloat(manifest.height)
  let drawRect: CGRect
  if sourceRatio > targetRatio {
    let drawWidth = CGFloat(manifest.height) * sourceRatio
    drawRect = CGRect(x: (CGFloat(manifest.width) - drawWidth) / 2, y: 0, width: drawWidth, height: CGFloat(manifest.height))
  } else {
    let drawHeight = CGFloat(manifest.width) / sourceRatio
    drawRect = CGRect(x: 0, y: (CGFloat(manifest.height) - drawHeight) / 2, width: CGFloat(manifest.width), height: drawHeight)
  }
  context.translateBy(x: 0, y: CGFloat(manifest.height))
  context.scaleBy(x: 1, y: -1)
  context.draw(cgImage, in: drawRect)
  return buffer
}

func append(_ url: URL, at index: Int) throws {
  while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.003) }
  let time = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(manifest.fps))
  guard adaptor.append(try pixelBuffer(from: url), withPresentationTime: time) else {
    throw writer.error ?? NSError(domain: "WallAliveVideo", code: 5)
  }
}

let cardFrames = manifest.fps * 3
var outputIndex = 0
for _ in 0..<cardFrames { try append(cardURL, at: outputIndex); outputIndex += 1 }
for frameIndex in 0..<manifest.frameCount {
  let filename = String(format: "frame-%05d.jpg", frameIndex)
  try append(framesDirectory.appendingPathComponent(filename), at: outputIndex)
  outputIndex += 1
}
for _ in 0..<cardFrames { try append(cardURL, at: outputIndex); outputIndex += 1 }

input.markAsFinished()
let writerSemaphore = DispatchSemaphore(value: 0)
writer.finishWriting { writerSemaphore.signal() }
writerSemaphore.wait()
guard writer.status == .completed else { throw writer.error ?? NSError(domain: "WallAliveVideo", code: 6) }

let composition = AVMutableComposition()
let videoAsset = AVURLAsset(url: silentURL)
let voiceAsset = AVURLAsset(url: voiceURL)
guard let sourceVideo = videoAsset.tracks(withMediaType: .video).first,
      let outputVideo = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
  throw NSError(domain: "WallAliveVideo", code: 7)
}
try outputVideo.insertTimeRange(CMTimeRange(start: .zero, duration: videoAsset.duration), of: sourceVideo, at: .zero)
outputVideo.preferredTransform = sourceVideo.preferredTransform
if let sourceAudio = voiceAsset.tracks(withMediaType: .audio).first,
   let outputAudio = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
  let audioDuration = CMTimeMinimum(voiceAsset.duration, videoAsset.duration)
  try outputAudio.insertTimeRange(CMTimeRange(start: .zero, duration: audioDuration), of: sourceAudio, at: .zero)
}

guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
  throw NSError(domain: "WallAliveVideo", code: 8)
}
exporter.outputURL = finalURL
exporter.outputFileType = .mp4
exporter.shouldOptimizeForNetworkUse = true
let exportSemaphore = DispatchSemaphore(value: 0)
exporter.exportAsynchronously { exportSemaphore.signal() }
exportSemaphore.wait()
guard exporter.status == .completed else { throw exporter.error ?? NSError(domain: "WallAliveVideo", code: 9) }
print("Created \(finalURL.path) with \(outputIndex) frames at \(manifest.fps) fps")
