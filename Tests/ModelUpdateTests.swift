import XCTest
@testable import ToshLLM

@MainActor
final class ModelUpdateTests: XCTestCase {
    private func compare(digest: String?, size: Int64?, remoteSHA: String?, remoteSize: Int64?) -> ModelUpdateState {
        ModelUpdateChecker.compare(localDigest: digest, localSize: size,
                                   remote: HFFileMetadata(sha256: remoteSHA, sizeBytes: remoteSize))
    }

    func testDigestDecidesWhenBothSidesHaveOne() {
        XCTAssertEqual(compare(digest: "ABC", size: 10, remoteSHA: "abc", remoteSize: 10), .upToDate)
        XCTAssertEqual(compare(digest: "abc", size: 10, remoteSHA: "def", remoteSize: 12),
                       .available(sizeBytes: 12))
        // A re-upload of the same size is only visible through the digest.
        XCTAssertEqual(compare(digest: "abc", size: 10, remoteSHA: "def", remoteSize: 10),
                       .available(sizeBytes: 10))
    }

    func testSizeIsTheFallbackWithoutARecordedDigest() {
        XCTAssertEqual(compare(digest: nil, size: 10, remoteSHA: "abc", remoteSize: 12),
                       .available(sizeBytes: 12))
        XCTAssertEqual(compare(digest: nil, size: 10, remoteSHA: "abc", remoteSize: 10), .unknown)
        XCTAssertEqual(compare(digest: nil, size: 10, remoteSHA: nil, remoteSize: nil), .unknown)
    }

    func testUnknownRatherThanFalseAlarmWhenSizesCannotBeCompared() {
        // Split models report every shard together, so their size is not passed in.
        XCTAssertEqual(compare(digest: nil, size: nil, remoteSHA: "abc", remoteSize: 12), .unknown)
        XCTAssertEqual(compare(digest: "abc", size: nil, remoteSHA: "abc", remoteSize: 12), .upToDate)
    }

    func testNonHuggingFaceSourcesAreSkipped() async {
        let metadata = await HuggingFaceAPI.fileMetadata(for: URL(string: "https://example.com/model.gguf")!)
        XCTAssertNil(metadata)
    }

    func testDigestSurvivesARoundTripThroughDefaults() {
        let file = "round-trip-\(UUID().uuidString).gguf"
        ModelStore.recordDigest("ABCDEF", forFile: file)
        XCTAssertEqual(ModelStore.digest(forFile: file), "abcdef")
        var map = UserDefaults.standard.dictionary(forKey: SettingsKeys.modelDigest) as? [String: String] ?? [:]
        map[file] = nil
        UserDefaults.standard.set(map, forKey: SettingsKeys.modelDigest)
    }
}
