import Foundation

enum DflashPolicy {
    static func autoEligible(isMoE _: Bool, ncmoe _: Int) -> Bool {
        true
    }

    static func shouldWarn(fractions: [Double]) -> Bool {
        fractions.count(where: { $0 >= 0.95 }) >= 3
    }
}
