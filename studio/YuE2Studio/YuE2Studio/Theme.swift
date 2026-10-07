import SwiftUI

/// Shared visual language for YuE2 Studio.
enum Studio {
    static let accent = Color(red: 0.62, green: 0.35, blue: 1.0)   // violet
    static let accent2 = Color(red: 0.2, green: 0.75, blue: 1.0)   // cyan
    static let accent3 = Color(red: 1.0, green: 0.35, blue: 0.6)   // pink

    static var gradient: LinearGradient {
        LinearGradient(colors: [accent, accent2],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    static var orbGradient: AngularGradient {
        AngularGradient(colors: [accent, accent3, accent2, accent],
                        center: .center)
    }

    /// Deterministic pastel hue from any string — session tiles, tags.
    static func nameColor(_ name: String) -> Color {
        var hash = 5381
        for byte in name.utf8 { hash = (hash &* 33) &+ Int(byte) }
        return Color(hue: Double(hash.magnitude % 360) / 360.0,
                     saturation: 0.65, brightness: 0.85)
    }

    /// Deterministic family color for a stem/track name.
    static func familyColor(_ name: String) -> Color {
        if name.contains("drum") || name.contains("kick") || name.contains("snare")
            || name.contains("hihat") || name.contains("tom") || name.contains("perc")
            || name.contains("cymbal") || name.contains("timpani") {
            return .orange
        }
        if name.contains("vocal") || name.contains("choir") || name.contains("adlib") {
            return .pink
        }
        if name.contains("bass") { return .purple }
        if name.contains("violin") || name.contains("viola") || name.contains("cell")
            || name.contains("contrabass") || name.contains("string") { return .blue }
        if name.contains("flute") || name.contains("oboe") || name.contains("clarinet")
            || name.contains("bassoon") || name.contains("sax")
            || name.contains("piccolo") { return .mint }
        if name.contains("trumpet") || name.contains("horn") || name.contains("trombone")
            || name.contains("tuba") { return .yellow }
        if name.contains("guitar") { return .green }
        if name.contains("synth") || name.contains("pad") || name.contains("lead") {
            return .indigo
        }
        if name.contains("piano") || name.contains("rhodes") || name.contains("organ")
            || name.contains("harp") || name.contains("key") { return .cyan }
        return .gray
    }

    static func icon(_ name: String) -> String {
        if name.contains("drum") || name.contains("kick") || name.contains("snare")
            || name.contains("hihat") || name.contains("tom") || name.contains("perc")
            || name.contains("cymbal") || name.contains("timpani") { return "drum.fill" }
        if name.contains("vocal") || name.contains("choir") || name.contains("adlib") {
            return "mic.fill"
        }
        if name.contains("bass") { return "guitars.fill" }
        if name.contains("violin") || name.contains("viola") || name.contains("cell")
            || name.contains("contrabass") || name.contains("string") {
            return "music.quarternote.3"
        }
        if name.contains("flute") || name.contains("oboe") || name.contains("clarinet")
            || name.contains("bassoon") || name.contains("sax") { return "wind" }
        if name.contains("trumpet") || name.contains("horn") || name.contains("trombone")
            || name.contains("tuba") { return "trumpet.fill" }
        if name.contains("guitar") { return "guitars.fill" }
        if name.contains("synth") || name.contains("pad") || name.contains("lead") {
            return "waveform.path.ecg"
        }
        return "pianokeys"
    }
}

/// Rounded gradient tile holding an SF Symbol.
struct IconTile: View {
    let symbol: String
    var color: Color = Studio.accent
    var size: CGFloat = 34

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: size * 0.42, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(
                LinearGradient(colors: [color, color.opacity(0.55)],
                               startPoint: .topLeading, endPoint: .bottomTrailing),
                in: RoundedRectangle(cornerRadius: size * 0.26, style: .continuous))
            .shadow(color: color.opacity(0.45), radius: 6, y: 2)
    }
}

/// Segmented LED meter — the pro-audio level indicator (green/amber/red).
struct VUMeter: View {
    var level: Double          // 0...1
    var segments = 14
    var width: CGFloat = 8
    var height: CGFloat = 120

    private func segColor(_ idx: Int) -> Color {
        if idx >= segments - 2 { return .red }
        if idx >= segments - 5 { return .orange }
        return .green
    }

    var body: some View {
        let lit = Int((max(0, min(1, level)) * Double(segments)).rounded())
        let segH = max(2, (height - CGFloat(segments - 1) * 2) / CGFloat(segments))
        VStack(spacing: 2) {
            ForEach(0..<segments, id: \.self) { row in
                let idx = segments - 1 - row   // drawn top→bottom
                RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                    .fill(idx < lit ? segColor(idx) : Color.white.opacity(0.07))
                    .frame(width: width, height: segH)
            }
        }
        .animation(.linear(duration: 0.08), value: lit)
    }
}

/// Self-animating meter — plausible signal envelope while a job runs.
struct LiveMeter: View {
    var active = true
    var seed: Double = 0
    var segments = 14
    var height: CGFloat = 120

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: !active)) { context in
            let t = context.date.timeIntervalSinceReferenceDate + seed
            let raw = abs(sin(t * 1.7) * 0.5 + sin(t * 2.9 + 1.3) * 0.3
                          + sin(t * 4.7 + 0.7) * 0.2)
            VUMeter(level: active ? min(1, raw) : 0.08,
                    segments: segments, height: height)
        }
    }
}

/// Thin scanning bar for compact status spots (status bar, export row).
struct ActivityMeter: View {
    var color: Color = .green
    var paused = false
    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: paused)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            let phase = (sin(t * 2.6) + 1) / 2          // 0...1 sweep
            GeometryReader { geo in
                let w = geo.size.width
                ZStack(alignment: .leading) {
                    Capsule().fill(color.opacity(0.18))
                    Capsule().fill(color)
                        .frame(width: max(10, w * 0.3))
                        .offset(x: phase * (w - max(10, w * 0.3)))
                }
            }
            .frame(height: 4)
        }
    }
}

/// Console fader — real drag gesture, tick marks, etched cap, dbl-click reset.
struct FaderView: View {
    @Binding var value: Double
    var range: ClosedRange<Double> = -24...6
    var height: CGFloat = 132
    var tint: Color = .white
    var onCommit: () -> Void = {}
    @State private var start: Double?

    private var span: Double { range.upperBound - range.lowerBound }

    var body: some View {
        let frac = CGFloat((value - range.lowerBound) / span)
        ZStack {
            VStack(spacing: 0) {
                ForEach(0..<9, id: \.self) { i in
                    if i > 0 { Spacer(minLength: 0) }
                    Rectangle().fill(.white.opacity(i == 2 ? 0.4 : 0.16))
                        .frame(width: i == 2 ? 22 : 16, height: 1)
                }
            }
            .frame(height: height)
            Capsule()
                .fill(.black.opacity(0.65))
                .frame(width: 4, height: height)
                .overlay(Capsule().stroke(.white.opacity(0.15), lineWidth: 1)
                    .frame(width: 4, height: height))
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(LinearGradient(colors: [Color(white: 0.95), Color(white: 0.55)],
                                     startPoint: .top, endPoint: .bottom))
                .frame(width: 30, height: 10)
                .overlay(Rectangle().fill(tint).frame(width: 30, height: 1.5))
                .shadow(color: .black.opacity(0.6), radius: 2, y: 1)
                .offset(y: (0.5 - frac) * (height - 10))
        }
        .frame(width: 44, height: height)
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { g in
                if start == nil { start = value }
                let delta = Double(-g.translation.height / height) * span
                value = min(range.upperBound, max(range.lowerBound, (start ?? value) + delta))
            }
            .onEnded { _ in start = nil; onCommit() })
        .onTapGesture(count: 2) { value = 0; onCommit() }
    }
}

/// Rotary knob — vertical drag adjusts across a 270° sweep, dbl-click centers.
struct RotaryKnob: View {
    @Binding var value: Double
    var range: ClosedRange<Double> = -1...1
    var size: CGFloat = 32
    var tint: Color = Studio.accent
    var onCommit: () -> Void = {}
    @State private var start: Double?

    var body: some View {
        let frac = (value - range.lowerBound) / (range.upperBound - range.lowerBound)
        let angle = -135.0 + frac * 270.0
        ZStack {
            Circle()
                .trim(from: 0, to: 0.75)
                .stroke(Color.white.opacity(0.15), lineWidth: 2)
                .rotationEffect(.degrees(135))
            Circle()
                .trim(from: 0, to: max(0.001, frac * 0.75))
                .stroke(tint, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                .rotationEffect(.degrees(135))
            Circle()
                .fill(LinearGradient(colors: [Color(white: 0.3), Color(white: 0.12)],
                                     startPoint: .top, endPoint: .bottom))
                .frame(width: size * 0.72, height: size * 0.72)
                .overlay(Circle().stroke(.white.opacity(0.12), lineWidth: 1)
                    .frame(width: size * 0.72, height: size * 0.72))
                .shadow(color: .black.opacity(0.6), radius: 2, y: 1)
            Capsule()
                .fill(.white)
                .frame(width: 2, height: size * 0.24)
                .offset(y: -size * 0.22)
                .rotationEffect(.degrees(angle))
        }
        .frame(width: size, height: size)
        .contentShape(Circle())
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { g in
                if start == nil { start = value }
                let delta = Double(-g.translation.height)
                    * (range.upperBound - range.lowerBound) / 120
                value = min(range.upperBound, max(range.lowerBound, (start ?? value) + delta))
            }
            .onEnded { _ in start = nil; onCommit() })
        .onTapGesture(count: 2) {
            value = (range.lowerBound + range.upperBound) / 2
            onCommit()
        }
    }
}

/// Slow, dim hero ring — restrained branding for the empty state.
struct PulseOrb: View {
    var size: CGFloat = 88
    var active = false

    var body: some View {
        TimelineView(.animation(paused: !active)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            let breathe = active ? 0.85 + 0.15 * sin(t * 3) : 0.7 + 0.08 * sin(t * 0.9)
            ZStack {
                Circle()
                    .stroke(Studio.orbGradient, lineWidth: 1.5)
                    .frame(width: size, height: size)
                    .opacity(breathe)
                    .rotationEffect(.degrees(t * (active ? 40 : 8)))
                Circle()
                    .stroke(Studio.orbGradient.opacity(0.4), lineWidth: 1)
                    .frame(width: size * 0.78, height: size * 0.78)
                    .rotationEffect(.degrees(-t * (active ? 55 : 12)))
                Image(systemName: "waveform")
                    .font(.system(size: size * 0.26, weight: .light))
                    .foregroundStyle(.white.opacity(0.85))
            }
        }
    }
}

/// Circular progress ring — replaces the orb while a download streams.
/// The displayed value chases the real byte count with exponential
/// smoothing every frame: always glides forward, slows near the target,
/// never jumps and never overshoots.
struct ProgressRing: View {
    var fraction: Double  // 0…1 — the real value
    @State private var shown = 0.0  // eased display value

    private var indeterminate: Bool { fraction <= 0.002 }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            ZStack {
                Circle()
                    .stroke(.white.opacity(0.12), lineWidth: 5)
                if indeterminate {
                    // No bytes yet — a slow travelling arc reads as "working".
                    Circle()
                        .trim(from: 0, to: 0.28)
                        .stroke(Studio.orbGradient,
                                style: StrokeStyle(lineWidth: 5,
                                                   lineCap: .round))
                        .rotationEffect(.degrees(t * 160 - 90))
                    Image(systemName: "waveform")
                        .font(.system(size: 16, weight: .light))
                        .foregroundStyle(.white.opacity(0.7))
                } else {
                    Circle()
                        .trim(from: 0, to: max(0.005, shown))
                        .stroke(Studio.orbGradient,
                                style: StrokeStyle(lineWidth: 5,
                                                   lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    Text("\(Int((shown * 100).rounded()))%")
                        .font(.system(size: 15, weight: .semibold)
                            .monospacedDigit())
                        .foregroundStyle(.white.opacity(0.9))
                        .contentTransition(.numericText())
                }
            }
            .onAppear { shown = fraction }
            .onChange(of: context.date) { _ in
                shown += (fraction - shown) * 0.06
            }
        }
    }
}

/// Static status dot with glow — no perpetual animation while idle.
struct StatusDot: View {
    var on: Bool
    var body: some View {
        Circle()
            .fill(on ? Color.green : Color.red)
            .frame(width: 8, height: 8)
            .shadow(color: (on ? Color.green : Color.red).opacity(0.9),
                    radius: on ? 5 : 1)
    }
}

/// Card container used across views.
struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(12)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.08)))
    }
}

/// Big gradient action button style.
struct PrimaryGradientButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.callout.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 16).padding(.vertical, 8)
            .background(Studio.gradient, in: Capsule())
            .shadow(color: Studio.accent.opacity(0.5), radius: 8, y: 3)
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(duration: 0.2), value: configuration.isPressed)
    }
}
