const std = @import("std");
const math = std.math;
const mem = std.mem;
const fmt = std.fmt;

pub const Colors = @import("predefined.zig").Colors;

pub const ColorError = error{
    InvalidHexLength,
    InvalidHexCharacter,
    InvalidPercentage,
    InvalidHslValues,
    NullPointer,
    BufferTooSmall,
};

pub const ColorConstants = struct {
    pub const luminance = struct {
        pub const r: f32 = 0.2126;
        pub const g: f32 = 0.7152;
        pub const b: f32 = 0.0722;
    };

    pub const conversion = struct {
        pub const one_div_255: f32 = 1.0 / 255.0;
        pub const max_component_value: u8 = 255;
    };

    pub const angle = struct {
        pub const max_degrees: f32 = 360.0;
        pub const max_saturation: f32 = 100.0;
        pub const max_lightness: f32 = 100.0;
    };
};

pub const Col3 = struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn rgb(r: u8, g: u8, b: u8) Col3 {
        return .{ .r = r, .g = g, .b = b };
    }

    pub fn hex(str: []const u8) ColorError!Col3 {
        if (str.len == 0) return ColorError.NullPointer;

        const hex_str = if (str.len > 0 and str[0] == '#') str[1..] else str;

        if (hex_str.len != 6) return ColorError.InvalidHexLength;

        const r = try parseHexComponent(hex_str[0..2]);
        const g = try parseHexComponent(hex_str[2..4]);
        const b = try parseHexComponent(hex_str[4..6]);

        return Col3.rgb(r, g, b);
    }

    pub fn asRgb(self: Col3) struct { r: u8, g: u8, b: u8 } {
        return .{ .r = self.r, .g = self.g, .b = self.b };
    }

    pub fn asHex(self: Col3, buffer: []u8) ColorError!void {
        if (buffer.len < 6) return ColorError.BufferTooSmall;

        try fmt.bufPrint(buffer[0..2], "{X:0>2}", .{self.r});
        try fmt.bufPrint(buffer[2..4], "{X:0>2}", .{self.g});
        try fmt.bufPrint(buffer[4..6], "{X:0>2}", .{self.b});
    }

    pub fn asHexWithHash(self: Col3, buffer: []u8) ColorError!void {
        if (buffer.len < 7) return ColorError.BufferTooSmall;

        buffer[0] = '#';
        try self.asHex(buffer[1..]);
    }

    pub inline fn asFloats(self: Col3) struct { r: f32, g: f32, b: f32 } {
        const factor = comptime ColorConstants.conversion.one_div_255;
        return .{
            .r = @as(f32, @floatFromInt(self.r)) * factor,
            .g = @as(f32, @floatFromInt(self.g)) * factor,
            .b = @as(f32, @floatFromInt(self.b)) * factor,
        };
    }

    pub fn lerp(self: Col3, other: Col3, t: f32) Col3 {
        const clamped_t = math.clamp(t, 0.0, 1.0);
        const inv_t = 1.0 - clamped_t;

        const self_floats = self.asFloats();
        const other_floats = other.asFloats();

        return Col3.rgb(
            @intFromFloat(math.clamp(self_floats.r * inv_t + other_floats.r * clamped_t, 0.0, 255.0)),
            @intFromFloat(math.clamp(self_floats.g * inv_t + other_floats.g * clamped_t, 0.0, 255.0)),
            @intFromFloat(math.clamp(self_floats.b * inv_t + other_floats.b * clamped_t, 0.0, 255.0)),
        );
    }

    pub fn darken(self: Col3, percent: u8) ColorError!Col3 {
        if (percent > 100) return ColorError.InvalidPercentage;

        const factor = 1.0 - (@as(f32, @floatFromInt(percent)) / 100.0);
        const floats = self.asFloats();

        return Col3.rgb(
            @intFromFloat(math.clamp(floats.r * factor, 0.0, 255.0)),
            @intFromFloat(math.clamp(floats.g * factor, 0.0, 255.0)),
            @intFromFloat(math.clamp(floats.b * factor, 0.0, 255.0)),
        );
    }

    pub fn lighten(self: Col3, percent: u8) ColorError!Col3 {
        if (percent > 100) return ColorError.InvalidPercentage;

        const factor = 1.0 + (@as(f32, @floatFromInt(percent)) / 100.0);
        const floats = self.asFloats();

        return Col3.rgb(
            @intFromFloat(math.clamp(floats.r * factor, 0.0, 255.0)),
            @intFromFloat(math.clamp(floats.g * factor, 0.0, 255.0)),
            @intFromFloat(math.clamp(floats.b * factor, 0.0, 255.0)),
        );
    }

    pub fn eql(self: Col3, other: Col3) bool {
        return self.r == other.r and self.g == other.g and self.b == other.b;
    }

    pub fn luminance(self: Col3) f32 {
        const floats = self.asFloats();
        return ColorConstants.luminance.r * floats.r +
            ColorConstants.luminance.g * floats.g +
            ColorConstants.luminance.b * floats.b;
    }

    pub fn isDark(self: Col3) bool {
        return self.luminance() < 0.5;
    }

    pub fn asHsl(self: Col3) struct { h: f32, s: f32, l: f32 } {
        const floats = self.asFloats();
        const max = @max(@max(floats.r, floats.g), floats.b);
        const min = @min(@min(floats.r, floats.g), floats.b);
        var h: f32 = 0.0;
        var s: f32 = 0.0;
        const l = (max + min) / 2.0;

        if (max != min) {
            const d = max - min;
            s = if (l > 0.5)
                d / (2.0 - max - min)
            else
                d / (max + min);

            h = switch (max) {
                floats.r => (floats.g - floats.b) / d + (if (floats.g < floats.b) 6.0 else 0.0),
                floats.g => (floats.b - floats.r) / d + 2.0,
                else => (floats.r - floats.g) / d + 4.0,
            };
            h *= 60.0;
        }

        return .{
            .h = h,
            .s = s * 100.0,
            .l = l * 100.0,
        };
    }

    pub fn hsl(h: f32, s: f32, l: f32) ColorError!Col3 {
        // Validate HSL ranges
        if (h < 0 or h >= ColorConstants.angle.max_degrees or
            s < 0 or s > ColorConstants.angle.max_saturation or
            l < 0 or l > ColorConstants.angle.max_lightness)
        {
            return ColorError.InvalidHslValues;
        }

        const s_normalized = s / 100.0;
        const l_normalized = l / 100.0;

        if (s_normalized == 0.0) {
            const gray = @as(u8, @intFromFloat(l_normalized * ColorConstants.conversion.max_component_value));
            return Col3.rgb(gray, gray, gray);
        }

        const q = if (l_normalized < 0.5)
            l_normalized * (1.0 + s_normalized)
        else
            l_normalized + s_normalized - (l_normalized * s_normalized);

        const p = 2.0 * l_normalized - q;
        const h_normalized = h / 360.0;

        const r = hueToRgb(p, q, h_normalized + 1.0 / 3.0);
        const g = hueToRgb(p, q, h_normalized);
        const b = hueToRgb(p, q, h_normalized - 1.0 / 3.0);

        return Col3.rgb(
            @intFromFloat(math.clamp(r * 255.0, 0.0, 255.0)),
            @intFromFloat(math.clamp(g * 255.0, 0.0, 255.0)),
            @intFromFloat(math.clamp(b * 255.0, 0.0, 255.0)),
        );
    }

    pub fn mix(self: Col3, other: Col3, weight: f32) Col3 {
        return self.lerp(other, weight);
    }

    pub fn multiply(self: Col3, other: Col3) Col3 {
        const self_floats = self.asFloats();
        const other_floats = other.asFloats();

        return Col3.rgb(
            @intFromFloat(math.clamp(self_floats.r * other_floats.r, 0.0, 255.0)),
            @intFromFloat(math.clamp(self_floats.g * other_floats.g, 0.0, 255.0)),
            @intFromFloat(math.clamp(self_floats.b * other_floats.b, 0.0, 255.0)),
        );
    }

    pub fn screen(self: Col3, other: Col3) Col3 {
        const self_floats = self.asFloats();
        const other_floats = other.asFloats();

        return Col3.rgb(
            @intFromFloat(math.clamp(1.0 - (1.0 - self_floats.r) * (1.0 - other_floats.r), 0.0, 255.0)),
            @intFromFloat(math.clamp(1.0 - (1.0 - self_floats.g) * (1.0 - other_floats.g), 0.0, 255.0)),
            @intFromFloat(math.clamp(1.0 - (1.0 - self_floats.b) * (1.0 - other_floats.b), 0.0, 255.0)),
        );
    }

    pub fn overlay(self: Col3, other: Col3) Col3 {
        const self_floats = self.asFloats();
        const other_floats = other.asFloats();

        return Col3.rgb(
            @intFromFloat(overlayComponent(self_floats.r, other_floats.r)),
            @intFromFloat(overlayComponent(self_floats.g, other_floats.g)),
            @intFromFloat(overlayComponent(self_floats.b, other_floats.b)),
        );
    }

    pub fn complement(self: Col3) Col3 {
        const hsl_vals = self.asHsl();
        return (Col3.hsl(
            @mod(hsl_vals.h + 180.0, ColorConstants.angle.max_degrees),
            hsl_vals.s,
            hsl_vals.l,
        ) catch unreachable);
    }

    pub fn distance(self: Col3, other: Col3) f32 {
        const dr = @as(f32, @floatFromInt(self.r)) - @as(f32, @floatFromInt(other.r));
        const dg = @as(f32, @floatFromInt(self.g)) - @as(f32, @floatFromInt(other.g));
        const db = @as(f32, @floatFromInt(self.b)) - @as(f32, @floatFromInt(other.b));
        return math.sqrt(dr * dr + dg * dg + db * db);
    }

    pub fn toF32x4(self: Col3, alpha: ?f32) [4]f32 {
        const floats = self.asFloats();
        return .{ floats.r, floats.g, floats.b, if (alpha) |a| math.clamp(a, 0.0, 1.0) else 1.0 };
    }

    pub fn toRgba(self: Col3, alpha: ?f32) struct { r: f32, g: f32, b: f32, a: f32 } {
        const floats = self.asFloats();
        return .{
            .r = floats.r,
            .g = floats.g,
            .b = floats.b,
            .a = if (alpha) |a| math.clamp(a, 0.0, 1.0) else 1.0,
        };
    }
};

inline fn parseHexComponent(str: []const u8) ColorError!u8 {
    return fmt.parseInt(u8, str, 16) catch {
        return ColorError.InvalidHexCharacter;
    };
}

inline fn hueToRgb(p: f32, q: f32, t: f32) f32 {
    var ht = t;
    if (ht < 0.0) ht += 1.0;
    if (ht > 1.0) ht -= 1.0;

    if (ht < 1.0 / 6.0) return p + (q - p) * 6.0 * ht;
    if (ht < 1.0 / 2.0) return q;
    if (ht < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - ht) * 6.0;
    return p;
}

inline fn overlayComponent(a: f32, b: f32) f32 {
    return if (a < 0.5)
        2.0 * a * b
    else
        1.0 - 2.0 * (1.0 - a) * (1.0 - b);
}
