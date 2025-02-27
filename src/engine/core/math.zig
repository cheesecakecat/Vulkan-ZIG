const std = @import("std");

pub const vec = @import("math/vec.zig");
pub const mat = @import("math/mat.zig");

pub const Vec2 = vec.Vec2;
pub const Vec4 = vec.Vec4;
pub const Mat4 = mat.Mat4;

pub const math = struct {
    pub const cos = std.math.cos;
    pub const sin = std.math.sin;
    pub const sqrt = std.math.sqrt;
};

pub fn clamp(value: anytype, min: @TypeOf(value), max: @TypeOf(value)) @TypeOf(value) {
    return @min(@max(value, min), max);
}

pub fn ortho(left: f32, right: f32, bottom: f32, top: f32) Mat4 {
    return Mat4.ortho2D(left, right, bottom, top);
}
