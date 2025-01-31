const std = @import("std");

pub const vec = @import("math/vec.zig");
pub const mat = @import("math/mat.zig");

pub const Vec2 = vec.Vec2;
pub const Vec4 = vec.Vec4;
pub const Mat4 = mat.Mat4;

// Re-export common math functions
pub const math = struct {
    pub const cos = std.math.cos;
    pub const sin = std.math.sin;
    pub const sqrt = std.math.sqrt;
};

// Convenience functions
pub fn ortho(left: f32, right: f32, bottom: f32, top: f32) Mat4 {
    return Mat4.ortho2D(left, right, bottom, top);
}

test {
    _ = vec;
    _ = mat;
}
