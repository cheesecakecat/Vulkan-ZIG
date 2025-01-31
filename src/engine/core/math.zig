const std = @import("std");
const Vector = std.meta.Vector;

// SIMD vector types
const Vec4f = @Vector(4, f32);
const Mat4x4f = [4]Vec4f;

pub fn cos(x: f32) f32 {
    return @cos(x);
}

pub fn sin(x: f32) f32 {
    return @sin(x);
}

// Convert standard matrix to SIMD matrix
fn toSIMD(mat: [4][4]f32) Mat4x4f {
    return .{
        Vec4f{ mat[0][0], mat[0][1], mat[0][2], mat[0][3] },
        Vec4f{ mat[1][0], mat[1][1], mat[1][2], mat[1][3] },
        Vec4f{ mat[2][0], mat[2][1], mat[2][2], mat[2][3] },
        Vec4f{ mat[3][0], mat[3][1], mat[3][2], mat[3][3] },
    };
}

// Convert SIMD matrix to standard matrix
fn fromSIMD(mat: Mat4x4f) [4][4]f32 {
    var result: [4][4]f32 = undefined;
    inline for (0..4) |i| {
        inline for (0..4) |j| {
            result[i][j] = mat[i][j];
        }
    }
    return result;
}

pub fn mat4x4_identity() [4][4]f32 {
    const simd_mat = Mat4x4f{
        Vec4f{ 1.0, 0.0, 0.0, 0.0 },
        Vec4f{ 0.0, 1.0, 0.0, 0.0 },
        Vec4f{ 0.0, 0.0, 1.0, 0.0 },
        Vec4f{ 0.0, 0.0, 0.0, 1.0 },
    };
    return fromSIMD(simd_mat);
}

pub fn mat4x4_ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) [4][4]f32 {
    const rml = right - left;
    const tmb = top - bottom;
    const fmn = far - near;

    return .{
        .{ 2.0 / rml, 0.0, 0.0, -(right + left) / rml },
        .{ 0.0, 2.0 / tmb, 0.0, -(top + bottom) / tmb },
        .{ 0.0, 0.0, -2.0 / fmn, -(far + near) / fmn },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

pub fn mat4x4_translate(x: f32, y: f32, z: f32) [4][4]f32 {
    const simd_mat = Mat4x4f{
        Vec4f{ 1.0, 0.0, 0.0, x },
        Vec4f{ 0.0, 1.0, 0.0, y },
        Vec4f{ 0.0, 0.0, 1.0, z },
        Vec4f{ 0.0, 0.0, 0.0, 1.0 },
    };
    return fromSIMD(simd_mat);
}

pub fn mat4x4_scale(x: f32, y: f32, z: f32) [4][4]f32 {
    const simd_mat = Mat4x4f{
        Vec4f{ x, 0.0, 0.0, 0.0 },
        Vec4f{ 0.0, y, 0.0, 0.0 },
        Vec4f{ 0.0, 0.0, z, 0.0 },
        Vec4f{ 0.0, 0.0, 0.0, 1.0 },
    };
    return fromSIMD(simd_mat);
}

pub fn mat4x4_rotate(angle: f32, x: f32, y: f32, z: f32) [4][4]f32 {
    const c = @cos(angle);
    const s = @sin(angle);
    const t = 1.0 - c;

    const simd_mat = Mat4x4f{
        Vec4f{ t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0 },
        Vec4f{ t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0 },
        Vec4f{ t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0 },
        Vec4f{ 0.0, 0.0, 0.0, 1.0 },
    };
    return fromSIMD(simd_mat);
}

pub fn mat4x4_multiply(a: [4][4]f32, b: [4][4]f32) [4][4]f32 {
    const simd_a = toSIMD(a);
    const simd_b = toSIMD(b);
    var result: Mat4x4f = undefined;

    inline for (0..4) |i| {
        const row = simd_a[i];
        inline for (0..4) |j| {
            const col = Vec4f{ simd_b[0][j], simd_b[1][j], simd_b[2][j], simd_b[3][j] };
            result[i][j] = @reduce(.Add, row * col);
        }
    }

    return fromSIMD(result);
}

pub fn vec2_add(a: [2]f32, b: [2]f32) [2]f32 {
    return .{ a[0] + b[0], a[1] + b[1] };
}

pub fn vec2_subtract(a: [2]f32, b: [2]f32) [2]f32 {
    return .{ a[0] - b[0], a[1] - b[1] };
}

pub fn vec2_multiply(a: [2]f32, b: [2]f32) [2]f32 {
    return .{ a[0] * b[0], a[1] * b[1] };
}

pub fn vec2_scale(a: [2]f32, s: f32) [2]f32 {
    return .{ a[0] * s, a[1] * s };
}

pub fn vec2_dot(a: [2]f32, b: [2]f32) f32 {
    return a[0] * b[0] + a[1] * b[1];
}

pub fn vec2_length(a: [2]f32) f32 {
    return @sqrt(vec2_dot(a, a));
}

pub fn vec2_normalize(a: [2]f32) [2]f32 {
    const len = vec2_length(a);
    return vec2_scale(a, 1.0 / len);
}

pub fn vec2_rotate(a: [2]f32, angle: f32) [2]f32 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        a[0] * c - a[1] * s,
        a[0] * s + a[1] * c,
    };
}

pub fn vec2_lerp(a: [2]f32, b: [2]f32, t: f32) [2]f32 {
    return vec2_add(vec2_scale(a, 1.0 - t), vec2_scale(b, t));
}

pub fn ortho(left: f32, right: f32, bottom: f32, top: f32) [4][4]f32 {
    const rml = right - left;
    const tmb = top - bottom;

    // Vulkan NDC is right-handed with Y pointing down and Z pointing into the screen
    // X and Y go from -1 to 1, Z goes from 0 to 1
    const simd_mat = Mat4x4f{
        Vec4f{ 2.0 / rml, 0.0, 0.0, 0.0 },
        Vec4f{ 0.0, 2.0 / tmb, 0.0, 0.0 },
        Vec4f{ 0.0, 0.0, 1.0, 0.0 },
        Vec4f{ -(right + left) / rml, -(top + bottom) / tmb, 0.0, 1.0 },
    };

    return fromSIMD(simd_mat);
}
