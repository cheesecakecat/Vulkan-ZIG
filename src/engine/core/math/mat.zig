const std = @import("std");
const vec = @import("vec.zig");
const Vec4 = vec.Vec4;
const Vec2 = vec.Vec2;

pub const Mat4 = extern struct {
    rows: [4]Vec4 align(16),

    const Self = @This();

    pub inline fn identity() Self {
        return .{
            .rows = .{
                Vec4.init(1, 0, 0, 0),
                Vec4.init(0, 1, 0, 0),
                Vec4.init(0, 0, 1, 0),
                Vec4.init(0, 0, 0, 1),
            },
        };
    }

    pub inline fn translation(tx: f32, ty: f32, tz: f32) Self {
        return fromRows(
            Vec4.init(1, 0, 0, 0),
            Vec4.init(0, 1, 0, 0),
            Vec4.init(0, 0, 1, 0),
            Vec4.init(tx, ty, tz, 1),
        );
    }

    pub inline fn fromRows(r0: Vec4, r1: Vec4, r2: Vec4, r3: Vec4) Self {
        return .{ .rows = .{ r0, r1, r2, r3 } };
    }

    pub inline fn fromColumns(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) Self {
        return .{
            .rows = .{
                Vec4.init(c0.x(), c1.x(), c2.x(), c3.x()),
                Vec4.init(c0.y(), c1.y(), c2.y(), c3.y()),
                Vec4.init(c0.z(), c1.z(), c2.z(), c3.z()),
                Vec4.init(c0.w(), c1.w(), c2.w(), c3.w()),
            },
        };
    }

    pub inline fn mul(self: Self, other: Self) Self {
        var result: Self = undefined;
        inline for (0..4) |i| {
            const row = self.rows[i];
            inline for (0..4) |j| {
                const col = Vec4.init(
                    other.rows[0].data[j],
                    other.rows[1].data[j],
                    other.rows[2].data[j],
                    other.rows[3].data[j],
                );
                result.rows[i].data[j] = row.dot(col);
            }
        }
        return result;
    }

    pub inline fn mulVec(self: Self, v: Vec4) Vec4 {
        var result: Vec4 = undefined;
        inline for (0..4) |i| {
            result.data[i] = self.rows[i].dot(v);
        }
        return result;
    }

    pub inline fn translate(tx: f32, ty: f32, tz: f32) Self {
        var result = identity();
        result.rows[3] = Vec4.init(tx, ty, tz, 1);
        return result;
    }

    pub inline fn scale(sx: f32, sy: f32, sz: f32) Self {
        return fromRows(
            Vec4.init(sx, 0, 0, 0),
            Vec4.init(0, sy, 0, 0),
            Vec4.init(0, 0, sz, 0),
            Vec4.init(0, 0, 0, 1),
        );
    }

    pub inline fn rotate(angle: f32, x: f32, y: f32, z: f32) Self {
        const c = @cos(angle);
        const s = @sin(angle);
        const t = 1.0 - c;

        const axis = Vec4.init(x, y, z, 0).normalize();
        const x_norm = axis.x();
        const y_norm = axis.y();
        const z_norm = axis.z();

        return fromRows(
            Vec4.init(
                t * x_norm * x_norm + c,
                t * x_norm * y_norm - s * z_norm,
                t * x_norm * z_norm + s * y_norm,
                0,
            ),
            Vec4.init(
                t * x_norm * y_norm + s * z_norm,
                t * y_norm * y_norm + c,
                t * y_norm * z_norm - s * x_norm,
                0,
            ),
            Vec4.init(
                t * x_norm * z_norm - s * y_norm,
                t * y_norm * z_norm + s * x_norm,
                t * z_norm * z_norm + c,
                0,
            ),
            Vec4.init(0, 0, 0, 1),
        );
    }

    pub inline fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Self {
        const rml = right - left;
        const tmb = top - bottom;
        const fmn = far - near;

        return fromColumns(
            Vec4.init(2.0 / rml, 0, 0, 0),
            Vec4.init(0, 2.0 / tmb, 0, 0),
            Vec4.init(0, 0, 1.0 / fmn, 0),
            Vec4.init(-(right + left) / rml, -(top + bottom) / tmb, -near / fmn, 1),
        );
    }

    pub inline fn ortho2D(left: f32, right: f32, bottom: f32, top: f32) Self {
        const rml = right - left;
        const tmb = top - bottom;

        return fromRows(
            Vec4.init(2.0 / rml, 0, 0, 0),
            Vec4.init(0, 2.0 / tmb, 0, 0),
            Vec4.init(0, 0, -1, 0),
            Vec4.init(-(right + left) / rml, -(top + bottom) / tmb, 0, 1),
        );
    }

    pub inline fn transpose(self: Self) Self {
        return fromColumns(
            self.rows[0],
            self.rows[1],
            self.rows[2],
            self.rows[3],
        );
    }

    pub inline fn toArray(self: Self) [16]f32 {
        var result: [16]f32 = undefined;
        inline for (0..4) |i| {
            inline for (0..4) |j| {
                result[i * 4 + j] = self.rows[i].data[j];
            }
        }
        return result;
    }

    pub inline fn fromArray(arr: [16]f32) Self {
        return fromRows(
            Vec4.fromArray(arr[0..4].*),
            Vec4.fromArray(arr[4..8].*),
            Vec4.fromArray(arr[8..12].*),
            Vec4.fromArray(arr[12..16].*),
        );
    }

    pub inline fn getRow(self: Self, row: usize) *const Vec4 {
        return &self.rows[row];
    }

    pub inline fn get(self: Self, row: usize, col: usize) f32 {
        return self.rows[row].data[col];
    }

    pub inline fn toArray2D(self: Self) [4][4]f32 {
        var result: [4][4]f32 = undefined;
        inline for (0..4) |i| {
            inline for (0..4) |j| {
                result[i][j] = self.rows[i].data[j];
            }
        }
        return result;
    }

    pub fn invert(self: Self) !Self {
        var out: [16]f32 = undefined;
        var det: f32 = undefined;

        out[0] = self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[2] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[2] -
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[1] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[1] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[2] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[1];

        out[4] = -self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[2] +
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[2] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[1] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[1] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[2] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[1];

        out[8] = self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[1] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[1] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[2] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[1] +
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[2] -
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[2] * self.rows[3].data[3];

        out[12] = -self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[1] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[1] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[2] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[1] -
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[2] +
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[2] * self.rows[3].data[3];

        out[1] = -self.rows[0].data[1] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[0] +
            self.rows[0].data[1] * self.rows[1].data[2] * self.rows[2].data[0] * self.rows[3].data[3] +
            self.rows[0].data[1] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[0] -
            self.rows[0].data[1] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[2] -
            self.rows[0].data[1] * self.rows[1].data[0] * self.rows[2].data[2] * self.rows[3].data[3] +
            self.rows[0].data[1] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[2];

        out[5] = self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[0] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[2] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[2] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[2];

        out[9] = -self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[0] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[1] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[1] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[1];

        out[13] = self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[0] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[1] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[1] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[1];

        out[2] = self.rows[0].data[1] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[0] -
            self.rows[0].data[1] * self.rows[1].data[2] * self.rows[2].data[0] * self.rows[3].data[3] -
            self.rows[0].data[1] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[0] +
            self.rows[0].data[1] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[2] +
            self.rows[0].data[1] * self.rows[1].data[0] * self.rows[2].data[2] * self.rows[3].data[3] -
            self.rows[0].data[1] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[2];

        out[6] = -self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[0] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[2] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[2] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[2];

        out[10] = self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[0] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[1] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[1] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[1];

        out[14] = -self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[0] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[1] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[1] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[1];

        out[3] = -self.rows[0].data[1] * self.rows[1].data[2] * self.rows[2].data[0] * self.rows[3].data[3] +
            self.rows[0].data[1] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[0] +
            self.rows[0].data[1] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[2] -
            self.rows[0].data[1] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[0] -
            self.rows[0].data[1] * self.rows[1].data[0] * self.rows[2].data[2] * self.rows[3].data[3] +
            self.rows[0].data[1] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[2];

        out[7] = self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[0] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[2] * self.rows[2].data[3] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[2] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[2] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[2] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[2];

        out[11] = -self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[0] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[1] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[1] * self.rows[3].data[3] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[1];

        out[15] = self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[0] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[1] * self.rows[2].data[3] * self.rows[3].data[0] -
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[1] * self.rows[3].data[0] +
            self.rows[0].data[0] * self.rows[1].data[3] * self.rows[2].data[0] * self.rows[3].data[1] +
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[1] * self.rows[3].data[3] -
            self.rows[0].data[0] * self.rows[1].data[0] * self.rows[2].data[3] * self.rows[3].data[1];

        det = self.rows[0].data[0] * out[0] + self.rows[0].data[1] * out[4] + self.rows[0].data[2] * out[8] + self.rows[0].data[3] * out[12];

        if (det == 0) {
            return error.SingularMatrix;
        }

        det = 1.0 / det;

        for (0..16) |i| {
            out[i] *= det;
        }

        return fromRows(
            Vec4.fromArray(out[0..4].*),
            Vec4.fromArray(out[4..8].*),
            Vec4.fromArray(out[8..12].*),
            Vec4.fromArray(out[12..16].*),
        );
    }

    pub fn transformPoint(self: Self, point: Vec2) Vec2 {
        const x = self.rows[0].data[0] * point.x() + self.rows[0].data[1] * point.y() + self.rows[0].data[3];
        const y = self.rows[1].data[0] * point.x() + self.rows[1].data[1] * point.y() + self.rows[1].data[3];
        return Vec2.init(x, y);
    }

    pub inline fn rotationX(angle: f32) Self {
        const c = @cos(angle);
        const s = @sin(angle);
        return fromRows(
            Vec4.init(1, 0, 0, 0),
            Vec4.init(0, c, -s, 0),
            Vec4.init(0, s, c, 0),
            Vec4.init(0, 0, 0, 1),
        );
    }

    pub inline fn rotationY(angle: f32) Self {
        const c = @cos(angle);
        const s = @sin(angle);
        return fromRows(
            Vec4.init(c, 0, s, 0),
            Vec4.init(0, 1, 0, 0),
            Vec4.init(-s, 0, c, 0),
            Vec4.init(0, 0, 0, 1),
        );
    }

    pub inline fn rotationZ(angle: f32) Self {
        const c = @cos(angle);
        const s = @sin(angle);
        return fromRows(
            Vec4.init(c, -s, 0, 0),
            Vec4.init(s, c, 0, 0),
            Vec4.init(0, 0, 1, 0),
            Vec4.init(0, 0, 0, 1),
        );
    }

    pub inline fn scaling(sx: f32, sy: f32, sz: f32) Self {
        return fromRows(
            Vec4.init(sx, 0, 0, 0),
            Vec4.init(0, sy, 0, 0),
            Vec4.init(0, 0, sz, 0),
            Vec4.init(0, 0, 0, 1),
        );
    }

    pub inline fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) Self {
        const f = 1.0 / @tan(fovy / 2.0);
        const nf = 1.0 / (near - far);

        return fromRows(
            Vec4.init(f / aspect, 0, 0, 0),
            Vec4.init(0, f, 0, 0),
            Vec4.init(0, 0, (far + near) * nf, -1),
            Vec4.init(0, 0, 2.0 * far * near * nf, 0),
        );
    }

    pub inline fn lookAt(eye: Vec4, center: Vec4, up: Vec4) Self {
        const f = center.sub(eye).normalize();
        const s = f.cross(up).normalize();
        const u = s.cross(f);

        return fromRows(
            Vec4.init(s.x(), s.y(), s.z(), -s.dot(eye)),
            Vec4.init(u.x(), u.y(), u.z(), -u.dot(eye)),
            Vec4.init(-f.x(), -f.y(), -f.z(), f.dot(eye)),
            Vec4.init(0, 0, 0, 1),
        );
    }
};
