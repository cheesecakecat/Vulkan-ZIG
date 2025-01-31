const std = @import("std");
const builtin = @import("builtin");

pub const Vec2 = extern struct {
    data: @Vector(2, f32) align(8),

    const Self = @This();

    pub inline fn init(x_val: f32, y_val: f32) Self {
        return .{ .data = .{ x_val, y_val } };
    }

    pub inline fn zero() Self {
        return .{ .data = .{ 0, 0 } };
    }

    pub inline fn one() Self {
        return .{ .data = .{ 1, 1 } };
    }

    pub inline fn splat(value: f32) Self {
        return .{ .data = @splat(value) };
    }

    pub inline fn x(self: Self) f32 {
        return self.data[0];
    }

    pub inline fn y(self: Self) f32 {
        return self.data[1];
    }

    pub inline fn add(self: Self, other: Self) Self {
        return .{ .data = self.data + other.data };
    }

    pub inline fn sub(self: Self, other: Self) Self {
        return .{ .data = self.data - other.data };
    }

    pub inline fn mul(self: Self, other: Self) Self {
        return .{ .data = self.data * other.data };
    }

    pub inline fn scale(self: Self, scalar: f32) Self {
        return .{ .data = self.data * @as(@Vector(2, f32), @splat(scalar)) };
    }

    pub inline fn dot(self: Self, other: Self) f32 {
        return @reduce(.Add, self.data * other.data);
    }

    pub inline fn length2(self: Self) f32 {
        return self.dot(self);
    }

    pub inline fn length(self: Self) f32 {
        return @sqrt(self.length2());
    }

    pub inline fn normalize(self: Self) Self {
        const len = self.length();
        return if (len > 0) self.scale(1.0 / len) else self;
    }

    pub inline fn rotate(self: Self, angle: f32) Self {
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        return .{
            .data = .{
                self.x() * cos_a - self.y() * sin_a,
                self.x() * sin_a + self.y() * cos_a,
            },
        };
    }

    pub inline fn lerp(self: Self, other: Self, t: f32) Self {
        return self.add(other.sub(self).scale(t));
    }

    pub inline fn toArray(self: Self) [2]f32 {
        return .{ self.x(), self.y() };
    }

    pub inline fn fromArray(arr: [2]f32) Self {
        return .{ .data = arr };
    }
};

pub const Vec4 = extern struct {
    data: @Vector(4, f32) align(16),

    const Self = @This();

    pub inline fn init(x_val: f32, y_val: f32, z_val: f32, w_val: f32) Self {
        return .{ .data = .{ x_val, y_val, z_val, w_val } };
    }

    pub inline fn zero() Self {
        return .{ .data = .{ 0, 0, 0, 0 } };
    }

    pub inline fn one() Self {
        return .{ .data = .{ 1, 1, 1, 1 } };
    }

    pub inline fn splat(value: f32) Self {
        return .{ .data = @splat(value) };
    }

    pub inline fn x(self: Self) f32 {
        return self.data[0];
    }

    pub inline fn y(self: Self) f32 {
        return self.data[1];
    }

    pub inline fn z(self: Self) f32 {
        return self.data[2];
    }

    pub inline fn w(self: Self) f32 {
        return self.data[3];
    }

    pub inline fn add(self: Self, other: Self) Self {
        return .{ .data = self.data + other.data };
    }

    pub inline fn sub(self: Self, other: Self) Self {
        return .{ .data = self.data - other.data };
    }

    pub inline fn mul(self: Self, other: Self) Self {
        return .{ .data = self.data * other.data };
    }

    pub inline fn scale(self: Self, scalar: f32) Self {
        return .{ .data = self.data * @as(@Vector(4, f32), @splat(scalar)) };
    }

    pub inline fn dot(self: Self, other: Self) f32 {
        return @reduce(.Add, self.data * other.data);
    }

    pub inline fn length2(self: Self) f32 {
        return self.dot(self);
    }

    pub inline fn length(self: Self) f32 {
        return @sqrt(self.length2());
    }

    pub inline fn normalize(self: Self) Self {
        const len = self.length();
        return if (len > 0) self.scale(1.0 / len) else self;
    }

    pub inline fn lerp(self: Self, other: Self, t: f32) Self {
        return self.add(other.sub(self).scale(t));
    }

    pub inline fn toArray(self: Self) [4]f32 {
        return .{ self.x(), self.y(), self.z(), self.w() };
    }

    pub inline fn fromArray(arr: [4]f32) Self {
        return .{ .data = arr };
    }
};
