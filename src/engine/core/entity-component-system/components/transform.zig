const std = @import("std");
const math = @import("../../math.zig");
const zigent = @import("../../../../deps/zigent.zig");

pub const TransformSpace = enum {
    Local,

    World,
};

pub const Transform = struct {
    local_position: math.Vec2 = math.Vec2.zero(),

    local_rotation: f32 = 0.0,

    local_scale: math.Vec2 = math.Vec2.one(),

    layer: u32 = 0,

    parent: ?*Transform = null,

    world_matrix: math.Mat4 = math.Mat4.identity(),

    local_matrix: math.Mat4 = math.Mat4.identity(),

    flags: packed struct {
        local_dirty: bool = true,
        world_dirty: bool = true,
    } = .{},

    pub fn init() Transform {
        return .{};
    }

    pub fn getWorldMatrix(self: *Transform) math.Mat4 {
        if (self.flags.world_dirty) {
            self.updateWorldMatrix();
        }
        return self.world_matrix;
    }

    pub fn getLocalMatrix(self: *Transform) math.Mat4 {
        if (self.flags.local_dirty) {
            self.updateLocalMatrix();
        }
        return self.local_matrix;
    }

    fn updateLocalMatrix(self: *Transform) void {
        var matrix = math.Mat4.identity();
        matrix = matrix.mul(math.Mat4.scale(self.local_scale.x(), self.local_scale.y(), 1));
        matrix = matrix.mul(math.Mat4.rotationZ(self.local_rotation));
        matrix = matrix.mul(math.Mat4.translate(self.local_position.x(), self.local_position.y(), 0));
        self.local_matrix = matrix;
        self.flags.local_dirty = false;
        self.flags.world_dirty = true;
    }

    fn updateWorldMatrix(self: *Transform) void {
        const local = self.getLocalMatrix();
        self.world_matrix = if (self.parent) |parent|
            parent.getWorldMatrix().mul(local)
        else
            local;
        self.flags.world_dirty = false;
    }

    pub fn setParent(self: *Transform, new_parent: ?*Transform) void {
        if (self.parent == new_parent) return;
        self.parent = new_parent;
        self.flags.world_dirty = true;
    }

    pub fn getWorldPosition(self: *Transform) math.Vec2 {
        const world = self.getWorldMatrix();
        return math.Vec2.init(world.get(3, 0), world.get(3, 1));
    }

    pub fn getWorldRotation(self: *Transform) f32 {
        if (self.parent) |parent| {
            return parent.getWorldRotation() + self.local_rotation;
        }
        return self.local_rotation;
    }

    pub fn getWorldScale(self: *Transform) math.Vec2 {
        if (self.parent) |parent| {
            const parent_scale = parent.getWorldScale();
            return math.Vec2.init(parent_scale.x() * self.local_scale.x(), parent_scale.y() * self.local_scale.y());
        }
        return self.local_scale;
    }

    pub fn setPosition(self: *Transform, x: f32, y: f32, space: TransformSpace) *Transform {
        switch (space) {
            .Local => {
                self.local_position = math.Vec2.init(x, y);
                self.flags.local_dirty = true;
            },
            .World => {
                const local_pos = if (self.parent) |parent| blk: {
                    const inv_parent = parent.getWorldMatrix().invert() catch return self;
                    const world_pos = math.Vec2.init(x, y);
                    break :blk inv_parent.transformPoint(world_pos);
                } else blk: {
                    break :blk math.Vec2.init(x, y);
                };
                self.local_position = local_pos;
                self.flags.local_dirty = true;
            },
        }
        return self;
    }

    pub fn setRotation(self: *Transform, angle: f32, space: TransformSpace) *Transform {
        switch (space) {
            .Local => {
                self.local_rotation = angle;
                self.flags.local_dirty = true;
            },
            .World => {
                self.local_rotation = if (self.parent) |parent|
                    angle - parent.getWorldRotation()
                else
                    angle;
                self.flags.local_dirty = true;
            },
        }
        return self;
    }

    pub fn setScale(self: *Transform, x: f32, y: f32, space: TransformSpace) *Transform {
        switch (space) {
            .Local => {
                self.local_scale = math.Vec2.init(x, y);
                self.flags.local_dirty = true;
            },
            .World => {
                if (self.parent) |parent| {
                    const parent_scale = parent.getWorldScale();
                    self.local_scale = math.Vec2.init(x / parent_scale.x(), y / parent_scale.y());
                } else {
                    self.local_scale = math.Vec2.init(x, y);
                }
                self.flags.local_dirty = true;
            },
        }
        return self;
    }

    pub fn setLayer(self: *Transform, new_layer: u32) *Transform {
        self.layer = new_layer;
        return self;
    }

    pub fn translate(self: *Transform, x: f32, y: f32, space: TransformSpace) *Transform {
        switch (space) {
            .Local => {
                const delta = math.Vec2.init(x, y);
                self.local_position = self.local_position.add(delta);
                self.flags.local_dirty = true;
            },
            .World => {
                const world_delta = if (self.parent) |parent| {
                    const inv_parent = parent.getWorldMatrix().invert() catch return self;
                    const delta = math.Vec2.init(x, y);
                    const transformed = inv_parent.transformPoint(delta);
                    transformed;
                } else {
                    math.Vec2.init(x, y);
                };
                self.local_position = self.local_position.add(world_delta);
                self.flags.local_dirty = true;
            },
        }
        return self;
    }

    pub fn rotate(self: *Transform, angle: f32, space: TransformSpace) *Transform {
        switch (space) {
            .Local => {
                self.local_rotation += angle;
                self.flags.local_dirty = true;
            },
            .World => {
                self.local_rotation += angle;
                self.flags.local_dirty = true;
            },
        }
        return self;
    }

    pub fn scaleBy(self: *Transform, x: f32, y: f32, space: TransformSpace) *Transform {
        switch (space) {
            .Local => {
                self.local_scale = self.local_scale.mul(math.Vec2.init(x, y));
                self.flags.local_dirty = true;
            },
            .World => {
                const scale_delta = math.Vec2.init(x, y);
                self.local_scale = self.local_scale.mul(scale_delta);
                self.flags.local_dirty = true;
            },
        }
        return self;
    }

    pub fn lookAt(self: *Transform, target_x: f32, target_y: f32, space: TransformSpace) *Transform {
        const current_pos = switch (space) {
            .Local => self.local_position,
            .World => self.getWorldPosition(),
        };

        const target = math.Vec2.init(target_x, target_y);
        const direction = target.sub(current_pos);
        const angle = std.math.atan2(f32, direction.y(), direction.x());

        return self.setRotation(angle, space);
    }
};
