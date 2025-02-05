const std = @import("std");
const math = @import("../../math.zig");
const zigent = @import("../../../../deps/zigent.zig");

pub const BlendMode = enum {
    Alpha,

    Additive,

    Multiply,

    Screen,
};

pub const MaterialProperties = struct {
    tint: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },

    texture_rect: [4]f32 = .{ 0.0, 0.0, 1.0, 1.0 },

    blend_mode: BlendMode = .Alpha,

    shader_id: u32 = 0,

    flags: packed struct {
        filter: bool = true,

        wrap: bool = false,

        depth_test: bool = false,

        cull_face: bool = false,

        color_write: bool = true,

        alpha_write: bool = true,

        _padding: u10 = 0,
    } = .{},
};

pub const VisibilityState = enum {
    Visible,

    Hidden,

    Culled,
};

pub const RenderPriority = enum(u8) {
    Background = 0,

    World = 128,

    Entity = 192,

    UI = 255,
};

pub const RenderableHooks = struct {
    on_show: ?*const fn (*Renderable) void = null,

    on_hide: ?*const fn (*Renderable) void = null,

    pre_render: ?*const fn (*Renderable) void = null,

    post_render: ?*const fn (*Renderable) void = null,
};

pub const Renderable = struct {
    texture_id: u32 = 0,

    material: MaterialProperties = .{},

    visibility: VisibilityState = .Visible,

    priority: RenderPriority = .Entity,

    hooks: RenderableHooks = .{},

    bounds: struct {
        min: math.Vec2 = math.Vec2.zero(),
        max: math.Vec2 = math.Vec2.zero(),
        dirty: bool = true,
    } = .{},

    debug: packed struct {
        show_bounds: bool = false,

        show_pivot: bool = false,

        show_uv_grid: bool = false,
        _padding: u5 = 0,
    } = .{},

    pub fn init() Renderable {
        return .{};
    }

    pub fn setTexture(self: *Renderable, id: u32) *Renderable {
        self.texture_id = id;
        self.bounds.dirty = true;
        return self;
    }

    pub fn setTextureRect(self: *Renderable, x: f32, y: f32, w: f32, h: f32) *Renderable {
        self.material.texture_rect = .{ x, y, w, h };
        self.bounds.dirty = true;
        return self;
    }

    pub fn setTint(self: *Renderable, r: f32, g: f32, b: f32, a: f32) *Renderable {
        self.material.tint = .{ r, g, b, a };
        return self;
    }

    pub fn setBlendMode(self: *Renderable, mode: BlendMode) *Renderable {
        self.material.blend_mode = mode;
        return self;
    }

    pub fn setVisibility(self: *Renderable, state: VisibilityState) void {
        if (self.visibility == state) return;

        const old_state = self.visibility;
        self.visibility = state;

        if (old_state != .Hidden and state == .Hidden) {
            if (self.hooks.on_hide) |hook| hook(self);
        } else if (old_state == .Hidden and state != .Hidden) {
            if (self.hooks.on_show) |hook| hook(self);
        }
    }

    pub fn setPriority(self: *Renderable, new_priority: RenderPriority) *Renderable {
        self.priority = new_priority;
        return self;
    }

    pub fn updateBounds(self: *Renderable, transform: math.Mat4) void {
        if (!self.bounds.dirty) return;

        const corners = [_]math.Vec2{
            math.Vec2.init(-0.5, -0.5),
            math.Vec2.init(0.5, -0.5),
            math.Vec2.init(-0.5, 0.5),
            math.Vec2.init(0.5, 0.5),
        };

        var min = math.Vec2.init(std.math.inf(f32), std.math.inf(f32));
        var max = math.Vec2.init(-std.math.inf(f32), -std.math.inf(f32));

        for (corners) |corner| {
            const transformed = transform.transformPoint(corner);
            min = min.min(transformed);
            max = max.max(transformed);
        }

        self.bounds.min = min;
        self.bounds.max = max;
        self.bounds.dirty = false;
    }

    pub fn containsPoint(self: *const Renderable, point: math.Vec2) bool {
        return point.x() >= self.bounds.min.x() and
            point.x() <= self.bounds.max.x() and
            point.y() >= self.bounds.min.y() and
            point.y() <= self.bounds.max.y();
    }

    pub fn getMaterialProperties(self: *const Renderable) MaterialProperties {
        return self.material;
    }

    pub fn preRender(self: *Renderable) void {
        if (self.hooks.pre_render) |hook| hook(self);
    }

    pub fn postRender(self: *Renderable) void {
        if (self.hooks.post_render) |hook| hook(self);
    }

    pub fn setDebugVisualization(self: *Renderable, show_bounds: bool, show_pivot: bool, show_uv_grid: bool) void {
        self.debug.show_bounds = show_bounds;
        self.debug.show_pivot = show_pivot;
        self.debug.show_uv_grid = show_uv_grid;
    }
};
