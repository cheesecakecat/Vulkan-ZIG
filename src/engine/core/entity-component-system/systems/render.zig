const std = @import("std");
const math = @import("../../math.zig");
const zigent = @import("../../../../deps/zigent.zig");
const sprite = @import("../../../vulkan/sprite.zig");
const Transform = @import("../components/transform.zig").Transform;
const Renderable = @import("../components/renderable.zig").Renderable;
const RenderPriority = @import("../components/renderable.zig").RenderPriority;
const logger = @import("../../../core/logger.zig");

pub const RenderSystemConfig = struct {
    max_sprites: u32 = 500_000,

    enable_culling: bool = true,

    enable_debug: bool = false,

    sort_mode: enum {
        Priority,

        PriorityAndDepth,

        Material,
    } = .PriorityAndDepth,
};

const RenderTransform = struct {
    world_matrix: math.Mat4,

    depth: f32,

    layer: u32,
};

const RenderQueueItem = struct {
    entity: zigent.Id,
    transform: RenderTransform,
    renderable: *Renderable,
};

pub const RenderSystem = struct {
    ctx: *zigent.Context,

    component_ids: struct {
        transform: zigent.Id,
        renderable: zigent.Id,
    },

    sprite_batch: *sprite.SpriteBatch,

    config: RenderSystemConfig,

    transform_cache: std.AutoHashMap(zigent.Id, RenderTransform),

    render_queue: std.ArrayList(RenderQueueItem),

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, ctx: *zigent.Context, transform_id: zigent.Id, renderable_id: zigent.Id, sprite_batch: *sprite.SpriteBatch, config: RenderSystemConfig) !*RenderSystem {
        const self = try allocator.create(RenderSystem);
        errdefer allocator.destroy(self);

        self.* = .{
            .ctx = ctx,
            .component_ids = .{
                .transform = transform_id,
                .renderable = renderable_id,
            },
            .sprite_batch = sprite_batch,
            .config = config,
            .transform_cache = std.AutoHashMap(zigent.Id, RenderTransform).init(allocator),
            .render_queue = std.ArrayList(RenderQueueItem).init(allocator),
            .allocator = allocator,
        };

        return self;
    }

    pub fn deinit(self: *RenderSystem) void {
        self.transform_cache.deinit();
        self.render_queue.deinit();
        self.allocator.destroy(self);
    }

    pub fn update(self: *RenderSystem, delta_time: f32) !void {
        _ = delta_time;

        self.transform_cache.clearRetainingCapacity();
        self.render_queue.clearRetainingCapacity();

        var entity: zigent.Id = 0;
        var entity_count: u32 = 0;
        while (self.ctx.isReady(entity)) : (entity += 1) {
            const transform = zigent.getAs(Transform, self.ctx, entity, self.component_ids.transform) orelse continue;
            const renderable = zigent.getAs(Renderable, self.ctx, entity, self.component_ids.renderable) orelse continue;

            entity_count += 1;

            try self.transform_cache.put(entity, .{
                .world_matrix = transform.getWorldMatrix(),
                .depth = transform.local_position.y(),
                .layer = transform.layer,
            });

            try self.render_queue.append(.{
                .entity = entity,
                .transform = self.transform_cache.get(entity).?,
                .renderable = renderable,
            });
        }

        switch (self.config.sort_mode) {
            .Priority => {
                std.sort.pdq(RenderQueueItem, self.render_queue.items, {}, comparePriority);
            },
            .PriorityAndDepth => {
                std.sort.pdq(RenderQueueItem, self.render_queue.items, {}, comparePriorityAndDepth);
            },
            .Material => {
                std.sort.pdq(RenderQueueItem, self.render_queue.items, {}, compareMaterial);
            },
        }

        self.sprite_batch.begin();

        for (self.render_queue.items) |item| {
            if (item.renderable.visibility == .Hidden) continue;

            const pos = math.Vec2.init(item.transform.world_matrix.get(3, 0), item.transform.world_matrix.get(3, 1));

            const scale = math.Vec2.init(item.transform.world_matrix.get(0, 0), item.transform.world_matrix.get(1, 1));

            try self.sprite_batch.drawSprite(pos, scale, std.math.atan2(item.transform.world_matrix.get(1, 0), item.transform.world_matrix.get(0, 0)), item.renderable.material.tint, item.renderable.texture_id, item.transform.layer, @as(u32, @intFromEnum(item.renderable.material.blend_mode)));

            item.renderable.preRender();

            item.renderable.postRender();

            if (self.config.enable_debug and item.renderable.debug.show_bounds) {
                try self.drawDebugBounds(item.renderable.bounds.min, item.renderable.bounds.max);
            }
        }

        try self.sprite_batch.end();
    }

    fn updateTransformCache(self: *RenderSystem, entity: zigent.Id, transform: *Transform) !void {
        var world_matrix = transform.getLocalMatrix();
        var depth: f32 = transform.local_position.y();
        var layer = transform.layer;

        if (transform.parent) |parent| {
            var parent_entity: zigent.Id = 0;
            var found = false;
            while (self.ctx.isReady(parent_entity)) : (parent_entity += 1) {
                const check_transform = zigent.getAs(Transform, self.ctx, parent_entity, self.component_ids.transform) orelse continue;
                if (check_transform == parent) {
                    found = true;
                    break;
                }
            }

            if (found) {
                const parent_transform = self.transform_cache.get(parent_entity) orelse {
                    return;
                };
                world_matrix = parent_transform.world_matrix.mul(world_matrix);
                depth += parent_transform.depth;
                layer = parent_transform.layer;
            }
        }

        try self.transform_cache.put(entity, .{
            .world_matrix = world_matrix,
            .depth = depth,
            .layer = layer,
        });
    }

    fn drawDebugBounds(self: *RenderSystem, min: math.Vec2, max: math.Vec2) !void {
        const color = [4]f32{ 0.0, 1.0, 0.0, 1.0 };
        const corners = [_]math.Vec2{
            min,
            math.Vec2.init(max.x(), min.y()),
            max,
            math.Vec2.init(min.x(), max.y()),
        };

        for (corners) |corner| {
            try self.sprite_batch.drawSprite(corner, math.Vec2.init(2, 2), 0, color, 0, 0xFFFF, 0);
        }
    }
};

fn comparePriority(context: void, a: RenderQueueItem, b: RenderQueueItem) bool {
    _ = context;
    return @intFromEnum(a.renderable.priority) < @intFromEnum(b.renderable.priority);
}

fn comparePriorityAndDepth(context: void, a: RenderQueueItem, b: RenderQueueItem) bool {
    _ = context;
    const a_priority = @intFromEnum(a.renderable.priority);
    const b_priority = @intFromEnum(b.renderable.priority);
    if (a_priority != b_priority) {
        return a_priority < b_priority;
    }
    return a.transform.depth < b.transform.depth;
}

fn compareMaterial(context: void, a: RenderQueueItem, b: RenderQueueItem) bool {
    _ = context;
    if (a.renderable.texture_id != b.renderable.texture_id) {
        return a.renderable.texture_id < b.renderable.texture_id;
    }
    if (a.renderable.material.shader_id != b.renderable.material.shader_id) {
        return a.renderable.material.shader_id < b.renderable.material.shader_id;
    }
    return @intFromEnum(a.renderable.material.blend_mode) < @intFromEnum(b.renderable.material.blend_mode);
}
