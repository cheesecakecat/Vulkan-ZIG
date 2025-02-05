const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const glfw = @import("mach-glfw");
const logger = @import("engine/core/logger.zig");
const vk = @import("engine/vulkan/context.main.zig");
const vk_types = @import("engine/vulkan/context.types.zig");
const sprite = @import("engine/vulkan/sprite.zig");
const Instance = @import("engine/vulkan/instance.zig");
const zigent = @import("deps/zigent.zig");
const math = @import("engine/core/math.zig");
const col3 = @import("engine/core/math/col3/col3.zig");

// Components
const Transform = @import("engine/core/entity-component-system/components/transform.zig").Transform;
const Renderable = @import("engine/core/entity-component-system/components/renderable.zig").Renderable;

// Systems
const RenderSystem = @import("engine/core/entity-component-system/systems/render.zig").RenderSystem;

const APP_VERSION = vk_types.makeVersion(0, 1, 0);
const ENGINE_VERSION = vk_types.makeVersion(0, 1, 0);

const MAX_SPRITES = 500_000;
const SPRITE_SIZE = 32.0;
const SPRITE_SPEED = 100.0;
const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;

fn randomColor() [4]f32 {
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const random = prng.random();
    return .{
        random.float(f32), // r
        random.float(f32), // g
        random.float(f32), // b
        1.0, // alpha
    };
}

const DemoSprite = struct {
    position: math.Vec2,
    color: [4]f32,
    scale: f32,
    layer: u32,
};

const Game = struct {
    allocator: std.mem.Allocator,
    window: glfw.Window,
    vk_ctx: *vk.Context,
    sprite_batch: *sprite.SpriteBatch,

    ecs_ctx: *zigent.Context,
    transform_id: zigent.Id,
    renderable_id: zigent.Id,
    render_system: *RenderSystem,

    should_quit: bool = false,
    last_time: i64,

    pub fn init(allocator: std.mem.Allocator) !*Game {
        const self = try allocator.create(Game);
        errdefer allocator.destroy(self);

        errdefer logger.deinit();

        logger.info("main: initializing...", .{});

        if (!glfw.init(.{})) {
            logger.err("glfw: failed to initialize - {?s}", .{glfw.getErrorString()});
            return error.GLFWInitFailed;
        }

        const window = glfw.Window.create(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            "VK-ZIG Demo",
            null,
            null,
            .{
                .client_api = .no_api,
                .resizable = true,
                .visible = false,
            },
        ) orelse {
            logger.err("glfw: failed to create window", .{});
            return error.WindowCreationFailed;
        };

        const vk_instance = try Instance.Instance.init(.{
            .application_name = "VK-ZIG Demo",
            .application_version = APP_VERSION,
            .engine_name = "VK-ZIG Engine",
            .engine_version = ENGINE_VERSION,
            .api_version = c.VK_API_VERSION_1_3,
            .enable_validation = true,
            .enable_debug_utils = true,
            .enable_surface_extensions = true,
            .enable_portability = true,
            .validation_features = .{},
            .debug_severity = .{
                .verbose = true,
                .info = true,
                .warning = true,
                .severity_error = true,
                .perf_warning = true,
            },
            .debug_message_type = .{
                .general = true,
                .validation = true,
                .performance = true,
            },
        }, allocator);

        const vk_ctx = try vk.Context.init(window, vk_instance, .{
            .instance_config = .{
                .application_name = "VK-ZIG Demo",
                .engine_name = "VK-ZIG Engine",
                .application_version = APP_VERSION,
                .engine_version = ENGINE_VERSION,
                .api_version = c.VK_API_VERSION_1_3,
                .enable_validation = true,
                .enable_debug_utils = true,
                .enable_surface_extensions = true,
                .enable_portability = true,
            },
            .vsync = true,
            .max_frames_in_flight = 2,
        }, allocator);

        const sprite_batch = try sprite.SpriteBatch.init(
            vk_ctx.inner.device.handle,
            vk_ctx.inner.instance.handle,
            vk_ctx.inner.physical_device.handle,
            vk_ctx.inner.device.getQueueFamilyIndices().graphics_family.?,
            MAX_SPRITES,
            null,
            allocator,
        );

        const ecs_ctx = try zigent.Context.init(allocator);

        const transform_id = try ecs_ctx.registerComponent(Transform);
        const renderable_id = try ecs_ctx.registerComponent(Renderable);

        const render_system = try RenderSystem.init(
            allocator,
            ecs_ctx,
            transform_id,
            renderable_id,
            sprite_batch,
            .{
                .max_sprites = MAX_SPRITES,
                .enable_culling = true,
                .enable_debug = true,
            },
        );

        self.* = .{
            .allocator = allocator,
            .window = window,
            .vk_ctx = vk_ctx,
            .sprite_batch = sprite_batch,
            .ecs_ctx = ecs_ctx,
            .transform_id = transform_id,
            .renderable_id = renderable_id,
            .render_system = render_system,
            .last_time = std.time.milliTimestamp(),
        };

        try self.setupInitialScene();

        try self.vk_ctx.prepareFirstFrame();

        self.vk_ctx.setClearColor(col3.Colors.black.r, col3.Colors.black.g, col3.Colors.black.b, 1.0);

        logger.info("main: initialization complete", .{});

        window.show();

        return self;
    }

    pub fn deinit(self: *Game) void {
        logger.info("main: shutting down...", .{});

        self.render_system.deinit();
        self.ecs_ctx.deinit();
        self.sprite_batch.deinit();
        self.vk_ctx.deinit();
        self.window.destroy();
        glfw.terminate();
        logger.deinit();

        self.allocator.destroy(self);
        logger.info("main: shutdown complete", .{});
    }

    fn setupInitialScene(self: *Game) !void {
        const entity = self.ecs_ctx.create();

        var transform = Transform.init();
        _ = transform.setPosition(
            @as(f32, @floatFromInt(WINDOW_WIDTH)) / 2.0,
            @as(f32, @floatFromInt(WINDOW_HEIGHT)) / 2.0,
            .World,
        );
        _ = transform.setScale(10, 10, .World);
        try self.ecs_ctx.add(entity, self.transform_id, transform);

        var renderable = Renderable.init();
        _ = renderable.setTexture(0);
        const color = randomColor();
        _ = renderable.setTint(color[0], color[1], color[2], color[3]);
        try self.ecs_ctx.add(entity, self.renderable_id, renderable);
    }

    pub fn update(self: *Game) !void {
        const current_time = std.time.milliTimestamp();
        const delta_time = @as(f32, @floatFromInt(current_time - self.last_time)) / 1000.0;
        self.last_time = current_time;

        glfw.pollEvents();
        if (self.window.shouldClose()) {
            self.should_quit = true;
            return;
        }

        try self.ecs_ctx.updateSystems(delta_time);
    }

    pub fn draw(self: *Game) !void {
        const current_time = std.time.milliTimestamp();
        const delta_time = @as(f32, @floatFromInt(current_time - self.last_time)) / 1000.0;
        self.last_time = current_time;

        try self.render_system.update(delta_time);

        try self.vk_ctx.endFrame(self.sprite_batch);
    }

    pub fn run(self: *Game) !void {
        logger.info("main: starting main loop", .{});

        while (!self.should_quit) {
            try self.update();
            try self.draw();
        }

        try self.vk_ctx.waitIdle();
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    try logger.init();
    defer logger.deinit();

    var game = try Game.init(gpa.allocator());
    defer game.deinit();

    try game.run();
}
