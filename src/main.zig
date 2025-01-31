// NOTE: It works!
// TODO: Fix performance warnings.
// NOTE: I love Vulkan validation.

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const glfw = @import("mach-glfw");
const logger = @import("engine/core/logger.zig");
const allocator = @import("engine/core/allocator.zig");
const vk = @import("engine/vulkan/context.main.zig");
const vk_types = @import("engine/vulkan/context.types.zig");
const sprite = @import("engine/vulkan/sprite.zig");
const Instance = @import("engine/vulkan/instance.zig");

const ENGINE_VERSION = c.VK_MAKE_VERSION(0, 1, 0);
const APP_VERSION = c.VK_MAKE_VERSION(0, 1, 0);

pub fn main() !void {
    try logger.init();
    defer logger.deinit();

    logger.info("app: starting vulkan sprite renderer demo", .{});

    if (!glfw.init(.{})) {
        logger.err("glfw: failed to initialize - {?s}", .{glfw.getErrorString()});
        return error.GLFWInitFailed;
    }
    defer glfw.terminate();

    const window = glfw.Window.create(800, 600, "VK-ZIG Demo", null, null, .{
        .client_api = .no_api,
        .resizable = true,
        .visible = false,
    }) orelse {
        logger.err("glfw: failed to create window", .{});
        return error.WindowCreationFailed;
    };
    defer window.destroy();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const instance_config = Instance.InstanceConfig{
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
    };

    var instance = Instance.Instance.init(instance_config, alloc) catch |err| {
        logger.err("vulkan: failed to create instance: {}", .{err});
        return err;
    };
    defer instance.deinit();

    logger.info("app: vulkan instance created successfully", .{});
    logger.info("API Version: {}.{}.{}", .{
        c.VK_VERSION_MAJOR(instance.getAPIVersion()),
        c.VK_VERSION_MINOR(instance.getAPIVersion()),
        c.VK_VERSION_PATCH(instance.getAPIVersion()),
    });

    var context = try vk.Context.init(window, instance, .{
        .instance_config = instance_config,
        .vsync = true,
        .max_frames_in_flight = 2,
    }, alloc);
    defer context.deinit();

    try context.prepareFirstFrame();

    window.show();

    window.setUserPointer(context);
    window.setIconifyCallback((struct {
        pub fn callback(win: glfw.Window, iconified: bool) void {
            const ctx = win.getUserPointer(vk.Context) orelse {
                logger.err("window: failed to get context in iconify callback", .{});
                return;
            };

            if (iconified) {
                logger.debug("window: minimized, stopping render loop", .{});
                ctx.inner.swapchain.handleWindowState(0, 0);
            } else {
                logger.debug("window: restored, resuming render loop", .{});
                ctx.inner.swapchain.handleWindowState(800, 600);
            }
        }
    }).callback);

    window.setFocusCallback((struct {
        pub fn callback(win: glfw.Window, focused: bool) void {
            const ctx = win.getUserPointer(vk.Context) orelse {
                logger.err("window: failed to get context in focus callback", .{});
                return;
            };

            if (!focused) {
                logger.debug("window: lost focus, pausing render loop", .{});
                ctx.inner.swapchain.handleWindowState(0, 0);
            } else {
                logger.debug("window: gained focus, resuming render loop", .{});
                ctx.inner.swapchain.handleWindowState(800, 600);
            }
        }
    }).callback);

    logger.info("window: created display surface 800x600", .{});

    var sprite_batch = try sprite.SpriteBatch.init(
        context.inner.device.handle,
        context.inner.device.physical_device,
        context.inner.device.queues.graphics.family,
        null,
        10000,
        alloc,
    );
    defer sprite_batch.deinit();

    const color = [4]f32{ 1.0, 0.0, 0.5, 1.0 };
    const uv_rect = [4]f32{ 0.0, 0.0, 1.0, 1.0 };

    var last_time = std.time.milliTimestamp();
    var frame_count: u32 = 0;
    var fps_timer: f32 = 0;

    while (!window.shouldClose()) {
        const current_time = std.time.milliTimestamp();
        const delta_time = @as(f32, @floatFromInt(current_time - last_time)) / 1000.0;
        last_time = current_time;

        glfw.pollEvents();

        fps_timer += delta_time;
        frame_count += 1;
        if (fps_timer >= 1.0) {
            const fps = @as(f32, @floatFromInt(frame_count)) / fps_timer;
            logger.info("FPS: {d:.1}", .{fps});
            fps_timer = 0;
            frame_count = 0;
        }

        try sprite_batch.begin();

        var y: f32 = 100;
        while (y < 500) : (y += 50) {
            var x: f32 = 100;
            while (x < 700) : (x += 50) {
                try sprite_batch.draw(
                    .{ x, y },
                    .{ 40, 40 },
                    0,
                    color,
                    0,
                    uv_rect,
                    0.0,
                    0,
                );
            }
        }

        try sprite_batch.end();
        try context.endFrame(sprite_batch);

        instance.update();

        switch (instance.power.mode) {
            .power_saving => {
                context.inner.config.vsync = true;
                context.inner.config.max_frames_in_flight = 1;
            },
            .performance => {
                context.inner.config.vsync = false;
                context.inner.config.max_frames_in_flight = 3;
            },
            .normal => {
                context.inner.config.vsync = true;
                context.inner.config.max_frames_in_flight = 2;
            },
        }

        if (instance.power.state.thermal_throttling) {
            logger.warn("thermal throttling active - reducing workload", .{});
            context.inner.config.max_frames_in_flight = 1;
        }
    }

    try context.waitIdle();
}
