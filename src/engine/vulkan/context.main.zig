const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const glfw = @import("mach-glfw");
const logger = @import("../core/logger.zig");
const math = @import("../core/math.zig");
const sprite = @import("sprite.zig");

const types = @import("context.types.zig");
const instance = @import("instance.zig");
const swapchain = @import("swapchain.zig");
const device = @import("device/logical.zig");
const physical = @import("device/physical.zig");
const sync = @import("sync.zig");
const commands = @import("commands.zig");
const pipeline = @import("pipeline.zig");
const col3 = @import("../core/math/col3/col3.zig");

const VERTICES_PER_QUAD = 4;

fn checkVkResult(result: c.VkResult) !void {
    return switch (result) {
        c.VK_SUCCESS => {},
        c.VK_ERROR_OUT_OF_HOST_MEMORY,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY,
        => error.OutOfMemory,
        c.VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
        c.VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        else => {
            logger.err("vulkan: unexpected error {}", .{result});
            return error.Unknown;
        },
    };
}

fn createSurface(vk_instance: c.VkInstance, window: glfw.Window) !c.VkSurfaceKHR {
    var surface: c.VkSurfaceKHR = undefined;
    const result = glfw.createWindowSurface(
        vk_instance,
        window,
        null,
        &surface,
    );

    if (result != 0) {
        logger.err("vulkan: failed to create window surface (glfw result: {})", .{result});
        return error.SurfaceCreationFailed;
    }

    return surface;
}

pub const Context = struct {
    inner: struct {
        instance: instance.Instance,
        surface: c.VkSurfaceKHR,
        physical_device: *physical.PhysicalDevice,
        device: device.Device,
        swapchain: *swapchain.Swapchain,
        sync_objects: sync.SyncObjects,
        command_pool: commands.CommandSystem.CommandPool,
        command_buffers: []commands.CommandSystem.CommandBuffer,
        pipeline: *pipeline.Pipeline,
        current_frame: u32,
        projection: [4][4]f32,
        frame_count: u64 = 0,
        last_fps_update: i128 = 0,
        frames_this_second: u32 = 0,
        last_frame_time: i128 = 0,
        frame_times: [60]f32 = [_]f32{0} ** 60,
        frame_time_index: usize = 0,
        config: types.Context.Config,
        clear_color: [4]f32 = .{ 0.0, 0.0, 0.0, 1.0 },
    },
    window: glfw.Window,
    allocator: std.mem.Allocator,

    pub fn init(window: glfw.Window, vk_instance: instance.Instance, config: types.Context.Config, alloc: std.mem.Allocator) !*Context {
        const self = try alloc.create(Context);
        errdefer alloc.destroy(self);

        const surface = try createSurface(vk_instance.handle, window);
        errdefer c.vkDestroySurfaceKHR(vk_instance.handle, surface, null);

        const phys_device = try physical.PhysicalDevice.selectBest(vk_instance.handle, surface, alloc);

        var vk_device = try device.Device.init(phys_device, .{
            .enable_robustness = false,
            .enable_dynamic_rendering = false,
            .enable_timeline_semaphores = true,
            .enable_synchronization2 = false,
            .enable_buffer_device_address = false,
            .enable_memory_priority = false,
            .enable_memory_budget = false,
            .enable_descriptor_indexing = false,
            .enable_maintenance4 = false,
            .enable_null_descriptors = false,
            .enable_shader_draw_parameters = false,
            .enable_host_query_reset = false,
        }, alloc);
        errdefer vk_device.deinit();

        var vk_sync = try sync.SyncObjects.init(
            vk_device.handle,
            .{
                .max_frames_in_flight = config.max_frames_in_flight,
                .enable_validation = config.instance_config.enable_validation,
                .enable_prediction = true,
                .batch_size = @min(config.max_frames_in_flight, 8),
            },
            alloc,
        );
        errdefer vk_sync.deinit();

        self.* = .{
            .inner = .{
                .instance = vk_instance,
                .surface = surface,
                .physical_device = phys_device,
                .device = vk_device,
                .swapchain = undefined,
                .sync_objects = vk_sync,
                .command_pool = undefined,
                .command_buffers = undefined,
                .pipeline = undefined,
                .current_frame = 0,
                .projection = undefined,
                .config = config,
            },
            .window = window,
            .allocator = alloc,
        };

        self.inner.command_pool = try commands.CommandSystem.CommandPool.init(
            vk_device.handle,
            vk_instance.handle,
            self.allocator,
            .{
                .queue_family_index = vk_device.getQueueFamilyIndices().graphics_family.?,
                .thread_safe = true,
                .enable_validation = config.instance_config.enable_validation,
                .enable_metrics = true,
            },
        );
        errdefer self.inner.command_pool.deinit();

        const command_buffers = try alloc.alloc(commands.CommandSystem.CommandBuffer, config.max_frames_in_flight);
        errdefer alloc.free(command_buffers);

        for (command_buffers) |*cmd_buf| {
            const buffer = try self.inner.command_pool.acquireBuffer(.{
                .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .usage = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .enable_validation = config.instance_config.enable_validation,
            });
            cmd_buf.* = buffer;
        }
        errdefer for (command_buffers) |*cmd_buf| self.inner.command_pool.releaseBuffer(cmd_buf);

        const fb_size = window.getFramebufferSize();

        var vk_pipeline = try pipeline.Pipeline.init(
            vk_device.handle,
            c.VK_FORMAT_B8G8R8A8_SRGB,
            .{ .width = @intCast(fb_size.width), .height = @intCast(fb_size.height) },
            alloc,
        );
        errdefer vk_pipeline.deinit();

        var vk_swapchain = try swapchain.Swapchain.init(
            vk_device.handle,
            vk_device.getQueueFamilyIndices(),
            surface,
            vk_device.getPhysicalDeviceHandle(),
            @intCast(fb_size.width),
            @intCast(fb_size.height),
            alloc,
            vk_pipeline.render_pass,
            swapchain.SwapchainConfig{
                .preferred_formats = &[_]c.VkFormat{
                    c.VK_FORMAT_B8G8R8A8_SRGB,
                    c.VK_FORMAT_R8G8B8A8_SRGB,
                },
                .preferred_color_space = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                .preferred_present_modes = &[_]c.VkPresentModeKHR{
                    c.VK_PRESENT_MODE_FIFO_KHR,
                    c.VK_PRESENT_MODE_MAILBOX_KHR,
                },
                .min_image_count = 2,
                .image_usage_flags = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    c.VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .transform_flags = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                .composite_alpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .old_swapchain = null,
                .enable_vsync = config.vsync,
                .enable_hdr = false,
                .enable_triple_buffering = config.max_frames_in_flight > 2,
                .enable_vrr = false,
                .enable_low_latency = !config.vsync,
                .enable_frame_pacing = true,
                .target_fps = 60,
                .power_save_mode = .Balanced,
            },
        );
        errdefer vk_swapchain.deinit();

        const projection = math.ortho(0, @floatFromInt(fb_size.width), @floatFromInt(fb_size.height), 0);

        logger.debug("framebuffer size: {}x{}", .{ fb_size.width, fb_size.height });
        logger.debug("projection matrix:", .{});
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection.get(0, 0), projection.get(0, 1), projection.get(0, 2), projection.get(0, 3) });
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection.get(1, 0), projection.get(1, 1), projection.get(1, 2), projection.get(1, 3) });
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection.get(2, 0), projection.get(2, 1), projection.get(2, 2), projection.get(2, 3) });
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection.get(3, 0), projection.get(3, 1), projection.get(3, 2), projection.get(3, 3) });

        self.inner.swapchain = vk_swapchain;
        self.inner.command_buffers = command_buffers;
        self.inner.pipeline = vk_pipeline;
        self.inner.projection = projection.toArray2D();

        return self;
    }

    pub fn deinit(self: *Context) void {
        for (self.inner.command_buffers) |*cmd_buf| {
            self.inner.command_pool.releaseBuffer(cmd_buf);
        }
        self.allocator.free(self.inner.command_buffers);
        self.inner.pipeline.deinit();
        self.inner.command_pool.deinit();
        self.inner.sync_objects.deinit();
        self.inner.swapchain.deinit();
        self.inner.device.deinit();

        c.vkDestroySurfaceKHR(self.inner.instance.handle, self.inner.surface, null);
        self.inner.instance.deinit();
        self.allocator.destroy(self);
    }

    pub fn setClearColor(self: *Context, r: f32, g: f32, b: f32, a: f32) void {
        self.inner.clear_color = .{ r, g, b, a };
    }

    pub fn endFrame(self: *Context, sprite_batch: *sprite.SpriteBatch) !void {
        const frame_start = std.time.nanoTimestamp();

        const fb_size = self.window.getFramebufferSize();
        if (fb_size.width == 0 or fb_size.height == 0) {
            self.inner.swapchain.handleWindowState(0, 0);
            std.time.sleep(16 * std.time.ns_per_ms);
            return;
        }

        const needs_resize = fb_size.width != self.inner.swapchain.extent.width or
            fb_size.height != self.inner.swapchain.extent.height;

        if (needs_resize) {
            try self.recreateSwapchain();
            return;
        }

        const image_available = self.inner.sync_objects.image_available_semaphores[self.inner.current_frame];
        const render_finished = self.inner.sync_objects.render_finished_semaphores[self.inner.current_frame];
        const in_flight_fence = self.inner.sync_objects.in_flight_fences[self.inner.current_frame];

        const fence_result = c.vkWaitForFences(
            self.inner.device.handle,
            1,
            &in_flight_fence,
            c.VK_TRUE,
            1000 * std.time.ns_per_ms,
        );

        if (fence_result != c.VK_SUCCESS) {
            logger.err("gpu: fence wait failed - attempting recovery", .{});
            try self.handleDeviceLost();
            return error.GPUHang;
        }

        try checkVkResult(c.vkResetFences(self.inner.device.handle, 1, &in_flight_fence));

        self.inner.swapchain.handleWindowState(@intCast(fb_size.width), @intCast(fb_size.height));

        const acquire_result = self.inner.swapchain.acquireNextImage(image_available) catch |err| {
            switch (err) {
                error.ImageAcquisitionFailed => {
                    try self.waitIdle();
                    try self.recreateSwapchain();
                    return;
                },
                else => return err,
            }
        };

        if (acquire_result.should_recreate) {
            try self.waitIdle();
            try self.recreateSwapchain();
            return;
        }

        const image_index = acquire_result.image_index;

        const cmd = &self.inner.command_buffers[self.inner.current_frame];
        try cmd.reset();
        try cmd.begin(c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        const clear_color = c.VkClearValue{
            .color = .{ .float32 = self.inner.clear_color },
        };

        const render_pass_info = c.VkRenderPassBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = self.inner.pipeline.render_pass,
            .framebuffer = self.inner.swapchain.framebuffers[image_index],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.inner.swapchain.extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
            .pNext = null,
        };

        c.vkCmdBeginRenderPass(cmd.handle, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(cmd.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.inner.pipeline.pipeline);

        const viewport = c.VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(self.inner.swapchain.extent.width),
            .height = @floatFromInt(self.inner.swapchain.extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        c.vkCmdSetViewport(cmd.handle, 0, 1, &viewport);

        const scissor = c.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.inner.swapchain.extent,
        };
        c.vkCmdSetScissor(cmd.handle, 0, 1, &scissor);

        c.vkCmdPushConstants(
            cmd.handle,
            self.inner.pipeline.pipeline_layout,
            c.VK_SHADER_STAGE_VERTEX_BIT,
            0,
            @sizeOf([4][4]f32),
            &self.inner.projection,
        );

        const vertex_buffers = [_]c.VkBuffer{
            sprite_batch.instance_buffer.handle,
        };
        const offsets = [_]c.VkDeviceSize{0};

        c.vkCmdBindVertexBuffers(cmd.handle, 0, 1, &vertex_buffers, &offsets);

        for (sprite_batch.draw_calls.items) |batch| {
            c.vkCmdDraw(
                cmd.handle,
                4,
                batch.instance_count,
                0,
                0,
            );
        }

        c.vkCmdEndRenderPass(cmd.handle);
        try cmd.end();

        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const wait_semaphores = [_]c.VkSemaphore{image_available};
        const signal_semaphores = [_]c.VkSemaphore{render_finished};

        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd.handle,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphores,
            .pNext = null,
        };

        try checkVkResult(c.vkQueueSubmit(
            self.inner.device.graphics_queue,
            1,
            &submit_info,
            in_flight_fence,
        ));

        const present_semaphores = [_]c.VkSemaphore{render_finished};
        const should_recreate = try self.inner.swapchain.presentImage(image_index, &present_semaphores);
        if (should_recreate) {
            try self.waitIdle();
            try self.recreateSwapchain();
            return;
        }

        const now = std.time.nanoTimestamp();
        self.inner.frames_this_second += 1;
        if (now - self.inner.last_fps_update >= 5 * std.time.ns_per_s) {
            const frame_time = if (self.inner.frame_count > 0)
                @as(f32, @floatFromInt(now - self.inner.last_frame_time)) / @as(f32, @floatFromInt(std.time.ns_per_ms))
            else
                0;

            var avg_frame_time: f32 = 0;
            for (self.inner.frame_times) |time| {
                avg_frame_time += time;
            }
            avg_frame_time /= @as(f32, @floatFromInt(self.inner.frame_times.len));

            const fps = @as(f32, @floatFromInt(self.inner.frames_this_second)) / 5.0;
            logger.info("perf: {d:.1} fps, frame_time={d:.2}ms (avg={d:.2}ms)", .{
                fps,
                frame_time,
                avg_frame_time,
            });

            self.inner.frames_this_second = 0;
            self.inner.last_fps_update = now;
        }

        const frame_end = std.time.nanoTimestamp();
        const frame_time = @as(f32, @floatFromInt(frame_end - frame_start)) / @as(f32, @floatFromInt(std.time.ns_per_ms));
        self.inner.frame_times[self.inner.frame_time_index] = frame_time;
        self.inner.frame_time_index = (self.inner.frame_time_index + 1) % self.inner.frame_times.len;
        self.inner.last_frame_time = frame_start;
        self.inner.frame_count += 1;

        self.inner.current_frame = (self.inner.current_frame + 1) % self.inner.sync_objects.max_frames_in_flight;
    }

    pub fn waitIdle(self: *Context) !void {
        try checkVkResult(c.vkDeviceWaitIdle(self.inner.device.handle));
    }

    pub fn recreateSwapchain(self: *Context) !void {
        const fb_size = self.window.getFramebufferSize();
        if (fb_size.width == 0 or fb_size.height == 0) {
            logger.debug("window: minimized, skipping swapchain recreation", .{});
            self.inner.swapchain.handleWindowState(0, 0);
            return;
        }

        logger.info("swapchain: recreating ({d}x{d})", .{ fb_size.width, fb_size.height });

        var all_fences_signaled = true;
        for (self.inner.sync_objects.in_flight_fences) |fence| {
            const result = c.vkWaitForFences(
                self.inner.device.handle,
                1,
                &fence,
                c.VK_TRUE,
                500 * std.time.ns_per_ms,
            );
            if (result != c.VK_SUCCESS) {
                all_fences_signaled = false;
                logger.warn("vulkan: fence wait timed out during resize, will force cleanup", .{});
                break;
            }
        }

        if (!all_fences_signaled) {
            _ = c.vkQueueWaitIdle(self.inner.device.graphics_queue);
            _ = c.vkQueueWaitIdle(self.inner.device.present_queue);

            for (self.inner.sync_objects.in_flight_fences) |fence| {
                _ = c.vkResetFences(self.inner.device.handle, 1, &fence);
            }

            const device_wait_result = c.vkDeviceWaitIdle(self.inner.device.handle);
            if (device_wait_result != c.VK_SUCCESS) {
                logger.err("vulkan: failed to wait for device idle: {}", .{device_wait_result});
                return error.DeviceLost;
            }
        }

        const old_swapchain = self.inner.swapchain;
        errdefer old_swapchain.deinit();

        try self.inner.command_pool.reset();
        for (self.inner.command_buffers) |*cmd_buf| {
            try cmd_buf.reset();
        }

        self.inner.swapchain = try swapchain.Swapchain.init(
            self.inner.device.handle,
            self.inner.device.getQueueFamilyIndices(),
            self.inner.surface,
            self.inner.device.getPhysicalDeviceHandle(),
            @intCast(fb_size.width),
            @intCast(fb_size.height),
            self.allocator,
            self.inner.pipeline.render_pass,
            swapchain.SwapchainConfig{
                .preferred_formats = &[_]c.VkFormat{
                    c.VK_FORMAT_B8G8R8A8_SRGB,
                    c.VK_FORMAT_R8G8B8A8_SRGB,
                },
                .preferred_color_space = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                .preferred_present_modes = &[_]c.VkPresentModeKHR{
                    c.VK_PRESENT_MODE_FIFO_KHR,
                    c.VK_PRESENT_MODE_MAILBOX_KHR,
                },
                .min_image_count = 2,
                .image_usage_flags = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    c.VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .transform_flags = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                .composite_alpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .old_swapchain = old_swapchain.handle,
                .enable_vsync = self.inner.config.vsync,
                .enable_hdr = false,
                .enable_triple_buffering = self.inner.config.max_frames_in_flight > 2,
                .enable_vrr = false,
                .enable_low_latency = !self.inner.config.vsync,
                .enable_frame_pacing = true,
                .target_fps = 60,
                .power_save_mode = .Balanced,
            },
        );

        old_swapchain.deinit();

        self.inner.projection = math.ortho(0, @floatFromInt(fb_size.width), @floatFromInt(fb_size.height), 0).toArray2D();
        logger.debug("window: updated projection matrix for new size", .{});

        self.inner.current_frame = 0;
        self.inner.frame_count = 0;
        self.inner.frames_this_second = 0;
        self.inner.last_fps_update = std.time.nanoTimestamp();

        try self.prepareFirstFrame();
    }

    pub fn handleDeviceLost(self: *Context) !void {
        logger.err("vulkan: device lost - attempting recovery", .{});

        std.time.sleep(100 * std.time.ns_per_ms);

        _ = c.vkQueueWaitIdle(self.inner.device.graphics_queue);

        for (self.inner.command_buffers) |*cmd_buf| {
            cmd_buf.reset() catch |err| {
                logger.err("vulkan: failed to reset command buffer during recovery: {}", .{err});
            };
        }

        for (self.inner.sync_objects.in_flight_fences) |fence| {
            const fence_result = c.vkResetFences(self.inner.device.handle, 1, &fence);
            if (fence_result != c.VK_SUCCESS) {
                logger.err("vulkan: failed to reset fence during recovery: {}", .{fence_result});
            }
        }

        for (self.inner.sync_objects.image_available_semaphores) |_| {
            _ = c.vkQueueWaitIdle(self.inner.device.graphics_queue);
        }
        for (self.inner.sync_objects.render_finished_semaphores) |_| {
            _ = c.vkQueueWaitIdle(self.inner.device.graphics_queue);
        }

        self.inner.current_frame = 0;
        self.inner.frame_count = 0;
        self.inner.frames_this_second = 0;
        self.inner.last_fps_update = std.time.nanoTimestamp();

        logger.info("vulkan: device recovery attempted", .{});
    }

    pub fn handleResize(self: *Context) !void {
        try checkVkResult(c.vkDeviceWaitIdle(self.inner.device.handle));

        var width: i32 = 0;
        var height: i32 = 0;
        var retry_count: u32 = 0;
        const MAX_RETRIES = 10;

        while (width == 0 or height == 0) {
            if (retry_count >= MAX_RETRIES) {
                logger.err("window: failed to get valid framebuffer size after {d} retries", .{MAX_RETRIES});
                return error.InvalidFramebufferSize;
            }

            const size = self.window.getFramebufferSize();
            width = size.width;
            height = size.height;

            if (width == 0 or height == 0) {
                retry_count += 1;
                std.time.sleep(50 * std.time.ns_per_ms);
                glfw.waitEvents();
            }
        }

        self.recreateSwapchain() catch |err| {
            logger.err("window: failed to recreate swapchain during resize: {}", .{err});
            switch (err) {
                error.DeviceLost => {
                    logger.err("window: device lost during resize, attempting recovery...", .{});

                    return err;
                },
                error.SurfaceLost => {
                    logger.err("window: surface lost during resize, attempting recreation...", .{});

                    return err;
                },
                error.OutOfMemory => {
                    logger.err("window: out of memory during swapchain recreation", .{});
                    return err;
                },
                error.InitializationFailed => {
                    logger.err("window: swapchain initialization failed", .{});
                    return err;
                },
                error.Unknown => {
                    logger.err("window: unknown error during swapchain recreation", .{});
                    return err;
                },
                else => return err,
            }
        };
    }

    pub fn prepareFirstFrame(self: *Context) !void {
        try checkVkResult(c.vkDeviceWaitIdle(self.inner.device.handle));

        const cmd = &self.inner.command_buffers[0];
        try cmd.reset();
        try cmd.begin(c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        const clear_color = c.VkClearValue{
            .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } },
        };

        const render_pass_info = c.VkRenderPassBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = self.inner.pipeline.render_pass,
            .framebuffer = self.inner.swapchain.framebuffers[0],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.inner.swapchain.extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
            .pNext = null,
        };

        c.vkCmdBeginRenderPass(cmd.handle, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdEndRenderPass(cmd.handle);
        try cmd.end();

        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd.handle,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
            .pNext = null,
        };

        try checkVkResult(c.vkQueueSubmit(
            self.inner.device.graphics_queue,
            1,
            &submit_info,
            null,
        ));

        try checkVkResult(c.vkDeviceWaitIdle(self.inner.device.handle));
    }
};
