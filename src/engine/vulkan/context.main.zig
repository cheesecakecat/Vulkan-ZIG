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
const device = @import("device.zig");
const swapchain = @import("swapchain.zig");
const sync = @import("sync.zig");
const commands = @import("commands.zig");
const pipeline = @import("pipeline.zig");

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

fn createSurface(vk_instance: ?*c.struct_VkInstance_T, window: glfw.Window) !c.VkSurfaceKHR {
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
        device: device.Device,
        swapchain: *swapchain.Swapchain,
        sync_objects: sync.SyncObjects,
        command_pool: commands.CommandPool,
        command_buffers: []commands.CommandBuffer,
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
    },
    window: glfw.Window,
    allocator: std.mem.Allocator,

    pub fn init(window: glfw.Window, vk_instance: instance.Instance, config: types.Context.Config, alloc: std.mem.Allocator) !*Context {
        const self = try alloc.create(Context);
        errdefer alloc.destroy(self);

        const surface = try createSurface(vk_instance.handle, window);
        errdefer c.vkDestroySurfaceKHR(vk_instance.handle, surface, null);

        var vk_device = try device.Device.init(vk_instance.handle, surface, alloc);
        errdefer vk_device.deinit();

        var vk_sync = try sync.SyncObjects.init(
            vk_device.logical,
            config.max_frames_in_flight,
            alloc,
        );
        errdefer vk_sync.deinit();

        self.* = .{
            .inner = .{
                .instance = vk_instance,
                .surface = surface,
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

        self.inner.command_pool = try commands.CommandPool.init(
            vk_device.logical,
            vk_device.queue_indices.graphics_family.?,
        );
        errdefer self.inner.command_pool.deinit();

        const command_buffers = try alloc.alloc(commands.CommandBuffer, config.max_frames_in_flight);
        errdefer alloc.free(command_buffers);

        for (command_buffers) |*cmd_buf| {
            cmd_buf.* = try commands.CommandBuffer.init(&self.inner.command_pool, c.VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        }
        errdefer for (command_buffers) |*cmd_buf| cmd_buf.deinit();

        const fb_size = window.getFramebufferSize();

        var vk_pipeline = try pipeline.Pipeline.init(
            vk_device.logical,
            c.VK_FORMAT_B8G8R8A8_SRGB,
            .{ .width = @intCast(fb_size.width), .height = @intCast(fb_size.height) },
            alloc,
        );
        errdefer vk_pipeline.deinit();

        var vk_swapchain = try swapchain.Swapchain.init(
            vk_device.logical,
            vk_device.queue_indices,
            surface,
            vk_device.physical,
            @intCast(fb_size.width),
            @intCast(fb_size.height),
            alloc,
            vk_pipeline.render_pass,
            null,
        );
        errdefer vk_swapchain.deinit();

        const projection = math.ortho(0, @floatFromInt(fb_size.width), @floatFromInt(fb_size.height), 0);

        logger.debug("Framebuffer size: {}x{}", .{ fb_size.width, fb_size.height });
        logger.debug("Projection matrix:", .{});
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection[0][0], projection[0][1], projection[0][2], projection[0][3] });
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection[1][0], projection[1][1], projection[1][2], projection[1][3] });
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection[2][0], projection[2][1], projection[2][2], projection[2][3] });
        logger.debug("  [{d:>10.4}, {d:>10.4}, {d:>10.4}, {d:>10.4}]", .{ projection[3][0], projection[3][1], projection[3][2], projection[3][3] });

        self.inner.swapchain = vk_swapchain;
        self.inner.command_buffers = command_buffers;
        self.inner.pipeline = vk_pipeline;
        self.inner.projection = projection;

        return self;
    }

    pub fn deinit(self: *Context) void {
        for (self.inner.command_buffers) |*cmd_buf| {
            cmd_buf.deinit();
        }
        self.allocator.free(self.inner.command_buffers);
        self.inner.pipeline.deinit();
        self.inner.command_pool.deinit();
        self.inner.sync_objects.deinit();
        self.inner.swapchain.deinit();
        self.inner.device.deinit();
        c.vkDestroySurfaceKHR(self.inner.instance.handle, self.inner.surface, null);
        self.allocator.destroy(self);
    }

    pub fn endFrame(self: *Context, sprite_batch: *sprite.SpriteBatch) !void {
        const frame_start = std.time.nanoTimestamp();

        const fb_size = self.window.getFramebufferSize();
        if (fb_size.width == 0 or fb_size.height == 0) {
            self.inner.swapchain.handleWindowState(0, 0);

            std.time.sleep(16 * std.time.ns_per_ms);
            return;
        }
        self.inner.swapchain.handleWindowState(@intCast(fb_size.width), @intCast(fb_size.height));

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

        const image_available = self.inner.sync_objects.image_available_semaphores[self.inner.current_frame];
        const render_finished = self.inner.sync_objects.render_finished_semaphores[self.inner.current_frame];
        const in_flight_fence = self.inner.sync_objects.in_flight_fences[self.inner.current_frame];

        const fence_result = c.vkWaitForFences(
            self.inner.device.logical,
            1,
            &in_flight_fence,
            c.VK_TRUE,
            std.time.ns_per_s,
        );

        if (fence_result == c.VK_TIMEOUT) {
            logger.err("gpu: fence wait timeout - possible GPU hang", .{});
            return error.GPUHang;
        }
        try checkVkResult(fence_result);

        try checkVkResult(c.vkResetFences(self.inner.device.logical, 1, &in_flight_fence));

        const acquire_result = try self.inner.swapchain.acquireNextImage(image_available);
        if (acquire_result.should_recreate) {
            logger.debug("swapchain: needs recreation", .{});
            try self.recreateSwapchain();
            return;
        }
        const image_index = acquire_result.image_index;

        const cmd = &self.inner.command_buffers[self.inner.current_frame];
        try cmd.reset();
        try cmd.begin(c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        const clear_color = c.VkClearValue{
            .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } },
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

        const vertex_buffers = [_]c.VkBuffer{sprite_batch.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(cmd.handle, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindIndexBuffer(cmd.handle, sprite_batch.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        if (sprite_batch.sprite_count > 0) {
            c.vkCmdDrawIndexed(cmd.handle, sprite_batch.sprite_count * 6, 1, 0, 0, 0);
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
            logger.debug("swapchain: needs recreation after present", .{});
            try self.recreateSwapchain();
            return;
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
        _ = c.vkDeviceWaitIdle(self.inner.device.logical);
    }

    pub fn recreateSwapchain(self: *Context) !void {
        _ = c.vkDeviceWaitIdle(self.inner.device.logical);

        const fb_size = self.window.getFramebufferSize();
        if (fb_size.width == 0 or fb_size.height == 0) {
            logger.debug("window: minimized, skipping swapchain recreation", .{});
            self.inner.swapchain.handleWindowState(0, 0);
            return;
        }

        logger.info("swapchain: recreating ({d}x{d})", .{ fb_size.width, fb_size.height });

        const old_swapchain = self.inner.swapchain;
        var config = old_swapchain.config;
        config.old_swapchain = old_swapchain.handle;

        self.inner.swapchain = try swapchain.Swapchain.init(
            self.inner.device.logical,
            self.inner.device.queue_indices,
            self.inner.surface,
            self.inner.device.physical,
            @intCast(fb_size.width),
            @intCast(fb_size.height),
            self.allocator,
            self.inner.pipeline.render_pass,
            config,
        );

        old_swapchain.deinit();

        self.inner.projection = math.ortho(0, @floatFromInt(fb_size.width), @floatFromInt(fb_size.height), 0);
        logger.debug("window: updated projection matrix for new size", .{});
    }

    pub fn handleResize(self: *Context) !void {
        var width: i32 = 0;
        var height: i32 = 0;
        while (width == 0 or height == 0) {
            const size = self.inner.window.getFramebufferSize();
            width = size.width;
            height = size.height;
            glfw.waitEvents();
        }

        try self.recreateSwapchain();
    }

    pub fn prepareFirstFrame(self: *Context) !void {
        // Wait for device to be fully ready
        try checkVkResult(c.vkDeviceWaitIdle(self.inner.device.logical));

        // Pre-warm the pipeline cache
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

        // Submit and wait for completion
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

        try checkVkResult(c.vkDeviceWaitIdle(self.inner.device.logical));
    }
};
