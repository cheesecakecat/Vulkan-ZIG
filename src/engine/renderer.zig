const std = @import("std");
const Window = @import("window.zig").Window;
const core = @import("vulkan/core.zig");
const device = @import("vulkan/device.zig");
const SwapChain = @import("vulkan/swapchain.zig").SwapChain;
const Pipeline = @import("vulkan/pipeline.zig").Pipeline;
const sync = @import("vulkan/sync.zig");
const command = @import("vulkan/command.zig");
const glfw = @import("mach-glfw");

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const Renderer = struct {
    window: *Window,
    instance: c.VkInstance,
    surface: c.VkSurfaceKHR,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    graphics_queue: c.VkQueue,
    present_queue: c.VkQueue,
    swapchain: SwapChain,
    pipeline: Pipeline,
    command_pool: *command.CommandPool,
    command_buffers: []*command.CommandBuffer,
    sync_objects: sync.SyncObjects,
    current_frame: u32,
    allocator: std.mem.Allocator,

    pub fn init(window: *Window, allocator: std.mem.Allocator) !Renderer {
        const instance = try core.createInstance();
        errdefer c.vkDestroyInstance(instance, null);

        var surface: c.VkSurfaceKHR = undefined;
        if (glfw.createWindowSurface(instance, window.handle.?, null, &surface) != c.VK_SUCCESS) {
            return error.SurfaceCreationFailed;
        }
        errdefer c.vkDestroySurfaceKHR(instance, surface, null);

        const physical_device = try device.pickPhysicalDevice(instance, surface);
        const logical_device_info = try device.createLogicalDevice(physical_device, surface, allocator);
        errdefer c.vkDestroyDevice(logical_device_info.device, null);

        // Get surface format for pipeline creation
        var swapchain_support = try device.querySwapChainSupport(physical_device, surface);
        defer swapchain_support.deinit();
        const surface_format = SwapChain.chooseSwapSurfaceFormat(swapchain_support.formats);

        // Create pipeline with surface format
        const pipeline = try Pipeline.init(
            logical_device_info.device,
            surface_format.format,
            @embedFile("../shaders/triangle.vert.spv"),
            @embedFile("../shaders/triangle.frag.spv"),
        );

        // Create swapchain with render pass
        const swapchain = try SwapChain.init(
            physical_device,
            logical_device_info.device,
            surface,
            window,
            allocator,
            pipeline.render_pass,
        );

        // Create command pool with reusable buffers
        const queue_family_indices = try device.findQueueFamilies(physical_device, surface);
        const command_pool = try command.CommandPool.init(
            logical_device_info.device,
            queue_family_indices.graphics_family.?,
            .{
                .is_reusable = true,
                .allow_reset = true,
            },
            .{ // Command pool configuration
                .initial_buffer_count = @intCast(swapchain.framebuffers.len),
                .max_free_buffers = 32,
                .thread_local = false,
                .batch_commands = true,
                .cache_state = true,
            },
            allocator,
        );
        errdefer command_pool.*.deinit();

        // Create command buffers
        var command_buffers = std.ArrayList(*command.CommandBuffer).init(allocator);
        errdefer command_buffers.deinit();

        // Allocate one command buffer per swapchain image
        var i: usize = 0;
        while (i < swapchain.framebuffers.len) : (i += 1) {
            const buffer = try command_pool.getBuffer();
            try command_buffers.append(buffer);
        }

        const command_buffers_slice = try command_buffers.toOwnedSlice();
        command_buffers.deinit();

        try recordCommandBuffers(command_buffers_slice, pipeline, swapchain);

        var sync_objects = try sync.SyncObjects.init(logical_device_info.device, allocator);
        errdefer sync_objects.deinit();

        return Renderer{
            .window = window,
            .instance = instance,
            .surface = surface,
            .physical_device = physical_device,
            .device = logical_device_info.device,
            .graphics_queue = logical_device_info.graphics_queue,
            .present_queue = logical_device_info.present_queue,
            .swapchain = swapchain,
            .pipeline = pipeline,
            .command_pool = command_pool,
            .command_buffers = command_buffers_slice,
            .sync_objects = sync_objects,
            .current_frame = 0,
            .allocator = allocator,
        };
    }

    fn recordCommandBuffers(
        buffers: []*command.CommandBuffer,
        pipeline: Pipeline,
        swapchain: SwapChain,
    ) !void {
        const clear_color = c.VkClearValue{
            .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } },
        };
        const clear_values = [_]c.VkClearValue{clear_color};

        for (buffers, 0..) |buffer, i| {
            try buffer.begin(.{ .one_time_submit = true });
            try buffer.beginRenderPass(
                pipeline.render_pass,
                swapchain.framebuffers[i],
                swapchain.extent,
                &clear_values,
                c.VK_SUBPASS_CONTENTS_INLINE,
            );

            // Set dynamic state
            try buffer.setViewport(
                0,
                0,
                @floatFromInt(swapchain.extent.width),
                @floatFromInt(swapchain.extent.height),
                0,
                1,
            );
            try buffer.setScissor(
                0,
                0,
                swapchain.extent.width,
                swapchain.extent.height,
            );

            try buffer.bindPipeline(pipeline.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS);
            try buffer.draw(3, 1, 0, 0); // Triangle
            try buffer.endRenderPass();
            try buffer.end();
        }
    }

    pub fn drawFrame(self: *Renderer) !void {
        try self.sync_objects.waitForFence(self.current_frame);

        var image_index: u32 = undefined;
        const result = c.vkAcquireNextImageKHR(
            self.device,
            self.swapchain.handle,
            std.math.maxInt(u64),
            self.sync_objects.image_available_semaphores[self.current_frame],
            null,
            &image_index,
        );

        if (result == c.VK_ERROR_OUT_OF_DATE_KHR) {
            try self.recreateSwapChain();
            return;
        } else if (result != c.VK_SUCCESS and result != c.VK_SUBOPTIMAL_KHR) {
            return error.SwapChainImageAcquisitionFailed;
        }

        try self.sync_objects.resetFence(self.current_frame);

        // Reset and re-record command buffer
        try self.command_buffers[image_index].reset();
        try self.command_buffers[image_index].begin(.{});
        try self.command_buffers[image_index].beginRenderPass(
            self.pipeline.render_pass,
            self.swapchain.framebuffers[image_index],
            self.swapchain.extent,
            &[_]c.VkClearValue{.{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } }},
            c.VK_SUBPASS_CONTENTS_INLINE,
        );

        // Set dynamic state
        try self.command_buffers[image_index].setViewport(
            0,
            0,
            @floatFromInt(self.swapchain.extent.width),
            @floatFromInt(self.swapchain.extent.height),
            0,
            1,
        );
        try self.command_buffers[image_index].setScissor(
            0,
            0,
            self.swapchain.extent.width,
            self.swapchain.extent.height,
        );

        try self.command_buffers[image_index].bindPipeline(self.pipeline.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS);
        try self.command_buffers[image_index].draw(3, 1, 0, 0);
        try self.command_buffers[image_index].endRenderPass();
        try self.command_buffers[image_index].end();

        // Submit command buffer
        try command.submit(
            self.graphics_queue,
            &[_]*command.CommandBuffer{self.command_buffers[image_index]},
            .{
                .wait_semaphores = &[_]c.VkSemaphore{
                    self.sync_objects.image_available_semaphores[self.current_frame],
                },
                .wait_stages = &[_]c.VkPipelineStageFlags{
                    c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                },
                .signal_semaphores = &[_]c.VkSemaphore{
                    self.sync_objects.render_finished_semaphores[self.current_frame],
                },
                .fence = self.sync_objects.in_flight_fences[self.current_frame],
            },
        );

        const present_info = c.VkPresentInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &[_]c.VkSemaphore{
                self.sync_objects.render_finished_semaphores[self.current_frame],
            },
            .swapchainCount = 1,
            .pSwapchains = &self.swapchain.handle,
            .pImageIndices = &image_index,
            .pResults = null,
        };

        const present_result = c.vkQueuePresentKHR(self.present_queue, &present_info);
        if (present_result == c.VK_ERROR_OUT_OF_DATE_KHR or present_result == c.VK_SUBOPTIMAL_KHR or self.window.framebuffer_resized) {
            self.window.framebuffer_resized = false;
            try self.recreateSwapChain();
        } else if (present_result != c.VK_SUCCESS) {
            return error.SwapChainPresentFailed;
        }

        self.current_frame = (self.current_frame + 1) % sync.MAX_FRAMES_IN_FLIGHT;
    }

    fn recreateSwapChain(self: *Renderer) !void {
        const size = self.window.getFramebufferSize();
        while (size.width == 0 or size.height == 0) {
            const new_size = self.window.getFramebufferSize();
            if (new_size.width > 0 and new_size.height > 0) break;
            glfw.waitEvents();
        }

        _ = c.vkDeviceWaitIdle(self.device);

        // Cleanup old resources
        self.command_pool.*.deinit();
        self.pipeline.deinit();
        self.swapchain.deinit();

        // Get surface format for pipeline creation
        var swapchain_support = try device.querySwapChainSupport(self.physical_device, self.surface);
        defer swapchain_support.deinit();
        const surface_format = SwapChain.chooseSwapSurfaceFormat(swapchain_support.formats);

        // Create pipeline with surface format
        self.pipeline = try Pipeline.init(
            self.device,
            surface_format.format,
            @embedFile("../shaders/triangle.vert.spv"),
            @embedFile("../shaders/triangle.frag.spv"),
        );

        // Create swapchain with render pass
        self.swapchain = try SwapChain.init(
            self.physical_device,
            self.device,
            self.surface,
            self.window,
            self.allocator,
            self.pipeline.render_pass,
        );

        // Recreate command pool and buffers
        const queue_family_indices = try device.findQueueFamilies(self.physical_device, self.surface);
        self.command_pool = try command.CommandPool.init(
            self.device,
            queue_family_indices.graphics_family.?,
            .{
                .is_reusable = true,
                .allow_reset = true,
            },
            .{
                .initial_buffer_count = @intCast(self.swapchain.framebuffers.len),
                .max_free_buffers = 32,
                .thread_local = false,
                .batch_commands = true,
                .cache_state = true,
            },
            self.allocator,
        );

        // Create new command buffers
        var command_buffers = std.ArrayList(*command.CommandBuffer).init(self.allocator);
        errdefer command_buffers.deinit();

        var i: usize = 0;
        while (i < self.swapchain.framebuffers.len) : (i += 1) {
            const buffer = try self.command_pool.getBuffer();
            try command_buffers.append(buffer);
        }

        // Free old command buffers slice
        self.allocator.free(self.command_buffers);

        // Store new command buffers and cleanup ArrayList
        self.command_buffers = try command_buffers.toOwnedSlice();
        command_buffers.deinit();
    }

    pub fn deinit(self: *Renderer) void {
        _ = c.vkDeviceWaitIdle(self.device);

        self.sync_objects.deinit();
        self.command_pool.*.deinit();
        self.allocator.free(self.command_buffers);
        self.pipeline.deinit();
        self.swapchain.deinit();
        c.vkDestroyDevice(self.device, null);
        c.vkDestroySurfaceKHR(self.instance, self.surface, null);
        c.vkDestroyInstance(self.instance, null);
    }
};
