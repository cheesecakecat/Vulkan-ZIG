const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");
const assert = std.debug.assert;
const builtin = @import("builtin");

pub const CommandSystem = struct {
    pub const Metrics = struct {
        command_count: std.atomic.Value(u64),
        submit_count: std.atomic.Value(u64),
        total_time_ns: std.atomic.Value(u64),
        peak_memory_bytes: std.atomic.Value(u64),
        cache_hits: std.atomic.Value(u64),
        cache_misses: std.atomic.Value(u64),

        pub fn init() Metrics {
            return .{
                .command_count = std.atomic.Value(u64).init(0),
                .submit_count = std.atomic.Value(u64).init(0),
                .total_time_ns = std.atomic.Value(u64).init(0),
                .peak_memory_bytes = std.atomic.Value(u64).init(0),
                .cache_hits = std.atomic.Value(u64).init(0),
                .cache_misses = std.atomic.Value(u64).init(0),
            };
        }
    };

    const LocalBufferCache = struct {
        buffers: std.ArrayList(CommandBuffer),
        free_list: std.ArrayList(usize),
        mutex: std.Thread.Mutex,
        metrics: Metrics,
        memory_pool: ?[]u8,
        pool_offset: usize,

        const POOL_SIZE = 1024 * 1024;

        pub fn init(allocator: std.mem.Allocator) !LocalBufferCache {
            const memory_pool = try allocator.alloc(u8, POOL_SIZE);

            return .{
                .buffers = std.ArrayList(CommandBuffer).init(allocator),
                .free_list = std.ArrayList(usize).init(allocator),
                .mutex = std.Thread.Mutex{},
                .metrics = Metrics.init(),
                .memory_pool = memory_pool,
                .pool_offset = 0,
            };
        }

        pub fn deinit(self: *LocalBufferCache, allocator: std.mem.Allocator) void {
            for (self.buffers.items) |*buffer| {
                if (buffer.handle != null) {
                    _ = buffer.reset() catch {};
                    c.vkFreeCommandBuffers(buffer.pool.device, buffer.pool.handle, 1, &buffer.handle);
                    buffer.handle = null;
                }
            }

            self.buffers.deinit();
            self.free_list.deinit();
            if (self.memory_pool) |pool| {
                allocator.free(pool);
                self.memory_pool = null;
            }
        }

        pub fn allocateFromPool(self: *LocalBufferCache, size: usize) ?[]u8 {
            if (self.pool_offset + size > POOL_SIZE) {
                self.resetPool();
            }

            if (self.pool_offset + size > POOL_SIZE) {
                return null;
            }

            const result = self.memory_pool[self.pool_offset .. self.pool_offset + size];
            self.pool_offset += size;
            return result;
        }

        pub fn resetPool(self: *LocalBufferCache) void {
            self.pool_offset = 0;
        }
    };

    pub const CommandError = error{
        CommandPoolCreationFailed,
        CommandPoolResetFailed,
        CommandBufferAllocationFailed,
        CommandBufferBeginFailed,
        CommandBufferEndFailed,
        CommandBufferResetFailed,
        CommandBufferSubmitFailed,
        OutOfMemory,
        InvalidHandle,
        ThreadingError,
        ValidationError,
        PoolExhausted,
        InvalidState,
    };

    pub const CommandPoolConfig = struct {
        flags: c.VkCommandPoolCreateFlags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_index: u32,
        initial_buffer_capacity: u32 = 16,
        thread_safe: bool = true,
        enable_validation: bool = if (builtin.mode == .Debug) true else false,
        enable_metrics: bool = true,
        buffer_reuse_threshold: u32 = 1000,
        max_buffers_per_thread: u32 = 256,
        memory_pool_size: usize = 1024 * 1024,
    };

    pub const CommandPool = struct {
        handle: c.VkCommandPool,
        device: c.VkDevice,
        instance: c.VkInstance,
        allocator: std.mem.Allocator,
        config: CommandPoolConfig,
        buffer_cache: std.AutoHashMap(std.Thread.Id, LocalBufferCache),
        mutex: std.Thread.Mutex,
        metrics: Metrics,
        last_error: ?[]const u8,
        validation_layer: ?struct {
            enabled: bool = true,
            debug_callback: ?c.VkDebugUtilsMessengerEXT = null,
            destroy_fn: ?*const fn (
                instance: c.VkInstance,
                messenger: c.VkDebugUtilsMessengerEXT,
                pAllocator: ?*const c.VkAllocationCallbacks,
            ) callconv(.C) void = null,
        } = if (builtin.mode == .Debug) .{} else null,

        const PFN_vkDestroyDebugUtilsMessengerEXT = fn (instance: c.VkInstance, messenger: c.VkDebugUtilsMessengerEXT, pAllocator: ?*const c.VkAllocationCallbacks) callconv(.C) void;

        pub fn init(device: c.VkDevice, instance: c.VkInstance, allocator: std.mem.Allocator, config: CommandPoolConfig) CommandError!CommandPool {
            assert(device != null);
            assert(instance != null);

            const pool_info = c.VkCommandPoolCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = @as(u32, config.flags) |
                    @as(u32, if (builtin.mode == .Debug) c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT else 0),
                .queueFamilyIndex = config.queue_family_index,
                .pNext = null,
            };

            var pool: c.VkCommandPool = undefined;
            const result = c.vkCreateCommandPool(device, &pool_info, null, &pool);

            switch (result) {
                c.VK_ERROR_OUT_OF_HOST_MEMORY, c.VK_ERROR_OUT_OF_DEVICE_MEMORY => return CommandError.OutOfMemory,
                c.VK_SUCCESS => {},
                else => return CommandError.CommandPoolCreationFailed,
            }

            var self = CommandPool{
                .handle = pool,
                .device = device,
                .instance = instance,
                .allocator = allocator,
                .config = config,
                .buffer_cache = std.AutoHashMap(std.Thread.Id, LocalBufferCache).init(allocator),
                .mutex = std.Thread.Mutex{},
                .metrics = Metrics.init(),
                .last_error = null,
                .validation_layer = if (builtin.mode == .Debug) .{
                    .enabled = true,
                    .debug_callback = null,
                    .destroy_fn = null,
                } else null,
            };

            if (self.validation_layer) |*v| {
                const destroy_fn = @as(
                    ?*const fn (
                        instance: c.VkInstance,
                        messenger: c.VkDebugUtilsMessengerEXT,
                        pAllocator: ?*const c.VkAllocationCallbacks,
                    ) callconv(.C) void,
                    @ptrCast(c.vkGetInstanceProcAddr(self.instance, "vkDestroyDebugUtilsMessengerEXT")),
                );
                v.destroy_fn = destroy_fn;
            }

            return self;
        }

        pub fn deinit(self: *CommandPool) void {
            if (self.config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();
            }

            if (self.device != null) {
                _ = c.vkDeviceWaitIdle(self.device);
            }

            var cache_iter = self.buffer_cache.iterator();
            while (cache_iter.next()) |entry| {
                for (entry.value_ptr.buffers.items) |*buffer| {
                    if (buffer.handle != null) {
                        _ = buffer.reset() catch {};
                    }
                }

                entry.value_ptr.deinit(self.allocator);
            }
            self.buffer_cache.deinit();

            if (self.handle != null and self.device != null) {
                if (self.validation_layer) |v| {
                    if (v.debug_callback) |callback| {
                        if (v.destroy_fn) |destroy_fn| {
                            destroy_fn(self.instance, callback, null);
                        }
                    }
                }
                c.vkDestroyCommandPool(self.device, self.handle, null);
                self.handle = null;
            }
        }

        pub fn acquireBuffer(self: *CommandPool, config: CommandBufferConfig) !CommandBuffer {
            if (self.config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();
            }

            const thread_id = std.Thread.getCurrentId();
            var cache_ptr: *LocalBufferCache = undefined;

            if (self.buffer_cache.getPtr(thread_id)) |existing| {
                cache_ptr = existing;
            } else {
                var new_cache = try LocalBufferCache.init(self.allocator);
                errdefer new_cache.deinit(self.allocator);

                try self.buffer_cache.put(thread_id, new_cache);
                cache_ptr = self.buffer_cache.getPtr(thread_id).?;
            }

            if (cache_ptr.buffers.items.len > (self.config.max_buffers_per_thread * 3) / 4) {
                var i: usize = 0;
                while (i < cache_ptr.buffers.items.len) {
                    if (!cache_ptr.buffers.items[i].is_in_use) {
                        _ = cache_ptr.buffers.swapRemove(i);
                    } else {
                        i += 1;
                    }
                }
            }

            if (cache_ptr.free_list.items.len > 0) {
                if (self.config.enable_metrics) {
                    _ = self.metrics.cache_hits.fetchAdd(1, .monotonic);
                }

                const index = cache_ptr.free_list.pop();
                cache_ptr.buffers.items[index].is_in_use = true;
                return cache_ptr.buffers.items[index];
            }

            if (cache_ptr.buffers.items.len >= self.config.max_buffers_per_thread) {
                return CommandError.PoolExhausted;
            }

            if (self.config.enable_metrics) {
                _ = self.metrics.cache_misses.fetchAdd(1, .monotonic);
            }

            var new_buffer = try CommandBuffer.init(self, config);
            errdefer new_buffer.deinit();

            try cache_ptr.buffers.append(new_buffer);

            const current_mem = @sizeOf(CommandBuffer) * cache_ptr.buffers.items.len;
            if (self.config.enable_metrics) {
                _ = self.metrics.peak_memory_bytes.fetchMax(current_mem, .monotonic);
            }

            return cache_ptr.buffers.items[cache_ptr.buffers.items.len - 1];
        }

        pub fn releaseBuffer(self: *CommandPool, buffer: *CommandBuffer) void {
            if (self.config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();
            }

            if (self.config.enable_metrics) {
                _ = self.metrics.submit_count.fetchAdd(1, .monotonic);
            }

            const thread_id = std.Thread.getCurrentId();
            if (self.buffer_cache.getPtr(thread_id)) |cache| {
                for (cache.buffers.items, 0..) |*buf, i| {
                    if (buf == buffer) {
                        _ = buffer.reset() catch {};
                        buf.is_in_use = false;
                        cache.free_list.append(i) catch {};
                        break;
                    }
                }
            }
        }

        pub fn getMetrics(self: *const CommandPool) struct {
            commands: u64,
            submits: u64,
            total_time_ms: u64,
            peak_memory_mb: u64,
            cache_hit_ratio: f32,
        } {
            const hits = self.metrics.cache_hits.load(.monotonic);
            const misses = self.metrics.cache_misses.load(.monotonic);
            const total = hits + misses;

            return .{
                .commands = self.metrics.command_count.load(.monotonic),
                .submits = self.metrics.submit_count.load(.monotonic),
                .total_time_ms = self.metrics.total_time_ns.load(.monotonic) / 1_000_000,
                .peak_memory_mb = self.metrics.peak_memory_bytes.load(.monotonic) / (1024 * 1024),
                .cache_hit_ratio = if (total > 0) @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(total)) else 0,
            };
        }

        pub fn reset(self: *CommandPool) CommandError!void {
            if (self.config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();
            }

            const result = c.vkResetCommandPool(self.device, self.handle, 0);
            if (result != c.VK_SUCCESS) {
                return CommandError.CommandPoolResetFailed;
            }

            var cache_iter = self.buffer_cache.iterator();
            while (cache_iter.next()) |entry| {
                entry.value_ptr.resetPool();
                entry.value_ptr.free_list.clearRetainingCapacity();
                for (entry.value_ptr.buffers.items) |*buffer| {
                    buffer.is_in_use = false;
                }
            }
        }
    };

    pub const CommandBufferConfig = struct {
        level: c.VkCommandBufferLevel = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        usage: c.VkCommandBufferUsageFlags = 0,
        enable_validation: bool = true,
    };

    pub const CommandBuffer = struct {
        handle: c.VkCommandBuffer,
        pool: *CommandPool,
        config: CommandBufferConfig,
        is_recording: bool = false,
        is_in_use: bool = false,
        last_error: ?[]const u8 = null,
        validation: ?struct {
            command_count: u32 = 0,
            last_command: []const u8 = "",
            start_time: i64 = 0,
        } = null,

        pub fn init(pool: *CommandPool, config: CommandBufferConfig) CommandError!CommandBuffer {
            assert(pool.handle != null);

            const alloc_info = c.VkCommandBufferAllocateInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = pool.handle,
                .level = config.level,
                .commandBufferCount = 1,
                .pNext = null,
            };

            var buffer: c.VkCommandBuffer = undefined;
            const result = c.vkAllocateCommandBuffers(pool.device, &alloc_info, &buffer);

            switch (result) {
                c.VK_ERROR_OUT_OF_HOST_MEMORY, c.VK_ERROR_OUT_OF_DEVICE_MEMORY => return CommandError.OutOfMemory,
                c.VK_SUCCESS => {},
                else => return CommandError.CommandBufferAllocationFailed,
            }

            return CommandBuffer{
                .handle = buffer,
                .pool = pool,
                .config = config,
                .validation = if (config.enable_validation) .{} else null,
            };
        }

        pub fn deinit(self: *CommandBuffer) void {
            if (self.handle != null and self.pool.handle != null) {
                c.vkFreeCommandBuffers(self.pool.device, self.pool.handle, 1, &self.handle);
                self.handle = null;
            }
        }

        pub fn begin(self: *CommandBuffer, usage: c.VkCommandBufferUsageFlags) CommandError!void {
            if (self.handle == null) return CommandError.InvalidHandle;
            if (self.is_recording) return;

            const begin_info = c.VkCommandBufferBeginInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = usage,
                .pInheritanceInfo = null,
                .pNext = null,
            };

            const result = c.vkBeginCommandBuffer(self.handle, &begin_info);
            if (result != c.VK_SUCCESS) {
                return CommandError.CommandBufferBeginFailed;
            }

            self.is_recording = true;
            if (self.validation) |*v| {
                v.command_count = 0;
                v.start_time = std.time.milliTimestamp();
            }
        }

        pub fn end(self: *CommandBuffer) CommandError!void {
            if (self.handle == null) return CommandError.InvalidHandle;
            if (!self.is_recording) return;

            const result = c.vkEndCommandBuffer(self.handle);
            if (result != c.VK_SUCCESS) {
                return CommandError.CommandBufferEndFailed;
            }

            self.is_recording = false;
            if (self.validation) |v| {
                if (v.command_count > 0) {
                    const duration = std.time.milliTimestamp() - v.start_time;
                    logger.debug("vulkan: command buffer completed: {} commands in {}ms", .{ v.command_count, duration });
                }
            }
        }

        pub fn reset(self: *CommandBuffer) CommandError!void {
            if (self.handle == null) return CommandError.InvalidHandle;

            const result = c.vkResetCommandBuffer(self.handle, 0);
            if (result != c.VK_SUCCESS) {
                return CommandError.CommandBufferResetFailed;
            }

            self.is_recording = false;
            if (self.validation) |*v| {
                v.command_count = 0;
            }
        }

        pub fn submit(
            self: *CommandBuffer,
            queue: c.VkQueue,
            wait_semaphores: []const c.VkSemaphore,
            wait_stages: []const c.VkPipelineStageFlags,
            signal_semaphores: []const c.VkSemaphore,
            fence: ?c.VkFence,
        ) CommandError!void {
            if (self.handle == null or queue == null) return CommandError.InvalidHandle;
            if (self.is_recording) return CommandError.CommandBufferEndFailed;

            if (wait_semaphores.len != wait_stages.len) {
                logger.err("vulkan: mismatched wait semaphores ({}) and stages ({}) lengths", .{ wait_semaphores.len, wait_stages.len });
                return CommandError.ValidationError;
            }

            const submit_info = c.VkSubmitInfo{
                .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = @intCast(wait_semaphores.len),
                .pWaitSemaphores = wait_semaphores.ptr,
                .pWaitDstStageMask = wait_stages.ptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &self.handle,
                .signalSemaphoreCount = @intCast(signal_semaphores.len),
                .pSignalSemaphores = signal_semaphores.ptr,
                .pNext = null,
            };

            const result = c.vkQueueSubmit(queue, 1, &submit_info, if (fence) |f| f else null);
            if (result != c.VK_SUCCESS) {
                self.last_error = "Queue submission failed";
                return CommandError.CommandBufferSubmitFailed;
            }

            if (self.validation) |*v| {
                logger.debug("vulkan: submitted command buffer with {} commands", .{v.command_count});
            }
        }
    };

    pub const CommandBufferBuilder = struct {
        buffer: *CommandBuffer,
        command_count: u32 = 0,

        pub fn init(buffer: *CommandBuffer) CommandError!CommandBufferBuilder {
            try buffer.begin(0);
            return CommandBufferBuilder{
                .buffer = buffer,
            };
        }

        pub fn end(self: *CommandBufferBuilder) CommandError!void {
            if (self.buffer.validation) |*v| {
                v.command_count = self.command_count;
            }
            try self.buffer.end();
        }

        fn recordCommand(self: *CommandBufferBuilder, comptime command_name: []const u8) void {
            self.command_count += 1;
            if (self.buffer.validation) |*v| {
                v.last_command = command_name;
            }
        }

        pub inline fn bindPipeline(self: *CommandBufferBuilder, pipeline: c.VkPipeline, bind_point: c.VkPipelineBindPoint) void {
            assert(pipeline != null);
            c.vkCmdBindPipeline(self.buffer.handle, bind_point, pipeline);
            self.recordCommand("bindPipeline");
        }

        pub inline fn setViewport(self: *CommandBufferBuilder, viewport: c.VkViewport) void {
            c.vkCmdSetViewport(self.buffer.handle, 0, 1, &viewport);
            self.recordCommand("setViewport");
        }

        pub inline fn setScissor(self: *CommandBufferBuilder, scissor: c.VkRect2D) void {
            c.vkCmdSetScissor(self.buffer.handle, 0, 1, &scissor);
            self.recordCommand("setScissor");
        }

        pub inline fn bindVertexBuffers(
            self: *CommandBufferBuilder,
            first_binding: u32,
            buffers: []const c.VkBuffer,
            offsets: []const c.VkDeviceSize,
        ) void {
            assert(buffers.len > 0);
            assert(buffers.len == offsets.len);

            c.vkCmdBindVertexBuffers(
                self.buffer.handle,
                first_binding,
                @intCast(buffers.len),
                buffers.ptr,
                offsets.ptr,
            );
            self.recordCommand("bindVertexBuffers");
        }

        pub inline fn bindIndexBuffer(
            self: *CommandBufferBuilder,
            buffer: c.VkBuffer,
            offset: c.VkDeviceSize,
            index_type: c.VkIndexType,
        ) void {
            assert(buffer != null);
            c.vkCmdBindIndexBuffer(self.buffer.handle, buffer, offset, index_type);
            self.recordCommand("bindIndexBuffer");
        }

        pub inline fn drawIndexed(
            self: *CommandBufferBuilder,
            comptime index_count: u32,
            comptime instance_count: u32,
            first_index: u32,
            vertex_offset: i32,
            first_instance: u32,
        ) void {
            assert(index_count > 0);
            c.vkCmdDrawIndexed(
                self.buffer.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
            self.recordCommand("drawIndexed");
        }

        pub inline fn beginRenderPass(
            self: *CommandBufferBuilder,
            render_pass: c.VkRenderPass,
            framebuffer: c.VkFramebuffer,
            render_area: c.VkRect2D,
            clear_values: []const c.VkClearValue,
        ) void {
            assert(render_pass != null);
            assert(framebuffer != null);

            const render_pass_info = c.VkRenderPassBeginInfo{
                .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = render_pass,
                .framebuffer = framebuffer,
                .renderArea = render_area,
                .clearValueCount = @intCast(clear_values.len),
                .pClearValues = clear_values.ptr,
                .pNext = null,
            };

            c.vkCmdBeginRenderPass(
                self.buffer.handle,
                &render_pass_info,
                c.VK_SUBPASS_CONTENTS_INLINE,
            );
            self.recordCommand("beginRenderPass");
        }

        pub inline fn endRenderPass(self: *CommandBufferBuilder) void {
            c.vkCmdEndRenderPass(self.buffer.handle);
            self.recordCommand("endRenderPass");
        }

        pub inline fn pushConstants(
            self: *CommandBufferBuilder,
            layout: c.VkPipelineLayout,
            stage_flags: c.VkShaderStageFlags,
            offset: u32,
            size: u32,
            values: *const anyopaque,
        ) void {
            assert(layout != null);
            assert(values != null);
            assert(size > 0);

            c.vkCmdPushConstants(
                self.buffer.handle,
                layout,
                stage_flags,
                offset,
                size,
                values,
            );
            self.recordCommand("pushConstants");
        }
    };
};
