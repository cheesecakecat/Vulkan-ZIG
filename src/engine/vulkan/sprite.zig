const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const math = @import("../core/math.zig");
const logger = @import("../core/logger.zig");
const resources = @import("resources.zig");

const INSTANCE_BUFFER_SIZE = 32 * 1024 * 1024;
const MAX_SPRITES_PER_BATCH = 500000;
const VERTICES_PER_QUAD = 4;

const CACHE_LINE_SIZE = 64;
const INSTANCE_BUFFER_ALIGNMENT = 256;

const BUFFER_GROWTH_FACTOR: f32 = 1.5;
const BUFFER_SHRINK_THRESHOLD: f32 = 0.3;

const MEMORY_POOL_CONFIG = struct {
    const STAGING_BLOCK_SIZE = 128 * 1024 * 1024;
    const MIN_ALLOCATION_SIZE = 1024 * 1024;
    const MAX_ALLOCATIONS = 1024;
};

const AtomicCounter = std.atomic.Value(u64);

pub const SpriteInstance = extern struct {
    transform: [4][4]f32 align(16),
    color: [4]f32 align(16),
    tex_rect: [4]f32 align(16),
    layer: u32 align(4),
    flags: u32 align(4),

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(SpriteInstance),
            .inputRate = c.VK_VERTEX_INPUT_RATE_INSTANCE,
        };
    }

    pub fn getAttributeDescriptions() [7]c.VkVertexInputAttributeDescription {
        return .{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "transform") + 0 * @sizeOf([4]f32),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "transform") + 1 * @sizeOf([4]f32),
            },
            .{
                .binding = 0,
                .location = 2,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "transform") + 2 * @sizeOf([4]f32),
            },
            .{
                .binding = 0,
                .location = 3,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "transform") + 3 * @sizeOf([4]f32),
            },

            .{
                .binding = 0,
                .location = 4,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "color"),
            },

            .{
                .binding = 0,
                .location = 5,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "tex_rect"),
            },

            .{
                .binding = 0,
                .location = 6,
                .format = c.VK_FORMAT_R32_UINT,
                .offset = @offsetOf(SpriteInstance, "layer"),
            },
        };
    }
};

const DrawBatch = struct {
    instance_count: u32,
    texture_id: u32,
    layer: u32,
    flags: u32,
};

pub const SpriteBatchStats = struct {
    draw_calls: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    sprites_drawn: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    buffer_resizes: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    batch_breaks: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    peak_memory_usage: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    gpu_time_ns: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    cpu_time_ns: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    cache_hits: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
    cache_misses: AtomicCounter align(CACHE_LINE_SIZE) = AtomicCounter.init(0),
};

pub const SpriteBatch = struct {
    instance_buffer: resources.Buffer,
    staging_buffer: resources.Buffer,

    draw_calls: std.ArrayList(DrawBatch),
    current_batch: ?DrawBatch,

    instance_data: std.ArrayList(SpriteInstance),

    stats: SpriteBatchStats,
    buffers_need_update: bool = false,

    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    queue_family: u32,
    allocator: std.mem.Allocator,
    max_sprites: u32,

    command_pool: c.VkCommandPool,
    transfer_queue: c.VkQueue,

    pub fn init(
        device: c.VkDevice,
        instance: c.VkInstance,
        physical_device: c.VkPhysicalDevice,
        queue_family: u32,
        max_sprites: u32,
        compute_shader: ?[]const u8,
        allocator: std.mem.Allocator,
    ) !*SpriteBatch {
        _ = instance;
        _ = compute_shader;

        const self = try allocator.create(SpriteBatch);
        errdefer allocator.destroy(self);

        self.device = device;
        self.physical_device = physical_device;
        self.queue_family = queue_family;
        self.allocator = allocator;
        self.max_sprites = max_sprites;
        self.current_batch = null;

        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = queue_family,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .pNext = null,
        };

        if (c.vkCreateCommandPool(device, &pool_info, null, &self.command_pool) != c.VK_SUCCESS) {
            return error.CommandPoolCreationFailed;
        }
        errdefer c.vkDestroyCommandPool(device, self.command_pool, null);

        c.vkGetDeviceQueue(device, queue_family, 0, &self.transfer_queue);

        self.staging_buffer = try resources.Buffer.init(
            device,
            physical_device,
            MEMORY_POOL_CONFIG.STAGING_BLOCK_SIZE,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer self.staging_buffer.deinit();

        self.instance_buffer = try resources.Buffer.init(
            device,
            physical_device,
            INSTANCE_BUFFER_SIZE,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
        errdefer self.instance_buffer.deinit();

        self.draw_calls = try std.ArrayList(DrawBatch).initCapacity(allocator, 64);
        self.instance_data = try std.ArrayList(SpriteInstance).initCapacity(allocator, max_sprites);

        self.stats = .{};

        logger.info("sprite: initialized batch renderer (max sprites: {d}, queue family: {d})", .{ max_sprites, queue_family });
        return self;
    }

    pub fn deinit(self: *SpriteBatch) void {
        _ = c.vkDeviceWaitIdle(self.device);

        self.instance_buffer.deinit();
        self.staging_buffer.deinit();

        c.vkDestroyCommandPool(self.device, self.command_pool, null);

        self.draw_calls.deinit();
        self.instance_data.deinit();

        self.allocator.destroy(self);
    }

    pub fn begin(self: *SpriteBatch) void {
        self.draw_calls.clearRetainingCapacity();
        if (self.buffers_need_update) {
            self.instance_data.clearRetainingCapacity();
        }
        self.current_batch = null;

        const start_time = std.time.nanoTimestamp();
        _ = self.stats.cpu_time_ns.store(@as(u64, @intCast(@max(0, start_time))), .release);
    }

    fn transferToDeviceLocal(
        self: *SpriteBatch,
        staging_buffer: *const resources.Buffer,
        device_buffer: *resources.Buffer,
        size: usize,
    ) !void {
        var command_buffer: c.VkCommandBuffer = undefined;
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = self.command_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
            .pNext = null,
        };

        if (c.vkAllocateCommandBuffers(self.device, &alloc_info, &command_buffer) != c.VK_SUCCESS) {
            return error.CommandBufferAllocationFailed;
        }
        defer c.vkFreeCommandBuffers(self.device, self.command_pool, 1, &command_buffer);

        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
            .pNext = null,
        };

        if (c.vkBeginCommandBuffer(command_buffer, &begin_info) != c.VK_SUCCESS) {
            return error.CommandBufferBeginFailed;
        }

        const copy_region = c.VkBufferCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size,
        };

        c.vkCmdCopyBuffer(
            command_buffer,
            staging_buffer.handle,
            device_buffer.handle,
            1,
            &copy_region,
        );

        if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
            return error.CommandBufferEndFailed;
        }

        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
            .pNext = null,
        };

        if (c.vkQueueSubmit(self.transfer_queue, 1, &submit_info, null) != c.VK_SUCCESS) {
            return error.QueueSubmitFailed;
        }

        if (c.vkQueueWaitIdle(self.transfer_queue) != c.VK_SUCCESS) {
            return error.QueueWaitFailed;
        }
    }

    fn resizeBuffer(
        self: *SpriteBatch,
        buffer: *resources.Buffer,
        new_size: usize,
        usage: c.VkBufferUsageFlags,
        memory_properties: c.VkMemoryPropertyFlags,
    ) !void {
        const aligned_size = std.mem.alignForward(usize, new_size, INSTANCE_BUFFER_ALIGNMENT);

        if (aligned_size == buffer.size) return;

        if (aligned_size < buffer.size) {
            const current_usage = @as(f32, @floatFromInt(new_size)) / @as(f32, @floatFromInt(buffer.size));
            if (current_usage > BUFFER_SHRINK_THRESHOLD) return;
        }

        var old_buffer = buffer.*;

        buffer.* = try resources.Buffer.init(
            self.device,
            self.physical_device,
            aligned_size,
            usage,
            memory_properties,
        );

        if (old_buffer.size > 0) {
            try self.transferToDeviceLocal(&old_buffer, buffer, @min(old_buffer.size, aligned_size));
        }

        old_buffer.deinit();
        _ = self.stats.buffer_resizes.fetchAdd(1, .monotonic);
    }

    pub fn end(self: *SpriteBatch) !void {
        const start_time = std.time.nanoTimestamp();
        _ = self.stats.cpu_time_ns.store(@as(u64, @intCast(@max(0, start_time))), .release);

        if (self.current_batch) |batch| {
            try self.draw_calls.append(batch);
            self.current_batch = null;
        }

        if (self.buffers_need_update) {
            if (self.instance_data.items.len > 0) {
                const instance_size = self.instance_data.items.len * @sizeOf(SpriteInstance);

                if (instance_size > self.instance_buffer.size) {
                    const new_size = @max(
                        @as(usize, @intFromFloat(@as(f32, @floatFromInt(instance_size)) * BUFFER_GROWTH_FACTOR)),
                        MEMORY_POOL_CONFIG.MIN_ALLOCATION_SIZE,
                    );
                    try self.resizeBuffer(
                        &self.instance_buffer,
                        new_size,
                        c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    );
                }

                try self.staging_buffer.map();
                defer self.staging_buffer.unmap();

                const instance_bytes = std.mem.sliceAsBytes(self.instance_data.items);
                try self.staging_buffer.copy(instance_bytes);
                try self.transferToDeviceLocal(&self.staging_buffer, &self.instance_buffer, instance_size);
            }

            self.buffers_need_update = false;
        }

        _ = self.stats.draw_calls.fetchAdd(self.draw_calls.items.len, .monotonic);
        _ = self.stats.sprites_drawn.fetchAdd(self.instance_data.items.len, .monotonic);
        const total_memory = self.instance_buffer.size;
        _ = self.stats.peak_memory_usage.fetchMax(total_memory, .monotonic);

        const end_time = std.time.nanoTimestamp();
        const elapsed = @as(u64, @intCast(@max(0, end_time - start_time)));
        _ = self.stats.cpu_time_ns.store(elapsed, .release);

        if (self.instance_data.items.len > 0 and
            self.instance_data.items.len % CACHE_LINE_SIZE == 0)
        {
            _ = self.stats.cache_hits.fetchAdd(1, .monotonic);
        }
    }

    pub fn drawSprite(
        self: *SpriteBatch,
        position: math.Vec2,
        size: math.Vec2,
        rotation: f32,
        color: [4]f32,
        texture_id: u32,
        layer: u32,
        flags: u32,
    ) !void {
        if (self.instance_data.items.len >= self.max_sprites) {
            return error.MaxSpritesExceeded;
        }

        self.buffers_need_update = true;

        const need_new_batch = if (self.current_batch) |batch|
            batch.texture_id != texture_id or
                batch.layer != layer or
                batch.instance_count >= MAX_SPRITES_PER_BATCH
        else
            true;

        if (need_new_batch) {
            if (self.current_batch) |batch| {
                try self.draw_calls.append(batch);
                _ = self.stats.batch_breaks.fetchAdd(1, .monotonic);
            }
            self.current_batch = .{
                .instance_count = 1,
                .texture_id = texture_id,
                .layer = layer,
                .flags = flags,
            };
        } else if (self.current_batch) |*batch| {
            batch.instance_count += 1;
        }

        var transform = math.Mat4.identity();
        transform = transform.mul(math.Mat4.scale(size.x() * 2.0, size.y() * 2.0, 1));
        transform = transform.mul(math.Mat4.rotate(rotation, 0, 0, 1));
        transform = transform.mul(math.Mat4.translate(position.x(), position.y(), 0));

        try self.instance_data.append(.{
            .transform = transform.toArray2D(),
            .color = color,
            .tex_rect = .{ 0, 0, 1, 1 },
            .layer = layer,
            .flags = flags,
        });

        if (self.instance_data.items.len > 0 and
            self.instance_data.items.len % CACHE_LINE_SIZE == 0)
        {
            _ = self.stats.cache_hits.fetchAdd(1, .monotonic);
        }
    }

    pub fn getStats(self: *const SpriteBatch) SpriteBatchStats {
        return self.stats;
    }

    pub fn resetStats(self: *SpriteBatch) void {
        self.stats = .{};
    }
};
