const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");

const MIN_ALLOCATION_SIZE: u64 = 1024 * 1024;
const MIN_BUFFER_ALIGNMENT: u64 = 256;

var total_allocated_memory: u64 = 0;
var buffer_count: u32 = 0;
var image_count: u32 = 0;

fn findMemoryType(
    physical_device: c.VkPhysicalDevice,
    type_filter: u32,
    properties: c.VkMemoryPropertyFlags,
) !u32 {
    var memory_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    for (0..memory_properties.memoryTypeCount) |i| {
        const type_matches = (type_filter & (@as(u32, 1) << @intCast(i))) != 0;
        const property_matches = (memory_properties.memoryTypes[i].propertyFlags & properties) == properties;

        if (type_matches and property_matches) {
            return @intCast(i);
        }
    }

    logger.err("resources: failed to find suitable memory type\nRequired properties: {x}", .{properties});
    return error.NoSuitableMemoryType;
}

const Allocation = struct {
    offset: u64,
    size: u64,
    alignment: u64,
    is_free: bool,
    next: ?*Allocation,
    prev: ?*Allocation,
};

const MemoryBlock = struct {
    memory: c.VkDeviceMemory,
    size: u64,
    used: u64,
    mapped_data: ?[*]u8,
    allocations: std.ArrayList(Allocation),
    mutex: std.Thread.Mutex,
    fragmentation: f32,
    physical_device: c.VkPhysicalDevice,

    const MIN_ALIGNMENT: u64 = 256;
    const DEFRAG_THRESHOLD: f32 = 0.3;

    fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        size: u64,
        memory_type_index: u32,
        properties: c.VkMemoryPropertyFlags,
        allocator: std.mem.Allocator,
    ) !MemoryBlock {
        const aligned_size = std.mem.alignForward(u64, size, MIN_ALLOCATION_SIZE);

        var memory: c.VkDeviceMemory = undefined;
        const alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = aligned_size,
            .memoryTypeIndex = memory_type_index,
            .pNext = null,
        };

        if (c.vkAllocateMemory(device, &alloc_info, null, &memory) != c.VK_SUCCESS) {
            return error.MemoryAllocationFailed;
        }

        var mapped_data: ?[*]u8 = null;
        if (properties & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) {
            var data: ?*anyopaque = null;
            if (c.vkMapMemory(device, memory, 0, aligned_size, 0, &data) != c.VK_SUCCESS) {
                c.vkFreeMemory(device, memory, null);
                return error.MemoryMapFailed;
            }
            mapped_data = @ptrCast(data);
        }

        logger.info("resources: created memory block ({} KB, {s})", .{
            aligned_size / 1024,
            if ((properties & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) "host-visible" else "device-local",
        });

        return MemoryBlock{
            .memory = memory,
            .size = aligned_size,
            .used = 0,
            .mapped_data = mapped_data,
            .allocations = std.ArrayList(Allocation).init(allocator),
            .mutex = std.Thread.Mutex{},
            .fragmentation = 0,
            .physical_device = physical_device,
        };
    }

    fn deinit(self: *MemoryBlock, device: c.VkDevice) void {
        if (self.mapped_data != null) {
            c.vkUnmapMemory(device, self.memory);
        }
        c.vkFreeMemory(device, self.memory, null);
        self.allocations.deinit();
    }

    fn allocate(self: *MemoryBlock, size: u64, alignment: u64) ?struct { offset: u64, size: u64 } {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.allocations.items) |*alloc| {
            if (alloc.is_free and alloc.size >= size) {
                const aligned_offset = std.mem.alignForward(u64, alloc.offset, alignment);
                if (aligned_offset + size <= alloc.offset + alloc.size) {
                    alloc.is_free = false;
                    self.updateFragmentation();
                    return .{ .offset = aligned_offset, .size = size };
                }
            }
        }

        const aligned_offset = std.mem.alignForward(u64, self.used, if (alignment < MIN_ALIGNMENT) MIN_ALIGNMENT else alignment);
        if (aligned_offset + size > self.size) {
            return null;
        }

        self.allocations.append(.{
            .offset = aligned_offset,
            .size = size,
            .alignment = alignment,
            .is_free = false,
            .next = null,
            .prev = if (self.allocations.items.len > 0) &self.allocations.items[self.allocations.items.len - 1] else null,
        }) catch return null;

        self.used = aligned_offset + size;
        self.updateFragmentation();
        return .{ .offset = aligned_offset, .size = size };
    }

    fn free(self: *MemoryBlock, offset: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.allocations.items) |*alloc| {
            if (alloc.offset == offset) {
                alloc.is_free = true;
                self.updateFragmentation();
                self.tryCoalesce(alloc);
                break;
            }
        }
    }

    fn updateFragmentation(self: *MemoryBlock) void {
        var free_space: u64 = 0;
        var largest_free_block: u64 = 0;

        for (self.allocations.items) |alloc| {
            if (alloc.is_free) {
                free_space += alloc.size;
                largest_free_block = @max(largest_free_block, alloc.size);
            }
        }

        if (free_space > 0) {
            const largest_f: f32 = @floatFromInt(largest_free_block);
            const free_f: f32 = @floatFromInt(free_space);
            self.fragmentation = 1.0 - (largest_f / free_f);
        } else {
            self.fragmentation = 0;
        }
    }

    fn tryCoalesce(self: *MemoryBlock, alloc: *Allocation) void {
        _ = self;

        if (alloc.next) |next| {
            if (next.is_free) {
                alloc.size += next.size;
                alloc.next = next.next;
                if (next.next) |n| {
                    n.prev = alloc;
                }
            }
        }

        if (alloc.prev) |prev| {
            if (prev.is_free) {
                prev.size += alloc.size;
                prev.next = alloc.next;
                if (alloc.next) |n| {
                    n.prev = prev;
                }
            }
        }
    }

    fn defragment(self: *MemoryBlock, device: c.VkDevice, cmd: c.VkCommandBuffer) !void {
        if (self.fragmentation < DEFRAG_THRESHOLD) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        var staging_buffer = try Buffer.init(
            device,
            self.physical_device,
            self.size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        defer staging_buffer.deinit();

        var new_offset: u64 = 0;
        for (self.allocations.items) |*alloc| {
            if (!alloc.is_free) {
                const src_offset = alloc.offset;
                const aligned_new_offset = std.mem.alignForward(u64, new_offset, alloc.alignment);

                const copy_region = c.VkBufferCopy{
                    .srcOffset = src_offset,
                    .dstOffset = aligned_new_offset,
                    .size = alloc.size,
                };

                c.vkCmdCopyBuffer(
                    cmd,
                    self.memory,
                    staging_buffer.handle,
                    1,
                    &copy_region,
                );

                alloc.offset = aligned_new_offset;
                new_offset = aligned_new_offset + alloc.size;
            }
        }

        const copy_region = c.VkBufferCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = new_offset,
        };

        c.vkCmdCopyBuffer(
            cmd,
            staging_buffer.handle,
            self.memory,
            1,
            &copy_region,
        );

        self.used = new_offset;
        self.updateFragmentation();
    }
};

pub const MemoryPool = struct {
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    blocks: std.ArrayList(MemoryBlock),
    properties: c.VkMemoryPropertyFlags,
    memory_type_index: u32,
    allocator: std.mem.Allocator,
    total_size: u64,
    used_size: u64,
    peak_size: u64,
    worker_threads: std.ArrayList(std.Thread),
    work_queue: std.ArrayList(WorkItem),
    queue_mutex: std.Thread.Mutex,
    queue_condition: std.Thread.Condition,
    defrag_thread: ?std.Thread,
    should_stop: std.atomic.Atomic(bool),

    const INITIAL_BLOCK_SIZE: u64 = 64 * 1024 * 1024;
    const AGGRESSIVE_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
    const MAX_BLOCKS: u32 = 16;
    const WORKER_COUNT: u32 = 4;
    const DEFRAG_INTERVAL: u64 = 60 * std.time.ns_per_s;

    const WorkItem = struct {
        operation: enum {
            Allocate,
            Free,
            Defragment,
        },
        size: u64,
        alignment: u64,
        block_index: usize,
        offset: u64,
        result: ?struct {
            memory: c.VkDeviceMemory,
            offset: u64,
        },
        completed: std.atomic.Atomic(bool),
    };

    pub fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        properties: c.VkMemoryPropertyFlags,
        memory_type_index: u32,
        initial_size: u64,
        alloc: std.mem.Allocator,
    ) !*MemoryPool {
        const size = if (initial_size == 0) INITIAL_BLOCK_SIZE else initial_size;

        logger.info("resources: creating memory pool ({d} MB, {s}, {d} workers)", .{
            size / (1024 * 1024),
            if (properties & c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0)
                "device-local"
            else if (properties & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0)
                "host-visible"
            else
                "other",
            WORKER_COUNT,
        });

        const pool = try alloc.create(MemoryPool);
        errdefer alloc.destroy(pool);

        var blocks = std.ArrayList(MemoryBlock).init(alloc);
        errdefer blocks.deinit();

        var workers = std.ArrayList(std.Thread).init(alloc);
        errdefer workers.deinit();

        var work_queue = std.ArrayList(WorkItem).init(alloc);
        errdefer work_queue.deinit();

        var block = try MemoryBlock.init(
            device,
            physical_device,
            size,
            memory_type_index,
            properties,
            alloc,
        );
        errdefer block.deinit(device);

        try blocks.append(block);

        pool.* = .{
            .device = device,
            .physical_device = physical_device,
            .blocks = blocks,
            .properties = properties,
            .memory_type_index = memory_type_index,
            .allocator = alloc,
            .total_size = size,
            .used_size = 0,
            .peak_size = 0,
            .worker_threads = workers,
            .work_queue = work_queue,
            .queue_mutex = .{},
            .queue_condition = .{},
            .defrag_thread = null,
            .should_stop = std.atomic.Atomic(bool).init(false),
        };

        var i: u32 = 0;
        while (i < WORKER_COUNT) : (i += 1) {
            const thread = try std.Thread.spawn(.{}, workerThread, .{pool});
            try pool.worker_threads.append(thread);
        }

        if (properties & c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0) {
            pool.defrag_thread = try std.Thread.spawn(.{}, defragThread, .{pool});
            logger.info("resources: started defrag thread for device-local memory pool", .{});
        }

        return pool;
    }

    pub fn deinit(self: *MemoryPool) void {
        logger.info("resources: shutting down memory pool...", .{});

        self.should_stop.store(true, .release);
        self.queue_condition.broadcast();

        for (self.worker_threads.items) |thread| {
            thread.join();
        }

        if (self.defrag_thread) |thread| {
            thread.join();
        }

        for (self.blocks.items) |*block| {
            block.deinit(self.device);
        }

        self.blocks.deinit();
        self.worker_threads.deinit();
        self.work_queue.deinit();
        self.allocator.destroy(self);

        logger.info("resources: destroyed memory pool (peak usage: {d} MB)", .{
            self.peak_size / (1024 * 1024),
        });
    }

    fn workerThread(pool: *MemoryPool) void {
        while (!pool.should_stop.load(.acquire)) {
            pool.queue_mutex.lock();
            while (pool.work_queue.items.len == 0 and !pool.should_stop.load(.acquire)) {
                pool.queue_condition.wait(&pool.queue_mutex);
            }

            if (pool.should_stop.load(.acquire)) {
                pool.queue_mutex.unlock();
                break;
            }

            const work = pool.work_queue.orderedRemove(0);
            pool.queue_mutex.unlock();

            switch (work.operation) {
                .Allocate => {
                    if (work.block_index < pool.blocks.items.len) {
                        if (pool.blocks.items[work.block_index].allocate(work.size, work.alignment)) |allocation| {
                            work.result = .{
                                .memory = pool.blocks.items[work.block_index].memory,
                                .offset = allocation.offset,
                            };
                        }
                    }
                },
                .Free => {
                    if (work.block_index < pool.blocks.items.len) {
                        pool.blocks.items[work.block_index].free(work.offset);
                    }
                },
                .Defragment => {
                    if (work.block_index < pool.blocks.items.len) {
                        var cmd = pool.device.createCommandBuffer() catch {
                            logger.err("resources: failed to create command buffer for defrag", .{});
                            continue;
                        };
                        defer cmd.deinit();

                        pool.blocks.items[work.block_index].defragment(pool.device, cmd.handle) catch |err| {
                            logger.err("resources: defragmentation failed: {}", .{err});
                        };
                    }
                },
            }

            work.completed.store(true, .release);
        }
    }

    fn defragThread(pool: *MemoryPool) void {
        while (!pool.should_stop.load(.acquire)) {
            std.time.sleep(DEFRAG_INTERVAL);

            for (pool.blocks.items, 0..) |block, i| {
                if (block.fragmentation > MemoryBlock.DEFRAG_THRESHOLD) {
                    logger.info("resources: defragmenting block {d} ({d} % fragmented)", .{
                        i, @as(u32, @intFromFloat(block.fragmentation * 100)),
                    });

                    pool.queue_mutex.lock();
                    pool.work_queue.append(.{
                        .operation = .Defragment,
                        .size = 0,
                        .alignment = 0,
                        .block_index = i,
                        .offset = 0,
                        .result = null,
                        .completed = std.atomic.Atomic(bool).init(false),
                    }) catch {
                        logger.err("resources: failed to queue defrag operation", .{});
                        pool.queue_mutex.unlock();
                        continue;
                    };
                    pool.queue_condition.signal();
                    pool.queue_mutex.unlock();
                }
            }
        }
    }

    pub fn allocate(self: *MemoryPool, size: u64, alignment: u64) !struct { memory: c.VkDeviceMemory, offset: u64 } {
        var work_items = std.ArrayList(WorkItem).init(self.allocator);
        defer work_items.deinit();

        self.queue_mutex.lock();
        for (self.blocks.items, 0..) |_, i| {
            try work_items.append(.{
                .operation = .Allocate,
                .size = size,
                .alignment = alignment,
                .block_index = i,
                .offset = 0,
                .result = null,
                .completed = std.atomic.Atomic(bool).init(false),
            });
            try self.work_queue.append(work_items.items[work_items.items.len - 1]);
        }
        self.queue_condition.broadcast();
        self.queue_mutex.unlock();

        while (true) {
            var all_completed = true;
            for (work_items.items) |*item| {
                if (!item.completed.load(.acquire)) {
                    all_completed = false;
                    continue;
                }
                if (item.result) |result| {
                    self.used_size += size;
                    self.peak_size = @max(self.peak_size, self.used_size);

                    if (self.used_size > self.total_size * 90 / 100) {
                        logger.warn("resources: memory pool at {d} % capacity", .{
                            (self.used_size * 100) / self.total_size,
                        });
                    }

                    return result;
                }
            }
            if (all_completed) break;
            std.time.sleep(100);
        }

        if (self.blocks.items.len >= MAX_BLOCKS) {
            logger.err("resources: memory pool exhausted ({d} blocks)\nPossible solutions:\n1. Increase MAX_BLOCKS\n2. Increase block size\n3. Check for memory leaks", .{MAX_BLOCKS});
            return error.OutOfMemory;
        }

        const block_size = @max(size, AGGRESSIVE_BLOCK_SIZE);
        var new_block = try MemoryBlock.init(
            self.device,
            self.physical_device,
            block_size,
            self.memory_type_index,
            self.properties,
            self.allocator,
        );
        errdefer new_block.deinit(self.device);

        try self.blocks.append(new_block);
        self.total_size += block_size;

        logger.info("resources: added new memory block ({d} MB, total: {d} MB)", .{
            block_size / (1024 * 1024),
            self.total_size / (1024 * 1024),
        });

        const allocation = new_block.allocate(size, alignment) orelse return error.AllocationFailed;
        self.used_size += size;
        self.peak_size = @max(self.peak_size, self.used_size);

        return .{ .memory = new_block.memory, .offset = allocation.offset };
    }

    pub fn free(self: *MemoryPool, memory: c.VkDeviceMemory, offset: u64, size: u64) void {
        self.queue_mutex.lock();
        defer self.queue_mutex.unlock();

        for (self.blocks.items, 0..) |block, i| {
            if (block.memory == memory) {
                self.work_queue.append(.{
                    .operation = .Free,
                    .size = size,
                    .alignment = 0,
                    .block_index = i,
                    .offset = offset,
                    .result = null,
                    .completed = std.atomic.Atomic(bool).init(false),
                }) catch {
                    logger.err("resources: failed to queue free operation", .{});
                    return;
                };
                self.queue_condition.signal();
                self.used_size -= size;
                break;
            }
        }
    }

    pub fn getStats(self: *const MemoryPool) struct {
        total_mb: f32,
        used_mb: f32,
        peak_mb: f32,
        block_count: u32,
        fragmentation: f32,
    } {
        var total_fragmentation: f32 = 0;
        for (self.blocks.items) |block| {
            total_fragmentation += block.fragmentation;
        }
        const avg_fragmentation = if (self.blocks.items.len > 0)
            total_fragmentation / @as(f32, @floatFromInt(self.blocks.items.len))
        else
            0;

        return .{
            .total_mb = @as(f32, @floatFromInt(self.total_size)) / (1024 * 1024),
            .used_mb = @as(f32, @floatFromInt(self.used_size)) / (1024 * 1024),
            .peak_mb = @as(f32, @floatFromInt(self.peak_size)) / (1024 * 1024),
            .block_count = @intCast(self.blocks.items.len),
            .fragmentation = avg_fragmentation,
        };
    }
};

pub const Buffer = struct {
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
    size: u64,
    mapped_data: ?*anyopaque,
    device: c.VkDevice,

    pub fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        size: u64,
        usage: c.VkBufferUsageFlags,
        memory_properties: c.VkMemoryPropertyFlags,
    ) !Buffer {
        if (size == 0) {
            logger.err("resources: attempted to create buffer with size 0", .{});
            return error.InvalidBufferSize;
        }

        if (size > 512 * 1024 * 1024) {
            logger.warn("resources: creating large buffer ({d} MB)", .{size / (1024 * 1024)});
        }

        var buffer: c.VkBuffer = undefined;
        const buffer_info = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .flags = 0,
            .pNext = null,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        if (c.vkCreateBuffer(device, &buffer_info, null, &buffer) != c.VK_SUCCESS) {
            logger.err("resources: failed to create buffer", .{});
            return error.BufferCreationFailed;
        }
        errdefer c.vkDestroyBuffer(device, buffer, null);

        var mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

        const memory_type = try findMemoryType(
            physical_device,
            mem_requirements.memoryTypeBits,
            memory_properties,
        );

        var memory: c.VkDeviceMemory = undefined;
        const alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = memory_type,
            .pNext = null,
        };

        if (c.vkAllocateMemory(device, &alloc_info, null, &memory) != c.VK_SUCCESS) {
            logger.err("resources: failed to allocate buffer memory", .{});
            return error.MemoryAllocationFailed;
        }
        errdefer c.vkFreeMemory(device, memory, null);

        if (c.vkBindBufferMemory(device, buffer, memory, 0) != c.VK_SUCCESS) {
            logger.err("resources: failed to bind buffer memory", .{});
            return error.MemoryBindFailed;
        }

        buffer_count += 1;
        total_allocated_memory += mem_requirements.size;

        if (buffer_count % 100 == 0) {
            logger.warn("resources: high buffer count ({d}), consider implementing buffer pooling", .{buffer_count});
        }

        logger.info("resources: created buffer ({d} KB, {s})", .{
            mem_requirements.size / 1024,
            if (memory_properties & c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0)
                "device-local"
            else if (memory_properties & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0)
                "host-visible"
            else
                "other",
        });

        return Buffer{
            .handle = buffer,
            .memory = memory,
            .size = size,
            .mapped_data = null,
            .device = device,
        };
    }

    pub fn deinit(self: *Buffer) void {
        if (self.mapped_data != null) {
            c.vkUnmapMemory(self.device, self.memory);
        }
        c.vkDestroyBuffer(self.device, self.handle, null);
        c.vkFreeMemory(self.device, self.memory, null);

        buffer_count -= 1;
        total_allocated_memory -= self.size;
    }

    pub fn map(self: *Buffer) !void {
        if (self.mapped_data != null) {
            logger.warn("resources: attempted to map already mapped buffer", .{});
            return;
        }

        var data: ?*anyopaque = null;
        if (c.vkMapMemory(
            self.device,
            self.memory,
            0,
            self.size,
            0,
            &data,
        ) != c.VK_SUCCESS) {
            logger.err("resources: failed to map buffer memory", .{});
            return error.MemoryMapFailed;
        }
        self.mapped_data = data;
    }

    pub fn unmap(self: *Buffer) void {
        if (self.mapped_data != null) {
            c.vkUnmapMemory(self.device, self.memory);
            self.mapped_data = null;
        }
    }

    pub fn copy(self: *Buffer, data: []const u8) !void {
        if (self.mapped_data == null) {
            return error.BufferNotMapped;
        }
        if (data.len > self.size) {
            return error.BufferTooSmall;
        }
        const dest = @as([*]u8, @ptrCast(self.mapped_data.?))[0..data.len];
        @memcpy(dest, data);
    }
};

pub const Image = struct {
    handle: c.VkImage,
    memory: c.VkDeviceMemory,
    view: c.VkImageView,
    size: c.VkExtent3D,
    format: c.VkFormat,
    layout: c.VkImageLayout,
    device: c.VkDevice,
    pool: ?*MemoryPool,

    pub fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        size: c.VkExtent3D,
        format: c.VkFormat,
        usage: c.VkImageUsageFlags,
        properties: c.VkMemoryPropertyFlags,
    ) !Image {
        const image_info = c.VkImageCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = c.VK_IMAGE_TYPE_2D,
            .extent = size,
            .mipLevels = 1,
            .arrayLayers = 1,
            .format = format,
            .tiling = c.VK_IMAGE_TILING_OPTIMAL,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .flags = 0,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .pNext = null,
        };

        var image: c.VkImage = undefined;
        if (c.vkCreateImage(device, &image_info, null, &image) != c.VK_SUCCESS) {
            return error.ImageCreationFailed;
        }
        errdefer c.vkDestroyImage(device, image, null);

        var mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetImageMemoryRequirements(device, image, &mem_requirements);

        const memory_type_index = try findMemoryType(
            physical_device,
            mem_requirements.memoryTypeBits,
            properties,
        );

        const alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = memory_type_index,
            .pNext = null,
        };

        var memory: c.VkDeviceMemory = undefined;
        if (c.vkAllocateMemory(device, &alloc_info, null, &memory) != c.VK_SUCCESS) {
            return error.MemoryAllocationFailed;
        }
        errdefer c.vkFreeMemory(device, memory, null);

        if (c.vkBindImageMemory(device, image, memory, 0) != c.VK_SUCCESS) {
            return error.MemoryBindFailed;
        }

        const view_info = c.VkImageViewCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = .{
                .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .flags = 0,
            .pNext = null,
        };

        var view: c.VkImageView = undefined;
        if (c.vkCreateImageView(device, &view_info, null, &view) != c.VK_SUCCESS) {
            return error.ImageViewCreationFailed;
        }

        return Image{
            .handle = image,
            .memory = memory,
            .view = view,
            .size = size,
            .format = format,
            .layout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .device = device,
            .pool = null,
        };
    }

    pub fn initFromPool(
        device: c.VkDevice,
        size: c.VkExtent3D,
        format: c.VkFormat,
        usage: c.VkImageUsageFlags,
        pool: *MemoryPool,
    ) !Image {
        const image_info = c.VkImageCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = c.VK_IMAGE_TYPE_2D,
            .extent = size,
            .mipLevels = 1,
            .arrayLayers = 1,
            .format = format,
            .tiling = c.VK_IMAGE_TILING_OPTIMAL,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .flags = 0,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .pNext = null,
        };

        var image: c.VkImage = undefined;
        if (c.vkCreateImage(device, &image_info, null, &image) != c.VK_SUCCESS) {
            return error.ImageCreationFailed;
        }
        errdefer c.vkDestroyImage(device, image, null);

        var mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetImageMemoryRequirements(device, image, &mem_requirements);

        const allocation = try pool.allocate(mem_requirements.size, mem_requirements.alignment);

        if (c.vkBindImageMemory(device, image, allocation.memory, allocation.offset) != c.VK_SUCCESS) {
            return error.MemoryBindFailed;
        }

        const view_info = c.VkImageViewCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = .{
                .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .flags = 0,
            .pNext = null,
        };

        var view: c.VkImageView = undefined;
        if (c.vkCreateImageView(device, &view_info, null, &view) != c.VK_SUCCESS) {
            return error.ImageViewCreationFailed;
        }

        return Image{
            .handle = image,
            .memory = allocation.memory,
            .view = view,
            .size = size,
            .format = format,
            .layout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .device = device,
            .pool = pool,
        };
    }

    pub fn deinit(self: *Image) void {
        c.vkDestroyImageView(self.device, self.view, null);
        c.vkDestroyImage(self.device, self.handle, null);
        if (self.pool == null) {
            c.vkFreeMemory(self.device, self.memory, null);
        }
    }

    pub fn transitionLayout(
        self: *Image,
        cmd: c.VkCommandBuffer,
        old_layout: c.VkImageLayout,
        new_layout: c.VkImageLayout,
    ) void {
        const barrier = c.VkImageMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .image = self.handle,
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcAccessMask = 0,
            .dstAccessMask = 0,
            .pNext = null,
        };

        var src_stage: c.VkPipelineStageFlags = undefined;
        var dst_stage: c.VkPipelineStageFlags = undefined;

        if (old_layout == c.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;
            src_stage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (old_layout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and new_layout == c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT;
            src_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst_stage = c.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            unreachable;
        }

        c.vkCmdPipelineBarrier(
            cmd,
            src_stage,
            dst_stage,
            0,
            0,
            null,
            0,
            null,
            1,
            &barrier,
        );

        self.layout = new_layout;
    }
};

pub const ResourceManager = struct {
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    buffer_pools: std.AutoHashMap(c.VkBufferUsageFlags, MemoryPool),
    image_pools: std.AutoHashMap(c.VkImageUsageFlags, MemoryPool),
    allocator: std.mem.Allocator,

    pub fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        allocator: std.mem.Allocator,
    ) !ResourceManager {
        return ResourceManager{
            .device = device,
            .physical_device = physical_device,
            .buffer_pools = std.AutoHashMap(c.VkBufferUsageFlags, MemoryPool).init(allocator),
            .image_pools = std.AutoHashMap(c.VkImageUsageFlags, MemoryPool).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ResourceManager) void {
        var buffer_pool_iter = self.buffer_pools.valueIterator();
        while (buffer_pool_iter.next()) |pool| {
            pool.deinit();
        }
        self.buffer_pools.deinit();

        var image_pool_iter = self.image_pools.valueIterator();
        while (image_pool_iter.next()) |pool| {
            pool.deinit();
        }
        self.image_pools.deinit();
    }

    pub fn getBufferPool(
        self: *ResourceManager,
        usage: c.VkBufferUsageFlags,
        properties: c.VkMemoryPropertyFlags,
    ) !*MemoryPool {
        const gop = try self.buffer_pools.getOrPut(usage);
        if (!gop.found_existing) {
            const pool = try MemoryPool.init(
                self.device,
                self.physical_device,
                properties,
                usage,
                0,
                self.allocator,
            );
            gop.value_ptr.* = pool;
        }
        return gop.value_ptr;
    }

    pub fn getImagePool(
        self: *ResourceManager,
        usage: c.VkImageUsageFlags,
        properties: c.VkMemoryPropertyFlags,
    ) !*MemoryPool {
        const gop = try self.image_pools.getOrPut(usage);
        if (!gop.found_existing) {
            const pool = try MemoryPool.init(
                self.device,
                self.physical_device,
                properties,
                usage,
                0,
                self.allocator,
            );
            gop.value_ptr.* = pool;
        }
        return gop.value_ptr;
    }
};

pub fn getMemoryStats() struct {
    total_memory_mb: f32,
    buffer_count: u32,
    image_count: u32,
} {
    return .{
        .total_memory_mb = @as(f32, @floatFromInt(total_allocated_memory)) / (1024 * 1024),
        .buffer_count = buffer_count,
        .image_count = image_count,
    };
}
